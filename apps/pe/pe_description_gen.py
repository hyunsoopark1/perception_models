"""
PLM Description Generator + Visualizer  (generative inference)
==============================================================
For each tracked identity, generates a structured description covering:
  • Motion       — what the person's body is doing
  • Social       — how the person relates to others in the scene
  • Activity     — the high-level activity being performed

Each identity is processed as a **video clip** (full frames from the
original video, sampled uniformly) fed to the Perception Language Model (PLM)
with a structured text prompt that includes the identity ID and bounding box
coordinates.  PLM is a *generative* model trained with vision+LLM — we use
it by asking it to answer three questions about the clip and parsing the
free-form text response.

This is fundamentally different from cosine-similarity classification.  The
model sees the full scene and is told which person to focus on, so the output
is specific to each person rather than picking the same "most common" category
bucket across everyone.

Processing loop
---------------
  for each window (stride = window = window_sec, default 6 s):
      for each identity visible in this window:
          frames = full video frames for this window   # (N, 3, H, W) tensor
          prompt = _make_prompt(identity_id)            # asks for Motion/Social/Activity
          response = PLM.generate([(prompt, frames)])
          motion, social, activity = parse(response)
          nearby_ids = spatial_proximity(tracks, identity, window)

Nearby detection
----------------
Uses bbox center distances from the track JSON — no extra video pass needed.
An identity is "nearby" when its center stays within proximity_scale × bbox_avg
for ≥ 30 % of co-visible frames in the same window.

Output JSON
-----------
    {
      "<identity_id>": [
        {
          "start_frame":        <int>,
          "end_frame":          <int>,
          "start_sec":          <float>,
          "end_sec":            <float>,
          "n_frames":           <int>,
          "motion":             "<free-form label>",
          "social_interaction": {
              "label":      "<free-form label>",
              "nearby_ids": ["id_2", "id_5"]
          },
          "activity":           "<free-form label>",
          "raw_response":       "<full PLM response text>",
          "description":        "<verbatim PLM output>"
        },
        ...
      ],
      ...
    }

Usage
-----
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --out          descriptions.json \\
        --out-video    overlay.mp4

    # Skip video rendering, process first 600 frames only:
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --max-frames   600 \\
        --no-video \\
        --out          descriptions.json

The full video frame is fed to PLM — no bbox cropping.  The bounding box
coordinates are passed in the text prompt so PLM knows which person to focus
on in each frame.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Prompt builder — per identity, per window
# ---------------------------------------------------------------------------

def _make_prompt(ident: str) -> str:
    """
    Build the PLM prompt for one identity + window.

    Each frame in the clip has a colored bounding box drawn directly onto
    the pixels marking the tracked person.  The prompt instructs the model
    to use that visible box as its spatial anchor rather than relying on
    numeric coordinates.
    """
    return (
        f"This is a video clip. In each frame a colored bounding box marks "
        f"a specific tracked person (ID '{ident}'). "
        f"Focus exclusively on the person inside the colored bounding box — "
        f"the box moves with the person across frames. "
        f"Answer the following questions about that person concisely. "
        f"If something is not clearly visible, respond with 'unclear' — do not guess.\n"
        "Motion: <this person's body movement in 2-5 words, or 'unclear'>\n"
        "Social: <who this person is near or interacting with in 2-5 words, or 'unclear'>\n"
        "Activity: <what this person is doing in 2-5 words, or 'unclear'>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLM generative description per identity per 2-sec window + video overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- inputs ---
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", default=None, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels (W H). "
                        "Inferred from the video file when omitted.")
    # --- model ---
    p.add_argument("--plm-ckpt", default="facebook/Perception-LM-8B", metavar="CKPT",
                   help="PLM checkpoint or HF model ID (default: facebook/Perception-LM-8B).")
    p.add_argument("--num-plm-frames", type=int, default=16, metavar="N",
                   help="Frames sampled per 2-sec clip for PLM (default: 16).")
    p.add_argument("--max-gen-len", type=int, default=120, metavar="N",
                   help="Max generated tokens per clip (default: 120).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (default: 0).")
    # --- windowing ---
    p.add_argument("--window-sec", type=float, default=6.0, metavar="S",
                   help="Clip duration in seconds (default: 6.0).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Override video FPS (inferred from file when omitted).")
    # --- runtime ---
    p.add_argument("--max-frames", type=int, default=None, metavar="N",
                   help="Stop after this many frames (default: all).")
    p.add_argument("--max-minutes", type=float, default=None, metavar="M",
                   help="Stop after this many minutes of video (default: all). "
                        "Overridden by --max-frames if both are given.")
    # --- proximity ---
    p.add_argument("--proximity-scale", type=float, default=2.0, metavar="S",
                   help="Nearby threshold = this × avg_bbox_dim (default: 2.0).")
    # --- output ---
    p.add_argument("--out", default="descriptions.json", metavar="PATH",
                   help="Output JSON path (default: descriptions.json).")
    p.add_argument("--out-video", default="description_overlay.mp4", metavar="PATH",
                   help="Output annotated video path (default: description_overlay.mp4).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering.")
    p.add_argument("--font-scale", type=float, default=0.42,
                   help="cv2 font scale for overlay labels (default: 0.42).")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print JSON (indent=2).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------

def _load_tracks(path: str) -> Dict[str, List]:
    with open(path) as f:
        data = json.load(f)
    return {
        ident: sorted(entries, key=lambda e: e[0])
        for ident, entries in data.items()
    }


def _build_frame_map(tracks: Dict[str, List]) -> Dict[int, List[Tuple]]:
    """frame_idx → [(identity, cx, cy, w, h)]"""
    fm: Dict[int, List] = defaultdict(list)
    for ident, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            fm[int(fidx)].append((ident, cx, cy, w, h))
    return dict(fm)


def _build_frame_annotations(tracks: Dict[str, List],
                              max_frames: int) -> Dict[int, List[Tuple]]:
    """frame_idx → [(identity, x1, y1, x2, y2)] corner format."""
    fa: Dict[int, List] = defaultdict(list)
    for ident, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = int(entry[0]), *entry[1:]
            if fidx >= max_frames:
                continue
            fa[fidx].append((ident,
                             int(cx - w / 2), int(cy - h / 2),
                             int(cx + w / 2), int(cy + h / 2)))
    return dict(fa)


# ---------------------------------------------------------------------------
# Proximity detection  (track data only, no video re-read)
# ---------------------------------------------------------------------------

def _find_nearby_ids(
    ident: str,
    wid: int,
    window_frames: int,
    tracks: Dict[str, List],
    proximity_scale: float,
    max_frames: int,
    min_close_ratio: float = 0.30,
) -> List[str]:
    frame_start = wid * window_frames
    frame_end   = (wid + 1) * window_frames

    target: Dict[int, Tuple] = {}
    for entry in tracks[ident]:
        fidx = int(entry[0])
        if frame_start <= fidx < frame_end and fidx < max_frames:
            target[fidx] = (float(entry[1]), float(entry[2]),
                            float(entry[3]), float(entry[4]))
    if not target:
        return []

    nearby: List[str] = []
    for other_id, other_entries in tracks.items():
        if other_id == ident:
            continue
        other: Dict[int, Tuple] = {}
        for entry in other_entries:
            fidx = int(entry[0])
            if fidx in target:
                other[fidx] = (float(entry[1]), float(entry[2]))

        co_visible = set(target.keys()) & set(other.keys())
        if not co_visible:
            continue

        close_count = sum(
            1 for fidx in co_visible
            if (
                (target[fidx][0] - other[fidx][0]) ** 2 +
                (target[fidx][1] - other[fidx][1]) ** 2
            ) ** 0.5 < proximity_scale * (target[fidx][2] + target[fidx][3]) / 2.0
        )
        if close_count / len(co_visible) >= min_close_ratio:
            nearby.append(other_id)

    return sorted(nearby)


# ---------------------------------------------------------------------------
# PLM response parsing
# ---------------------------------------------------------------------------

def _parse_plm_response(text: str) -> Tuple[str, str, str]:
    """
    Extract Motion / Social / Activity from the PLM's response.

    Robust to common PLM formatting variations:
      - "Motion: sitting still"
      - "**Motion:** sitting still"   (markdown bold)
      - "Motion - sitting still"      (dash separator)
      - lowercase / mixed case
    Returns empty string for any field the model didn't fill in.
    """
    motion = social = activity = ""
    for line in text.splitlines():
        # Strip markdown bold markers before matching
        stripped = line.strip().replace("**", "")
        low = stripped.lower()
        for prefix, field in (
            ("motion",   "motion"),
            ("social",   "social"),
            ("activity", "activity"),
        ):
            m = re.match(rf"^{prefix}\s*[:–\-]\s*", low)
            if m:
                val = stripped[m.end():].strip()
                # Discard unanswered template placeholders like <...>
                if re.fullmatch(r"<[^>]*>", val):
                    val = ""
                if field == "motion":
                    motion = val
                elif field == "social":
                    social = val
                elif field == "activity":
                    activity = val
                break
    return motion, social, activity


# ---------------------------------------------------------------------------
# Description composer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-frame bbox annotation
# ---------------------------------------------------------------------------

def _annotate_frame(
    pil_frame: PILImage.Image,
    cx: float, cy: float, w: float, h: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> PILImage.Image:
    """
    Draw the tracked person's bbox onto a copy of the frame (RGB).

    This embeds the spatial location directly into the pixels so PLM's
    visual attention can follow the moving box across frames, rather than
    relying on a single static bbox coordinate in the text prompt.
    """
    import cv2
    frame = np.array(pil_frame)          # H×W×3 uint8, RGB
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(frame.shape[1], int(cx + w / 2))
    y2 = min(frame.shape[0], int(cy + h / 2))
    # cv2 expects BGR but we keep the array as RGB — swap color channels
    bgr_color = (color[2], color[1], color[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, thickness)
    return PILImage.fromarray(frame)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> Tuple[int, int, int]:
    import cv2
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _darken(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Visualization drawing
# ---------------------------------------------------------------------------

def _draw_window_overlay(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ident: str,
    win: Optional[Dict],
    color: Tuple[int, int, int],
    font_scale: float,
) -> None:
    """
    Draw bbox with identity badge + M/S/A rows above it:

        M: <motion>                    ← topmost, identity color
        S: <social> (id_2, id_5)       ← darker
        A: <activity>                  ← darkest
        ┌[id]──── bbox ───────────────┐
        │                             │
        └─────────────────────────────┘
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft   = 1
    pad  = 3
    MAX_CHARS = 70   # truncate long text to prevent horizontal overflow
    (_, th), baseline = cv2.getTextSize("A", font, font_scale, ft)

    def _draw_row(y_bottom: int, text: str, bg: Tuple) -> int:
        """Draw a filled label row; returns the top y of the drawn row."""
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, ft)
        y_top = y_bottom - th - 2 * pad
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), bg, -1)
        cv2.putText(frame, text, (x1 + pad, y_bottom - pad - baseline),
                    font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)
        return y_top

    # --- bounding box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # --- identity badge pinned to top-left corner of box ---
    id_tag = f"[{ident}]"
    (tw, _), _ = cv2.getTextSize(id_tag, font, font_scale, ft)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 2 * pad, y1 + th + 2 * pad), color, -1)
    cv2.putText(frame, id_tag, (x1 + pad, y1 + th + pad - baseline),
                font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)

    if win is None:
        return

    # --- collect field values ---
    motion      = win.get("motion", "")
    social_dict = win.get("social_interaction", {})
    activity    = win.get("activity", "")
    nearby_ids: List[str] = (social_dict.get("nearby_ids", [])
                             if isinstance(social_dict, dict) else [])
    social_lbl  = (social_dict.get("label", "")
                   if isinstance(social_dict, dict) else str(social_dict))
    if nearby_ids:
        social_lbl += f" ({', '.join(nearby_ids)})"

    # Build rows bottom → top above the box.
    # First item ends up closest to the box; last item is topmost.
    rows: List[Tuple[str, Tuple]] = []
    if activity:
        rows.append((f"A: {activity[:MAX_CHARS]}", _darken(color, 0.38)))
    if social_lbl:
        rows.append((f"S: {social_lbl[:MAX_CHARS]}", _darken(color, 0.60)))
    if motion:
        rows.append((f"M: {motion[:MAX_CHARS]}", color))

    if not rows:
        return

    y_bottom = max(y1, len(rows) * (th + 2 * pad) + 4)
    for text, bg in rows:
        y_bottom = _draw_row(y_bottom, text, bg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python required.  pip install opencv-python-headless")

    # ------------------------------------------------------------------
    # 0. Resolve frame size (auto-detect from video when not supplied)
    # ------------------------------------------------------------------
    if args.image_size is not None:
        frame_w, frame_h = args.image_size
    else:
        _probe = cv2.VideoCapture(args.video)
        if not _probe.isOpened():
            sys.exit(f"Cannot open video to detect resolution: {args.video}")
        frame_w = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _probe.release()
        if frame_w == 0 or frame_h == 0:
            sys.exit("Could not read frame size from video. "
                     "Supply it manually with --image-size W H.")
    print(f"Frame size: {frame_w}×{frame_h}")

    # ------------------------------------------------------------------
    # 1. Load PLM + set up generator
    # ------------------------------------------------------------------
    print(f"Loading PLM from {args.plm_ckpt} …")
    from apps.plm.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
        load_consolidated_model_and_tokenizer,
    )
    from core.transforms.video_transform import get_video_transform

    plm_model, plm_tokenizer, plm_config = load_consolidated_model_and_tokenizer(
        args.plm_ckpt
    )
    plm_transform = get_video_transform(image_res=plm_model.vision_model.image_size)
    print(f"  vision input size: {plm_model.vision_model.image_size}px")

    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
        until=[],           # let the model generate all 3 lines
        dtype="bf16",
        device="cuda",
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, plm_model, plm_tokenizer)

    print(f"  generator ready  (max_gen_len={args.max_gen_len}, "
          f"temperature={args.temperature})")
    print(f"\nPrompt template per clip includes: identity ID, bbox (x1,y1,x2,y2) "
          f"in the original frame, and Motion/Social/Activity questions.\n")

    # ------------------------------------------------------------------
    # 2. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    tracks    = _load_tracks(args.track_file)
    frame_map = _build_frame_map(tracks)
    print(f"  {len(tracks)} identities, "
          f"{sum(len(v) for v in tracks.values())} track entries")

    # ------------------------------------------------------------------
    # 3. Probe video
    # ------------------------------------------------------------------
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total     = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps           = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    if args.max_frames is not None:
        max_frames = args.max_frames
    elif args.max_minutes is not None:
        max_frames = int(args.max_minutes * 60 * fps)
    else:
        max_frames = vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 4. Single-pass read — buffer full frames per window, generate per window
    #
    # Each unique video frame is stored once (shared across all identities
    # visible in that frame).  Per-identity we only keep bbox coordinates.
    # ------------------------------------------------------------------
    # identity → {wid: result_dict}
    identity_windows: Dict[str, Dict[int, Dict]] = {ident: {} for ident in tracks}

    def _process_window(
        wid: int,
        frame_cache: Dict[int, PILImage.Image],          # frame_idx → full PIL frame
        bbox_map_w:  Dict[str, List[Tuple]],             # identity → [(fidx,cx,cy,w,h)]
    ) -> None:
        """Run PLM generation for every identity visible in window *wid*."""
        start_fr  = wid * window_frames
        end_fr    = (wid + 1) * window_frames - 1
        start_sec = round(start_fr / fps, 3)
        end_sec   = round((end_fr + 1) / fps, 3)

        print(f"\n  window {wid}  [{start_sec:.1f}s – {end_sec:.1f}s]")

        for ident, bbox_entries in bbox_map_w.items():
            if not bbox_entries:
                continue

            # Collect full frames (in frame order, from the shared cache)
            frame_indices = [e[0] for e in bbox_entries]
            pil_frames    = [frame_cache[fidx] for fidx in frame_indices]
            bbox_coords   = [(e[1], e[2], e[3], e[4]) for e in bbox_entries]

            # Uniform subsample to num_plm_frames (keep bbox_coords in sync)
            n = len(pil_frames)
            if n > args.num_plm_frames:
                idxs = [int(round(i * (n - 1) / (args.num_plm_frames - 1)))
                        for i in range(args.num_plm_frames)]
                pil_frames  = [pil_frames[i]  for i in idxs]
                bbox_coords = [bbox_coords[i] for i in idxs]
            elif n == 1:
                pil_frames  = pil_frames  * 2   # PLM needs ≥ 2 frames
                bbox_coords = bbox_coords * 2

            # Prompt tells the model to attend to the drawn colored bbox
            prompt = _make_prompt(ident)

            # Annotate each frame with its own per-frame bbox so PLM's visual
            # attention can follow the moving person across the clip.
            ann_color = _identity_color(ident)
            annotated = [
                _annotate_frame(f, cx, cy, w, h, color=ann_color)
                for f, (cx, cy, w, h) in zip(pil_frames, bbox_coords)
            ]

            # Annotated frames → (N, 3, H, W) tensor
            frames_tensor, _ = plm_transform._process_multiple_images_pil(annotated)

            # PLM generative inference
            responses, _, _ = generator.generate([(prompt, frames_tensor)])
            raw = responses[0].strip()

            motion, social, activity = _parse_plm_response(raw)

            nearby_ids = _find_nearby_ids(
                ident, wid, window_frames, tracks,
                args.proximity_scale, max_frames,
            )

            win = {
                "start_frame":        start_fr,
                "end_frame":          end_fr,
                "start_sec":          start_sec,
                "end_sec":            end_sec,
                "n_frames":           len(pil_frames),
                "motion":             motion,
                "social_interaction": {
                    "label":      social,
                    "nearby_ids": nearby_ids,
                },
                "activity":           activity,
                "description":        raw,
            }
            identity_windows[ident][wid] = win

            nearby_str = f"  nearby={nearby_ids}" if nearby_ids else ""
            print(f"    [{ident}]  M:{motion!r}  S:{social!r}  A:{activity!r}{nearby_str}")
            if not all([motion, social, activity]):
                print(f"      raw → {raw!r}")

    print(f"\nReading {args.video} and running PLM on 2-sec full-frame clips …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    current_wid: int = -1
    # Shared full-frame cache for current window (frame_idx → PIL image)
    win_frame_cache: Dict[int, PILImage.Image] = {}
    # Per-identity bbox entries for current window: [(frame_idx, cx, cy, w, h)]
    win_bbox_map: Dict[str, List] = defaultdict(list)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        wid = frame_idx // window_frames

        # Window boundary crossed — process the completed window then clear buffers
        if wid != current_wid and current_wid >= 0:
            _process_window(current_wid, win_frame_cache, win_bbox_map)
            win_frame_cache = {}
            win_bbox_map    = defaultdict(list)

        current_wid = wid

        if frame_idx in frame_map:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Store the full frame once, shared across all identities in this frame
            pil_frame = PILImage.fromarray(rgb)
            win_frame_cache[frame_idx] = pil_frame
            for (ident, cx, cy, w, h) in frame_map[frame_idx]:
                win_bbox_map[ident].append((frame_idx, cx, cy, w, h))

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  read {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    print()

    # Process last (possibly partial) window
    if current_wid >= 0 and win_frame_cache:
        _process_window(current_wid, win_frame_cache, win_bbox_map)

    # ------------------------------------------------------------------
    # 5. Build results + write JSON
    # ------------------------------------------------------------------
    results: Dict[str, List[Dict]] = {
        ident: [win for _, win in sorted(identity_windows[ident].items())]
        for ident in tracks
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=(2 if args.pretty else None), ensure_ascii=False)
    print(f"\nSaved JSON → {out_path}  ({len(results)} identities)")

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 6. Render annotated video  (second pass — no GPU work)
    # ------------------------------------------------------------------
    print(f"\nRendering overlay video …")
    frame_anns = _build_frame_annotations(tracks, max_frames)
    identity_colors = {ident: _identity_color(ident) for ident in tracks}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot re-open video: {args.video}")

    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = Path(args.out_video)
    out_vid.parent.mkdir(parents=True, exist_ok=True)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(out_vid), fourcc, fps, (vid_w, vid_h))

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        for ident, x1, y1, x2, y2 in frame_anns.get(frame_idx, []):
            wid  = frame_idx // window_frames
            wins = identity_windows.get(ident, {})
            if wid not in wins:
                earlier = [k for k in wins if k <= wid]
                wid = max(earlier) if earlier else None  # type: ignore[assignment]
            win_data = wins.get(wid) if wid is not None else None  # type: ignore[arg-type]

            _draw_window_overlay(
                frame, x1, y1, x2, y2,
                ident, win_data, identity_colors[ident], args.font_scale,
            )

        # Timestamp watermark
        cv2.putText(frame, f"t={frame_idx / fps:.1f}s", (8, vid_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  render {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved video → {out_vid}")
