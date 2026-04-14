"""
PLM Description Generator + Visualizer  (generative inference)
==============================================================
For each tracked identity, generates a structured description covering:
  • Motion       — what the person's body is doing
  • Social       — how the person relates to others in the scene
  • Activity     — the high-level activity being performed

Each identity is processed as a **2-second video clip** (bbox crops stacked
into a short video) fed to the Perception Language Model (PLM) with a
structured text prompt.  PLM is a *generative* model trained with vision+LLM
— we use it by asking it to answer three questions about the clip and parsing
the free-form text response.

This is fundamentally different from cosine-similarity classification.  The
model sees the actual clip and is asked to describe it in natural language,
so the output is specific to each person rather than picking the same
"most common" category bucket across everyone.

Processing loop
---------------
  for each 2-second window (stride = window = 2 s):
      for each identity visible in this window:
          crops  = [crop_bbox(frame, bbox) for frame in window_frames]
          frames = VideoTransform(crops)        # (N, 3, H, W) tensor
          prompt = DESCRIPTION_PROMPT           # asks for Motion/Social/Activity
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
          "description":        "A person ..., ..., and ..."
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
        --image-size   1920 1080 \\
        --out          descriptions.json \\
        --out-video    overlay.mp4

    # Skip video rendering, process first 600 frames only:
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --image-size   1920 1080 \\
        --max-frames   600 \\
        --no-video \\
        --out          descriptions.json
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
# Structured description prompt sent to PLM for each 2-sec clip
# ---------------------------------------------------------------------------

DESCRIPTION_PROMPT = (
    "This is a short video clip of a single person. "
    "Answer the following questions concisely:\n"
    "Motion: <describe the person's body movement in 2-5 words>\n"
    "Social: <describe who the person is near or interacting with in 2-5 words>\n"
    "Activity: <describe what the person is doing in 2-5 words>"
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
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
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
    p.add_argument("--window-sec", type=float, default=2.0, metavar="S",
                   help="Clip duration in seconds (default: 2.0).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Override video FPS (inferred from file when omitted).")
    # --- crop ---
    p.add_argument("--context-scale", type=float, default=1.5, metavar="S",
                   help="Expand bbox before cropping (default: 1.5).")
    # --- runtime ---
    p.add_argument("--max-frames", type=int, default=None, metavar="N",
                   help="Stop after this many frames (default: all).")
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
# Crop helper
# ---------------------------------------------------------------------------

def _crop_bbox(
    pil_image: PILImage.Image,
    cx: float, cy: float, w: float, h: float,
    context_scale: float,
    frame_w: int, frame_h: int,
) -> PILImage.Image:
    ew, eh = w * context_scale, h * context_scale
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(frame_w, int(cx + ew / 2))
    y2 = min(frame_h, int(cy + eh / 2))
    if x2 <= x1 or y2 <= y1:
        x1 = max(0, int(cx) - 1)
        y1 = max(0, int(cy) - 1)
        x2 = min(frame_w, x1 + 2)
        y2 = min(frame_h, y1 + 2)
    return pil_image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# PLM response parsing
# ---------------------------------------------------------------------------

def _parse_plm_response(text: str) -> Tuple[str, str, str]:
    """
    Extract Motion / Social / Activity labels from the PLM's response text.

    Looks for lines starting with "Motion:", "Social:", "Activity:" (case-
    insensitive).  Falls back to "not determined" for any missing field.
    """
    motion = social = activity = "not determined"
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("motion:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                motion = val
        elif low.startswith("social:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                social = val
        elif low.startswith("activity:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                activity = val
    return motion, social, activity


# ---------------------------------------------------------------------------
# Description composer
# ---------------------------------------------------------------------------

def _compose_description(motion: str, social: str, activity: str,
                          nearby_ids: List[str]) -> str:
    s = social
    if nearby_ids:
        s += f" ({', '.join(nearby_ids)})"
    return f"A person {motion}, {s}, and {activity}."


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
    win: Optional[Dict],
    color: Tuple[int, int, int],
    font_scale: float,
) -> None:
    """
    Draw bbox + 3 description rows above it:

        ┌───────────────────────────────────────┐
        │ M: <motion>                           │  ← identity color
        │ S: <social> (id_2, id_5)              │  ← darker
        │ A: <activity>                         │  ← darkest
        └──── bbox ─────────────────────────────┘
    """
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if win is None:
        return

    motion   = win.get("motion", "")
    social   = win.get("social_interaction", {})
    activity = win.get("activity", "")
    nearby_ids: List[str] = social.get("nearby_ids", []) if isinstance(social, dict) else []
    social_lbl = social.get("label", "") if isinstance(social, dict) else str(social)
    if nearby_ids:
        social_lbl += f" ({', '.join(nearby_ids)})"

    rows = [
        (f"M: {motion}",   color),
        (f"S: {social_lbl}", _darken(color, 0.60)),
        (f"A: {activity}", _darken(color, 0.38)),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft   = 1
    pad  = 3
    (_, th), baseline = cv2.getTextSize("A", font, font_scale, ft)
    row_h = th + 2 * pad
    y_bottom = max(y1, len(rows) * row_h + 4)

    # Draw rows bottom-to-top: Activity closest to box, Motion at top
    for text, bg in reversed(rows):
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, ft)
        y_top = y_bottom - th - 2 * pad
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), bg, -1)
        cv2.putText(frame, text, (x1 + pad, y_bottom - pad - baseline),
                    font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)
        y_bottom = y_top


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python required.  pip install opencv-python-headless")

    frame_w, frame_h = args.image_size

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
    print(f"\nPrompt used for every clip:\n"
          f"  {DESCRIPTION_PROMPT!r}\n")

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
    max_frames    = args.max_frames or vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 4. Single-pass read — buffer crops per window, generate per window
    # ------------------------------------------------------------------
    # identity → {wid: result_dict}
    identity_windows: Dict[str, Dict[int, Dict]] = {ident: {} for ident in tracks}

    def _process_window(wid: int, wcrops: Dict[str, List[PILImage.Image]]) -> None:
        """Run PLM generation for every identity that has crops in this window."""
        start_fr  = wid * window_frames
        end_fr    = (wid + 1) * window_frames - 1
        start_sec = round(start_fr / fps, 3)
        end_sec   = round((end_fr + 1) / fps, 3)

        print(f"\n  window {wid}  [{start_sec:.1f}s – {end_sec:.1f}s]")

        for ident, crops in wcrops.items():
            if not crops:
                continue

            n = len(crops)
            # Uniform subsample to num_plm_frames
            if n > args.num_plm_frames:
                idxs = [int(round(i * (n - 1) / (args.num_plm_frames - 1)))
                        for i in range(args.num_plm_frames)]
                crops = [crops[i] for i in idxs]
            elif n == 1:
                crops = crops * 2   # PLM needs ≥ 2 frames

            # Convert PIL crops → (N, 3, H, W) tensor via VideoTransform
            frames_tensor, _ = plm_transform._process_multiple_images_pil(crops)

            # PLM generative inference with structured prompt
            responses, _, _ = generator.generate([(DESCRIPTION_PROMPT, frames_tensor)])
            raw = responses[0].strip()

            motion, social, activity = _parse_plm_response(raw)

            nearby_ids = _find_nearby_ids(
                ident, wid, window_frames, tracks,
                args.proximity_scale, max_frames,
            )

            description = _compose_description(motion, social, activity, nearby_ids)

            win = {
                "start_frame":        start_fr,
                "end_frame":          end_fr,
                "start_sec":          start_sec,
                "end_sec":            end_sec,
                "n_frames":           len(crops),
                "motion":             motion,
                "social_interaction": {
                    "label":      social,
                    "nearby_ids": nearby_ids,
                },
                "activity":           activity,
                "raw_response":       raw,
                "description":        description,
            }
            identity_windows[ident][wid] = win

            nearby_str = f"  nearby={nearby_ids}" if nearby_ids else ""
            print(f"    [{ident}]  M:{motion!r}  S:{social!r}  A:{activity!r}{nearby_str}")
            if "not determined" in (motion, social, activity):
                print(f"      raw → {raw!r}")

    print(f"\nReading {args.video} and running PLM on 2-sec crop clips …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    current_wid: int = -1
    window_crops: Dict[str, List[PILImage.Image]] = defaultdict(list)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        wid = frame_idx // window_frames

        # Window boundary crossed — process the completed window
        if wid != current_wid and current_wid >= 0:
            _process_window(current_wid, window_crops)
            window_crops = defaultdict(list)

        current_wid = wid

        if frame_idx in frame_map:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_frame = PILImage.fromarray(rgb)
            for (ident, cx, cy, w, h) in frame_map[frame_idx]:
                crop = _crop_bbox(pil_frame, cx, cy, w, h,
                                  args.context_scale, frame_w, frame_h)
                window_crops[ident].append(crop)

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  read {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    print()

    # Process last (possibly partial) window
    if current_wid >= 0 and any(window_crops.values()):
        _process_window(current_wid, window_crops)

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
                win_data, identity_colors[ident], args.font_scale,
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
