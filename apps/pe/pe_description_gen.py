"""
PLM Description Generator + Visualizer
=======================================
For each tracked identity, generates a structured description covering:
  • Motion       — what the person's body is doing
  • Social       — how the person relates to others in the scene
  • Activity     — the high-level activity being performed

The encoder is the **Perception Language Model (PLM)**, which processes a
2-second video clip per identity per window.  This is strictly better than
encoding single frames because the LLM backbone can reason about temporal
patterns (movement, interaction dynamics) across the clip.

Processing loop
---------------
  for each 2-second window (stride = window = 2 s):
      for each identity visible in this window:
          crops  = [crop_bbox(frame, bbox) for frame in window_frames]
          feat   = PLM.encode_video(crops, num_frames=16)
          motion, social, activity = argmax(cosine_sim(feat, text_labels))
          nearby_ids = spatial_proximity(tracks, this_identity, this_window)

Both the JSON and the annotated video include timestamps for every window.

Proximity detection
-------------------
Within each window, other identities whose bbox centers stay within
  proximity_scale × (w + h) / 2
of the target for ≥ 30 % of co-visible frames are reported as nearby_ids.
No extra video pass is needed — all positions come from the track JSON.

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
          "motion":             {"label": "...", "score": <float>},
          "social_interaction": {
              "label":      "...",
              "score":      <float>,
              "nearby_ids": ["id_2", "id_5"]
          },
          "activity":           {"label": "...", "score": <float>},
          "description":        "A person walking, interacting with id_2, and playing."
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

    # Limit to first 5 minutes, skip video rendering:
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --image-size   1920 1080 \\
        --max-frames   9000 \\
        --no-video \\
        --out          descriptions.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Category text candidates
# ---------------------------------------------------------------------------

MOTION_LABELS: List[str] = [
    "a person standing still",
    "a person walking",
    "a person running",
    "a person sitting",
    "a person crouching or squatting",
    "a person jumping",
    "a person lying down",
    "a person crawling",
    "a person dancing or moving rhythmically",
    "a person making hand gestures",
]

SOCIAL_LABELS: List[str] = [
    "a person alone with no one nearby",
    "a person talking face to face with one other person",
    "a person interacting with a small group of peers",
    "a person in a large group",
    "a person interacting with an adult or teacher",
    "a person sitting or standing side by side with others",
    "a person watching others from a distance",
]

ACTIVITY_LABELS: List[str] = [
    "a person reading a book or document",
    "a person playing with toys or objects",
    "a person drawing writing or doing craftwork",
    "a person eating or drinking",
    "a person building or assembling something",
    "a person using a phone computer or electronic device",
    "a person exercising or doing physical activity",
    "a person talking or having a conversation",
    "a person exploring or looking around",
    "a person resting or doing nothing",
    "a person playing a musical instrument",
    "a person cleaning or tidying up",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLM-based per-identity description generator + video visualizer.",
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
                   help="Frames uniformly sampled per 2-sec clip for PLM (default: 16).")
    p.add_argument("--pool", default="mean", choices=["mean", "mean_all", "last"],
                   help="PLM hidden-state pooling strategy (default: mean).")
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
    p.add_argument("--softmax", action="store_true",
                   help="Use softmax-normalised scores instead of raw cosine sims.")
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
    """frame_idx → [(identity, x1, y1, x2, y2)] in corner format."""
    fa: Dict[int, List] = defaultdict(list)
    for ident, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            fidx = int(fidx)
            if fidx >= max_frames:
                continue
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            fa[fidx].append((ident, x1, y1, x2, y2))
    return dict(fa)


# ---------------------------------------------------------------------------
# Proximity detection (track-data only, no video re-read)
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
    """Return sorted list of identities spatially close to *ident* in window *wid*."""
    frame_start = wid * window_frames
    frame_end   = (wid + 1) * window_frames

    # Build frame → (cx, cy, w, h) for the target within this window
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
    ew = w * context_scale
    eh = h * context_scale
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
# PLM video encoding from PIL frames (no temp file needed)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _encode_frames_plm(
    model,
    tokenizer,
    transform,
    pil_frames: List[PILImage.Image],
    num_frames: int = 16,
    pool: str = "mean",
) -> Optional[torch.Tensor]:
    """
    Encode a list of PIL images as a video clip through PLM.

    Uniformly subsamples to *num_frames*, converts through VideoTransform,
    then runs the full PLM forward pass (vision encoder + LLM layers).

    Returns a normalised embedding tensor [dim] on the model's device,
    or None if the frame list is empty.
    """
    if not pil_frames:
        return None

    n = len(pil_frames)
    if n > num_frames:
        # Uniform subsample
        idxs = [int(round(i * (n - 1) / (num_frames - 1))) for i in range(num_frames)]
        pil_frames = [pil_frames[i] for i in idxs]
    elif n == 1:
        pil_frames = pil_frames * 2  # PLM needs ≥ 2 frames

    # transform._process_multiple_images_pil → (N, 3, H, W) float32 on CPU
    frames, _ = transform._process_multiple_images_pil(pil_frames)

    param = next(model.parameters())
    dev, dtype = param.device, param.dtype

    # Build token sequence with image placeholders
    text_ids, image_pos = tokenizer._tokenize_for_generation("", frames)
    token_values = torch.tensor([text_ids], dtype=torch.long, device=dev)
    image_pos_index = torch.full(token_values.shape, -1, dtype=torch.int, device=dev)
    image_pos_index[0, image_pos] = torch.arange(len(image_pos), dtype=torch.int, device=dev)

    images = frames.to(device=dev, dtype=dtype)

    # Full PLM forward — return hidden states before LM head
    _, hidden = model.forward(
        token_values,
        images=images,
        image_pos_index=image_pos_index,
        num_chunks=[frames.size(0)],
        media_type=["video"],
        return_hidden_states=True,
    )  # hidden: (1, seqlen, dim)

    if pool == "mean":
        img_pos_t = torch.tensor(image_pos, device=hidden.device)
        embedding = hidden[0, img_pos_t].mean(dim=0)
    elif pool == "mean_all":
        embedding = hidden[0].mean(dim=0)
    else:  # last
        embedding = hidden[0, -1, :]

    return F.normalize(embedding, dim=-1)


# ---------------------------------------------------------------------------
# Best-label selection
# ---------------------------------------------------------------------------

def _best_match(
    feat: torch.Tensor,        # [dim] on any device
    text_feats: torch.Tensor,  # [N, dim] on same device
    labels: List[str],
    use_softmax: bool,
) -> Tuple[str, float]:
    cos = (feat @ text_feats.T).cpu()
    display = F.softmax(cos, dim=0) if use_softmax else cos
    best = int(display.argmax())
    return labels[best], float(display[best])


# ---------------------------------------------------------------------------
# Description composer
# ---------------------------------------------------------------------------

def _compose_description(
    motion: str, social: str, activity: str, nearby_ids: List[str]
) -> str:
    def _strip(label: str) -> str:
        for pfx in ("a person ", "a child "):
            if label.lower().startswith(pfx):
                return label[len(pfx):]
        return label

    m, s, a = _strip(motion), _strip(social), _strip(activity)
    if nearby_ids:
        s += f" ({', '.join(nearby_ids)})"
    return f"A person {m}, {s}, and {a}."


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
    Draw a bounding box with three description rows above it:

        ┌─────────────────────────────────────────────┐
        │ M: <motion>  score                          │ ← identity color
        │ S: <social> (id_2, id_5)  score             │ ← darker
        │ A: <activity>  score                        │ ← darkest
        └──── bbox ───────────────────────────────────┘
    """
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if win is None:
        return

    motion = win["motion"]
    social = win["social_interaction"]
    activity = win["activity"]
    nearby_ids: List[str] = social.get("nearby_ids", [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    pad = 3
    (_, th), baseline = cv2.getTextSize("A", font, font_scale, ft)
    row_h = th + 2 * pad

    color_m = color
    color_s = _darken(color, 0.60)
    color_a = _darken(color, 0.38)

    def _draw_row(y_bottom: int, text: str, bg: Tuple) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, ft)
        y_top = y_bottom - th - 2 * pad
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), bg, -1)
        cv2.putText(frame, text, (x1 + pad, y_bottom - pad - baseline),
                    font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)
        return y_top

    # Reserve headroom above bbox (3 rows)
    y_bottom = max(y1, 3 * row_h + 4)

    # Activity — closest to box
    a_lbl = activity["label"].replace("a person ", "")
    y_bottom = _draw_row(y_bottom, f"A: {a_lbl}  {activity['score']:.2f}", color_a)

    # Social — with nearby IDs if any
    s_lbl = social["label"].replace("a person ", "")
    if nearby_ids:
        s_lbl += f" ({', '.join(nearby_ids)})"
    y_bottom = _draw_row(y_bottom, f"S: {s_lbl}  {social['score']:.2f}", color_s)

    # Motion — topmost
    m_lbl = motion["label"].replace("a person ", "")
    _draw_row(y_bottom, f"M: {m_lbl}  {motion['score']:.2f}", color_m)


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
    # 1. Load PLM
    # ------------------------------------------------------------------
    print(f"Loading PLM from {args.plm_ckpt} …")
    from apps.plm.generate import load_consolidated_model_and_tokenizer
    from core.transforms.video_transform import get_video_transform
    from extract_plm_features import encode_text

    plm_model, plm_tokenizer, _ = load_consolidated_model_and_tokenizer(args.plm_ckpt)
    plm_transform = get_video_transform(image_res=plm_model.vision_model.image_size)
    print(f"  vision input size: {plm_model.vision_model.image_size}px")

    # ------------------------------------------------------------------
    # 2. Encode category text labels
    # ------------------------------------------------------------------
    print("Encoding category labels …")
    motion_feats   = torch.stack([encode_text(plm_model, plm_tokenizer, t, args.pool)
                                  for t in MOTION_LABELS])    # [N, dim] cuda
    social_feats   = torch.stack([encode_text(plm_model, plm_tokenizer, t, args.pool)
                                  for t in SOCIAL_LABELS])
    activity_feats = torch.stack([encode_text(plm_model, plm_tokenizer, t, args.pool)
                                  for t in ACTIVITY_LABELS])
    print(f"  motion {len(MOTION_LABELS)} | social {len(SOCIAL_LABELS)} "
          f"| activity {len(ACTIVITY_LABELS)}")

    # ------------------------------------------------------------------
    # 3. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    tracks   = _load_tracks(args.track_file)
    frame_map = _build_frame_map(tracks)
    print(f"  {len(tracks)} identities, "
          f"{sum(len(v) for v in tracks.values())} track entries")

    # ------------------------------------------------------------------
    # 4. Probe video
    # ------------------------------------------------------------------
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total     = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps          = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    max_frames   = args.max_frames or vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 5. Single-pass video read — stream crops window by window
    #
    # Memory strategy: buffer ONE window at a time.  When the window ID
    # changes, encode + classify + clear before continuing.
    # ------------------------------------------------------------------
    # identity → {wid: window_result_dict}
    identity_windows: Dict[str, Dict[int, Dict]] = {ident: {} for ident in tracks}

    def _process_window(wid: int, wcrops: Dict[str, List[PILImage.Image]]) -> None:
        """Encode all accumulated crops for *wid* and store results."""
        start_fr = wid * window_frames
        end_fr   = (wid + 1) * window_frames - 1
        start_sec = round(start_fr / fps, 3)
        end_sec   = round((end_fr + 1) / fps, 3)

        print(f"\n  window {wid}  [{start_sec:.1f}s – {end_sec:.1f}s]")

        for ident, crops in wcrops.items():
            if not crops:
                continue

            embedding = _encode_frames_plm(
                plm_model, plm_tokenizer, plm_transform,
                crops, args.num_plm_frames, args.pool,
            )
            if embedding is None:
                continue

            motion_lbl,   motion_sc   = _best_match(embedding, motion_feats,
                                                     MOTION_LABELS, args.softmax)
            social_lbl,   social_sc   = _best_match(embedding, social_feats,
                                                     SOCIAL_LABELS, args.softmax)
            activity_lbl, activity_sc = _best_match(embedding, activity_feats,
                                                     ACTIVITY_LABELS, args.softmax)

            nearby_ids = _find_nearby_ids(
                ident, wid, window_frames, tracks,
                args.proximity_scale, max_frames,
            )

            description = _compose_description(
                motion_lbl, social_lbl, activity_lbl, nearby_ids
            )

            win = {
                "start_frame":        start_fr,
                "end_frame":          end_fr,
                "start_sec":          start_sec,
                "end_sec":            end_sec,
                "n_frames":           len(crops),
                "motion": {
                    "label": motion_lbl,
                    "score": round(motion_sc, 4),
                },
                "social_interaction": {
                    "label":      social_lbl,
                    "score":      round(social_sc, 4),
                    "nearby_ids": nearby_ids,
                },
                "activity": {
                    "label": activity_lbl,
                    "score": round(activity_sc, 4),
                },
                "description": description,
            }
            identity_windows[ident][wid] = win

            nearby_str = f" → nearby: {nearby_ids}" if nearby_ids else ""
            print(f"    [{ident}]  M: {motion_lbl!r}  "
                  f"S: {social_lbl!r}{nearby_str}  "
                  f"A: {activity_lbl!r}")

    print(f"\nReading {args.video} and encoding 2-sec crops …")
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

    # Process last (possibly incomplete) window
    if current_wid >= 0 and any(window_crops.values()):
        _process_window(current_wid, window_crops)

    # ------------------------------------------------------------------
    # 6. Build results list + write JSON
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
    # 7. Render annotated video (second pass — no GPU work)
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

            # Use the current window, or the most recent earlier one
            if wid not in wins:
                earlier = [k for k in wins if k <= wid]
                wid = max(earlier) if earlier else None  # type: ignore[assignment]

            win_data = wins.get(wid) if wid is not None else None  # type: ignore[arg-type]

            _draw_window_overlay(
                frame, x1, y1, x2, y2,
                win_data,
                identity_colors[ident],
                args.font_scale,
            )

        # Timestamp watermark
        ts = f"t={frame_idx / fps:.1f}s"
        cv2.putText(frame, ts, (8, vid_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  render {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved video → {out_vid}")
