"""
PE Crop Visualizer

Takes a video with bounding box tracks, crops each tracked region from the
raw frames, encodes it using the *baseline* PE vision encoder (no learned
head), compares with text descriptions, and renders an annotated video.

Key difference from pe_activity_viz.py
---------------------------------------
pe_activity_viz.py  →  pre-saved patch tokens  →  cross-attention head  →  embedding
pe_crop_viz.py      →  crop bbox from frame    →  pe.encode_image(crop) →  embedding

No --features file and no --head-checkpoint are needed.  The full PE model
is loaded once and applied directly to each cropped region.

Pipeline
--------
    video + track   ──► crop bbox from frame  ──► pe.encode_image(crop)
                                                          │
                                                  per-frame feat [E]
                                                          │
                                                 mean-pool per window
                                                          │
    text descriptions ──► text encoder ──► text feats [Q, E]
                                                          │
                                               cosine similarity
                                                          │
                                          best label per identity
                                                          │
    original video   ──► annotated frames  ──► output video

Usage
-----
    # Full pipeline — annotated video output:
    python apps/pe/pe_crop_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --out           annotated.mp4

    # Custom activity labels:
    python apps/pe/pe_crop_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --out           annotated.mp4 \\
        --text "a child running" "a child walking" "a child reading a book"

    # Custom labels from file + scores overlay:
    python apps/pe/pe_crop_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --out           annotated.mp4 \\
        --text-file     queries.txt \\
        --scores-overlay

    # Scores table only — no video rendering:
    python apps/pe/pe_crop_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --no-video

Default activity labels (used when --text / --text-file are omitted):
  • a child running
  • a child walking
  • a child playing with blocks
  • a child playing with a house toy
  • a child reading a book
  • a child talking to friends
  • a child playing musical instruments
  • a child talking to a teacher
  • a child doing hand manipulation
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
# Default activity descriptions
# ---------------------------------------------------------------------------

DEFAULT_TEXTS = [
    "a child running",
    "a child walking",
    "a child playing with blocks",
    "a child playing with a house toy",
    "a child reading a book",
    "a child talking to friends",
    "a child playing musical instruments",
    "a child talking to a teacher",
    "a child doing hand manipulation",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Annotate a video with per-identity activity labels using "
                    "the baseline PE visual encoder (no learned head).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format track JSON: "
                        "{id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
    p.add_argument("--model", default="PE-Core-G14-448", metavar="NAME",
                   help="PE model name (default: PE-Core-G14-448).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    p.add_argument("--text", nargs="+", default=None, metavar="PHRASE",
                   help="Activity descriptions (default: 9 child-activity labels).")
    p.add_argument("--text-file", default=None, metavar="PATH",
                   help="Plain-text file with one activity description per line. "
                        "Overrides --text and the built-in defaults.")
    p.add_argument("--out", default="annotated.mp4", metavar="PATH",
                   help="Output annotated video path (default: annotated.mp4).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering — only print the scores table.")
    p.add_argument("--batch-size", type=int, default=16, metavar="N",
                   help="Crops per encoding batch (default: 16).")
    p.add_argument("--scores-overlay", action="store_true",
                   help="Draw a similarity bar chart in the video corner.")
    p.add_argument("--font-scale", type=float, default=0.55,
                   help="cv2 font scale for labels (default: 0.55).")
    p.add_argument("--window-sec", type=float, default=1.0, metavar="S",
                   help="Aggregate frames into windows of this many seconds "
                        "(default: 1).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Video frame rate — inferred from the video file when "
                        "omitted.")
    p.add_argument("--context-scale", type=float, default=2.0, metavar="S",
                   help="Expand bbox by this factor around its center before "
                        "cropping (default: 2.0 = twice the original size).")
    p.add_argument("--softmax", action="store_true",
                   help="Display softmax-normalised scores instead of raw "
                        "cosine similarities (makes relative differences clearer).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def _load_all_tracks(path: str) -> Dict[str, List]:
    """
    Load every identity from an identity-format track file.

    Returns
    -------
    dict mapping identity → list of [frame_idx, cx, cy, w, h]
        Entries are sorted by frame_idx.
        (cx, cy) are the bbox center in pixels; (w, h) are width/height.
    """
    with open(path) as f:
        data = json.load(f)
    return {
        identity: sorted(entries, key=lambda e: e[0])
        for identity, entries in data.items()
    }


def _build_frame_annotations(path: str) -> Dict[int, List]:
    """
    Build frame_index → list of (identity, x1, y1, x2, y2) for bbox drawing.
    Converts center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    """
    with open(path) as f:
        data = json.load(f)

    frame_anns: Dict[int, List] = defaultdict(list)
    for identity, entries in data.items():
        for entry in entries:
            frame_idx, cx, cy, w, h = entry
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            frame_anns[int(frame_idx)].append((identity, x1, y1, x2, y2))
    return dict(frame_anns)


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def _crop_bbox(
    pil_image: PILImage.Image,
    cx: float, cy: float, w: float, h: float,
    context_scale: float,
    frame_w: int, frame_h: int,
) -> PILImage.Image:
    """
    Crop a context-expanded bbox from a PIL image.

    The bbox (center_x, center_y, w, h) is enlarged by *context_scale* around
    its center before cropping.  Coordinates are clamped to the frame bounds.
    """
    ew = w * context_scale
    eh = h * context_scale
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(frame_w, int(cx + ew / 2))
    y2 = min(frame_h, int(cy + eh / 2))
    if x2 <= x1 or y2 <= y1:
        # Degenerate box — fall back to a 2×2 patch at center
        x1 = max(0, int(cx) - 1)
        y1 = max(0, int(cy) - 1)
        x2 = min(frame_w, x1 + 2)
        y2 = min(frame_h, y1 + 2)
    return pil_image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> Tuple[int, int, int]:
    """Deterministic, visually distinct BGR colour for an identity string."""
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = __import__("cv2").cvtColor(hsv, __import__("cv2").COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Drawing helpers  (identical API to pe_activity_viz.py)
# ---------------------------------------------------------------------------

def _draw_box_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    score: float,
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
) -> None:
    """Draw a filled-label bounding box on frame (in-place, BGR)."""
    import cv2
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}  {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, ft)
    pad = 3
    label_y2 = max(y1, th + 2 * pad)
    label_y1 = label_y2 - th - 2 * pad

    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 2 * pad, label_y2), color, -1)
    cv2.putText(
        frame, text,
        (x1 + pad, label_y2 - pad - baseline),
        font, font_scale, (255, 255, 255), ft, cv2.LINE_AA,
    )


def _draw_scores_overlay(
    frame: np.ndarray,
    texts: List[str],
    scores: List[float],
    title: str = "",
    x: int = 10,
    y: int = 10,
    font_scale: float = 0.45,
) -> None:
    """
    Draw a semi-transparent scores bar chart panel in the top-left corner.

    Each row:  [label text]  [████░░░░]  score_value
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    lh = int(cv2.getTextSize("A", font, font_scale, ft)[0][1] * 2.2)
    bar_w = 120
    col_w = max(cv2.getTextSize(t, font, font_scale, ft)[0][0] for t in texts) + 6
    panel_w = col_w + bar_w + 60 + 8
    rows = len(texts) + (1 if title else 0)
    panel_h = rows * lh + 8

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cy_pos = y + 6
    if title:
        cv2.putText(frame, title, (x + 4, cy_pos + lh - 4),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy_pos += lh

    max_score = max(scores) if scores else 1.0
    for t, score in zip(texts, scores):
        cv2.putText(frame, t, (x + 4, cy_pos + lh - 6),
                    font, font_scale, (200, 200, 200), ft, cv2.LINE_AA)
        filled = int(bar_w * score / max(max_score, 1e-6))
        bx = x + col_w
        cv2.rectangle(frame, (bx, cy_pos + 4), (bx + filled, cy_pos + lh - 4),
                      (100, 200, 100), -1)
        cv2.rectangle(frame, (bx, cy_pos + 4), (bx + bar_w, cy_pos + lh - 4),
                      (120, 120, 120), 1)
        sv = f"{score:.2f}"
        cv2.putText(frame, sv, (bx + bar_w + 4, cy_pos + lh - 6),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy_pos += lh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required.\n"
                 "Install with:  pip install opencv-python-headless")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_w, frame_h = args.image_size

    # Text labels
    if args.text_file is not None:
        tf_path = Path(args.text_file)
        if not tf_path.exists():
            sys.exit(f"--text-file not found: {tf_path}")
        texts = [ln.strip() for ln in tf_path.read_text().splitlines() if ln.strip()]
        if not texts:
            sys.exit(f"--text-file is empty: {tf_path}")
        print(f"Loaded {len(texts)} queries from {tf_path}")
    else:
        texts = args.text if args.text is not None else DEFAULT_TEXTS

    # ------------------------------------------------------------------
    # 1. Load PE model (visual + text encoder — no learned head)
    # ------------------------------------------------------------------
    print(f"Loading PE model ({args.model}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    pretrained = not args.no_pretrained
    pe_model = CLIP.from_config(
        args.model, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()

    enc_image_size: int = pe_model.visual.image_size
    img_transform = get_image_transform(enc_image_size)
    print(f"  PE encoder image size: {enc_image_size}px")

    # ------------------------------------------------------------------
    # 2. Encode text descriptions
    # ------------------------------------------------------------------
    tokenizer = get_text_tokenizer(pe_model.context_length)
    with torch.no_grad():
        text_feats = pe_model.encode_text(
            tokenizer(texts).to(device), normalize=True
        )  # [Q, E]
    print(f"  {len(texts)} text descriptions encoded.")

    # ------------------------------------------------------------------
    # 3. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    all_tracks = _load_all_tracks(args.track_file)
    print(f"  {len(all_tracks)} identities found.")

    # frame_idx → list of (identity, cx, cy, w, h)
    frame_to_crops: Dict[int, List] = defaultdict(list)
    for identity, entries in all_tracks.items():
        for entry in entries:
            frame_idx, cx, cy, w, h = entry
            frame_to_crops[int(frame_idx)].append((identity, cx, cy, w, h))

    # Probe video for FPS and frame count
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps: float = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    print(f"  fps={fps:.3f}  window={args.window_sec}s  ({window_frames} frames/window)")

    # ------------------------------------------------------------------
    # 4. Read video once — encode crops per identity per frame
    # ------------------------------------------------------------------
    # identity → {frame_idx: feature tensor [E]}
    identity_frame_feats: Dict[str, Dict[int, torch.Tensor]] = {
        identity: {} for identity in all_tracks
    }

    print(f"\nEncoding crops from {args.video} …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    frame_idx = 0
    batch_crops: List[torch.Tensor] = []
    batch_keys: List[Tuple[str, int]] = []   # (identity, frame_idx)

    def _flush_batch() -> None:
        if not batch_crops:
            return
        imgs = torch.stack(batch_crops).to(device)
        with torch.no_grad():
            feats = pe_model.encode_image(imgs, normalize=True)   # [B, E]
        for (id_, fidx), feat in zip(batch_keys, feats):
            identity_frame_feats[id_][fidx] = feat.cpu()
        batch_crops.clear()
        batch_keys.clear()

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx in frame_to_crops:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_frame = PILImage.fromarray(rgb)

            for (identity, cx, cy, w, h) in frame_to_crops[frame_idx]:
                crop = _crop_bbox(
                    pil_frame, cx, cy, w, h,
                    args.context_scale, frame_w, frame_h,
                )
                batch_crops.append(img_transform(crop))
                batch_keys.append((identity, frame_idx))

                if len(batch_crops) >= args.batch_size:
                    _flush_batch()

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == vid_total:
            print(f"\r  {frame_idx}/{vid_total} frames", end="", flush=True)

    _flush_batch()
    cap.release()
    print()

    # ------------------------------------------------------------------
    # 5. Windowed similarity — per identity
    # ------------------------------------------------------------------
    # identity → {"windows": {wid: {...}}, "color": (B,G,R), "window_frames": int}
    identity_info: Dict[str, Dict] = {}

    for identity, entries in all_tracks.items():
        frame_feats_map = identity_frame_feats[identity]
        if not frame_feats_map:
            print(f"  Skipping {identity}: no encoded frames.")
            continue

        # Group track frames into window buckets
        buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for entry in entries:
            fidx = int(entry[0])
            wid = fidx // window_frames
            if fidx in frame_feats_map:
                buckets[wid].append(frame_feats_map[fidx])

        windows: Dict[int, Dict] = {}
        for wid in sorted(buckets):
            feats_list = buckets[wid]
            w_mean = F.normalize(
                torch.stack(feats_list).mean(dim=0), dim=-1
            )  # [E]
            cos_sims = (w_mean.to(device) @ text_feats.T).cpu()   # [Q]
            display = F.softmax(cos_sims, dim=0) if args.softmax else cos_sims
            best_idx = int(display.argmax())
            start_fr = wid * window_frames
            end_fr = (wid + 1) * window_frames - 1

            windows[wid] = {
                "label":       texts[best_idx],
                "score":       display[best_idx].item(),
                "all_scores":  display.tolist(),
                "start_frame": start_fr,
                "end_frame":   end_fr,
                "start_sec":   start_fr / fps,
                "end_sec":     (end_fr + 1) / fps,
                "n_frames":    len(feats_list),
            }

        identity_info[identity] = {
            "windows":       windows,
            "color":         _identity_color(identity),
            "window_frames": window_frames,
        }

        for wid, w in sorted(windows.items()):
            print(f"  {identity:>10}  "
                  f"[{w['start_sec']:5.1f}s – {w['end_sec']:5.1f}s  "
                  f"{w['n_frames']:4d} frames]  "
                  f"→  {w['label']!r}  ({w['score']:.3f})")

    # ------------------------------------------------------------------
    # 6. Print scores table
    # ------------------------------------------------------------------
    col_w = 20
    header = f"  {'Identity':<10}  {'Window':<14}" + "".join(
        f"  {t[:col_w-2]:<{col_w}}" for t in texts
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for identity, info in identity_info.items():
        for wid, w in sorted(info["windows"].items()):
            time_str = f"{w['start_sec']:.1f}–{w['end_sec']:.1f}s"
            print(f"  {identity:<10}  {time_str:<14}", end="")
            for i, sc in enumerate(w["all_scores"]):
                marker = "◆" if i == w["all_scores"].index(max(w["all_scores"])) else " "
                print(f"  {marker}{sc:+.3f}{'':<{col_w-7}}", end="")
            print()
    print("─" * len(header))

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 7. Build frame → annotation lookup from raw track data
    # ------------------------------------------------------------------
    frame_annotations = _build_frame_annotations(args.track_file)

    # ------------------------------------------------------------------
    # 8. Read original video + render annotated output
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot re-open video: {args.video}")

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, vid_fps, (vid_w, vid_h))

    print(f"\nRendering {args.video}  →  {out_path}")
    print(f"  {vid_w}×{vid_h}  {vid_fps:.2f} fps  {vid_total} frames")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotations = frame_annotations.get(frame_idx, [])

        for identity, x1, y1, x2, y2 in annotations:
            if identity not in identity_info:
                continue
            info  = identity_info[identity]
            color = info["color"]

            # Find which window this frame belongs to
            wid  = frame_idx // info["window_frames"]
            wins = info["windows"]
            if wid not in wins:
                earlier = [k for k in wins if k <= wid]
                if not earlier:
                    continue
                wid = max(earlier)
            w = wins[wid]

            _draw_box_label(
                frame, x1, y1, x2, y2,
                label=w["label"],
                score=w["score"],
                color=color,
                font_scale=args.font_scale,
            )

            # Scores overlay for the first identity in the frame
            if args.scores_overlay and identity == annotations[0][0]:
                _draw_scores_overlay(
                    frame,
                    texts=texts,
                    scores=w["all_scores"],
                    title=f"{identity}  {w['start_sec']:.0f}–{w['end_sec']:.0f}s",
                    x=10, y=10,
                    font_scale=args.font_scale * 0.85,
                )

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == vid_total:
            print(f"\r  {frame_idx}/{vid_total} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")
