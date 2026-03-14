"""
PE Combined Visualizer

Runs *both* methods on the same video and overlays their predictions on each
tracked bounding box side-by-side in a single output video.

Methods
-------
  H (Head)  : patch tokens from --features  ──► cross-attention head ──► embedding
  C (Crop)  : crop bbox from raw video frame ──► pe.encode_image(crop) ──► embedding

Both embeddings are compared with the same text descriptions via cosine
similarity.  For each tracked identity and time window the video shows:

    ┌──────────────────────────────┐
    │ H: <head label>    score     │   ← identity colour
    │ C: <crop label>    score     │   ← darker shade of same colour
    └──────────────────────────────┘
    ┌ ─ ─ ─ ─  expanded bbox  ─ ─ ┐
    │                              │
    │         tracked person       │
    │                              │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘

With --scores-overlay two bar-chart panels appear in the top-left corner,
one per method.

Bounding boxes are drawn at 2× the raw tracker size (--context-scale 4.0
doubles the area relative to the previous default of 2.0).

Usage
-----
    python apps/pe/pe_combined_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --features      patch_features.pt \\
        --head-checkpoint head.pt \\
        --out           annotated_combined.mp4

    # Custom labels + scores overlay:
    python apps/pe/pe_combined_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --features      patch_features.pt \\
        --head-checkpoint head.pt \\
        --out           annotated_combined.mp4 \\
        --text "a child running" "a child walking" "a child reading a book" \\
        --scores-overlay

    # Labels from file, scores table only (no video):
    python apps/pe/pe_combined_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --features      patch_features.pt \\
        --head-checkpoint head.pt \\
        --text-file     queries.txt \\
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
        description="Overlay predictions from both the cross-attention head and "
                    "the baseline PE crop encoder on a single annotated video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- inputs ---
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format track JSON: "
                        "{id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) from "
                        "pe_extract_patch_features.py  (used by the head method).")
    # --- head method ---
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention weights (.pt).")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Cross-attention heads — must match training (default: 8).")
    # --- PE model ---
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name (inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    # --- text labels ---
    p.add_argument("--text", nargs="+", default=None, metavar="PHRASE",
                   help="Activity descriptions (default: 9 child-activity labels).")
    p.add_argument("--text-file", default=None, metavar="PATH",
                   help="Plain-text file with one activity description per line. "
                        "Overrides --text and the built-in defaults.")
    # --- output ---
    p.add_argument("--out", default="annotated_combined.mp4", metavar="PATH",
                   help="Output annotated video path (default: annotated_combined.mp4).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering — only print the scores table.")
    # --- runtime ---
    p.add_argument("--batch-size", type=int, default=16, metavar="N",
                   help="Frames per encoding batch (default: 16).")
    p.add_argument("--window-sec", type=float, default=1.0, metavar="S",
                   help="Window length in seconds for temporal aggregation (default: 1).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Frame rate — inferred from video/features when omitted.")
    p.add_argument("--context-scale", type=float, default=4.0, metavar="S",
                   help="Expand bbox by this factor for feature extraction AND drawing "
                        "(default: 4.0 — twice the area of the previous default 2.0).")
    p.add_argument("--softmax", action="store_true",
                   help="Display softmax-normalised scores instead of raw cosine "
                        "similarities.")
    # --- visual ---
    p.add_argument("--scores-overlay", action="store_true",
                   help="Draw dual similarity bar-chart panels in the video corner.")
    p.add_argument("--font-scale", type=float, default=0.55,
                   help="cv2 font scale for labels (default: 0.55).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track / annotation helpers
# ---------------------------------------------------------------------------

def _load_all_tracks(path: str) -> Dict[str, List]:
    """Return identity → sorted list of [frame_idx, cx, cy, w, h]."""
    with open(path) as f:
        data = json.load(f)
    return {
        identity: sorted(entries, key=lambda e: e[0])
        for identity, entries in data.items()
    }


def _build_frame_annotations(
    path: str,
    draw_scale: float,
    frame_w: int,
    frame_h: int,
) -> Dict[int, List]:
    """
    frame_index → list of (identity, x1, y1, x2, y2) for bbox drawing.

    The bbox is expanded by *draw_scale* around its center so the rendered
    box matches the region that was used for feature extraction.
    """
    with open(path) as f:
        data = json.load(f)

    frame_anns: Dict[int, List] = defaultdict(list)
    for identity, entries in data.items():
        for entry in entries:
            frame_idx, cx, cy, w, h = entry
            ew = w * draw_scale
            eh = h * draw_scale
            x1 = max(0, int(cx - ew / 2))
            y1 = max(0, int(cy - eh / 2))
            x2 = min(frame_w, int(cx + ew / 2))
            y2 = min(frame_h, int(cy + eh / 2))
            frame_anns[int(frame_idx)].append((identity, x1, y1, x2, y2))
    return dict(frame_anns)


# ---------------------------------------------------------------------------
# Crop helper  (for the crop method)
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
# Windowed similarity helper
# ---------------------------------------------------------------------------

def _compute_windows(
    entries: List,
    frame_feats_map: Dict[int, torch.Tensor],
    text_feats: torch.Tensor,
    texts: List[str],
    window_frames: int,
    fps: float,
    device: torch.device,
    softmax: bool,
) -> Dict[int, Dict]:
    """Mean-pool per window, cosine-sim vs text, return window dict."""
    buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for entry in entries:
        fidx = int(entry[0])
        wid = fidx // window_frames
        if fidx in frame_feats_map:
            buckets[wid].append(frame_feats_map[fidx])

    windows: Dict[int, Dict] = {}
    for wid in sorted(buckets):
        feats_list = buckets[wid]
        w_mean = F.normalize(torch.stack(feats_list).mean(dim=0), dim=-1)   # [E]
        cos_sims = (w_mean.to(device) @ text_feats.T).cpu()                 # [Q]
        display = F.softmax(cos_sims, dim=0) if softmax else cos_sims
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
    return windows


def _get_window(windows: Dict[int, Dict], wid: int) -> Optional[Dict]:
    """Return window data for *wid*, falling back to the nearest earlier window."""
    if wid in windows:
        return windows[wid]
    earlier = [k for k in windows if k <= wid]
    if not earlier:
        return None
    return windows[max(earlier)]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> Tuple[int, int, int]:
    """Deterministic, visually distinct BGR colour for an identity string."""
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = __import__("cv2").cvtColor(hsv, __import__("cv2").COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _darken(color: Tuple[int, int, int], factor: float = 0.55) -> Tuple[int, int, int]:
    """Return a darkened version of a BGR colour."""
    return tuple(max(0, int(c * factor)) for c in color)   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_label_row(
    frame: np.ndarray,
    x: int, y_bottom: int,
    text: str,
    color: Tuple[int, int, int],
    font_scale: float,
) -> int:
    """
    Draw one filled label row.  Returns the y-coordinate of the row's top edge
    so the caller can stack another row above it.
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, ft)
    pad = 3
    y_top = y_bottom - th - 2 * pad
    cv2.rectangle(frame, (x, y_top), (x + tw + 2 * pad, y_bottom), color, -1)
    cv2.putText(frame, text,
                (x + pad, y_bottom - pad - baseline),
                font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)
    return y_top


def _draw_combined_box_labels(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    head_label: str, head_score: float,
    crop_label: str, crop_score: float,
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
) -> None:
    """
    Draw the expanded bounding box with two stacked label rows above it.

      ┌─────────────────────────────────┐  ← identity colour   (head)
      │ H: <head_label>    score        │
      ├─────────────────────────────────┤  ← darker shade      (crop)
      │ C: <crop_label>    score        │
      └─────────────────────────────────┘
      ┌ ─ ─ bbox ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
    """
    import cv2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    crop_color = _darken(color)
    # Row 2 (crop) — directly above the box top
    import cv2 as _cv2
    font = _cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    (_, th), _ = _cv2.getTextSize("A", font, font_scale, ft)
    row_h = th + 2 * 3   # text height + 2*pad

    row2_bottom = max(y1, 2 * row_h)
    row2_top = _draw_label_row(
        frame, x1, row2_bottom,
        f"C: {crop_label}  {crop_score:.2f}",
        crop_color, font_scale,
    )
    # Row 1 (head) — above row 2
    row1_bottom = max(row2_top, row_h)
    _draw_label_row(
        frame, x1, row1_bottom,
        f"H: {head_label}  {head_score:.2f}",
        color, font_scale,
    )


def _panel_width(texts: List[str], font_scale: float, bar_w: int = 120) -> int:
    """Return the pixel width of a scores panel for the given texts."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_w = max(cv2.getTextSize(t, font, font_scale, 1)[0][0] for t in texts) + 6
    return col_w + bar_w + 60 + 8


def _draw_scores_panel(
    frame: np.ndarray,
    texts: List[str],
    scores: List[float],
    title: str,
    x: int, y: int,
    font_scale: float = 0.45,
    bar_w: int = 120,
) -> None:
    """Draw a single semi-transparent bar-chart panel at (x, y)."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    lh = int(cv2.getTextSize("A", font, font_scale, ft)[0][1] * 2.2)
    col_w = max(cv2.getTextSize(t, font, font_scale, ft)[0][0] for t in texts) + 6
    panel_w = col_w + bar_w + 60 + 8
    rows = len(texts) + (1 if title else 0)
    panel_h = rows * lh + 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cy = y + 6
    if title:
        cv2.putText(frame, title, (x + 4, cy + lh - 4),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy += lh

    max_score = max(scores) if scores else 1.0
    for t, score in zip(texts, scores):
        cv2.putText(frame, t, (x + 4, cy + lh - 6),
                    font, font_scale, (200, 200, 200), ft, cv2.LINE_AA)
        filled = int(bar_w * score / max(max_score, 1e-6))
        bx = x + col_w
        cv2.rectangle(frame, (bx, cy + 4), (bx + filled, cy + lh - 4),
                      (100, 200, 100), -1)
        cv2.rectangle(frame, (bx, cy + 4), (bx + bar_w, cy + lh - 4),
                      (120, 120, 120), 1)
        cv2.putText(frame, f"{score:.2f}", (bx + bar_w + 4, cy + lh - 6),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy += lh


def _draw_dual_scores_overlay(
    frame: np.ndarray,
    texts: List[str],
    head_scores: List[float],
    crop_scores: List[float],
    title: str,
    x: int = 10, y: int = 10,
    font_scale: float = 0.45,
) -> None:
    """Draw two side-by-side bar-chart panels — left: head, right: crop."""
    bar_w = 100
    pw = _panel_width(texts, font_scale, bar_w)
    gap = 8
    _draw_scores_panel(frame, texts, head_scores,
                       f"H  {title}", x, y, font_scale, bar_w)
    _draw_scores_panel(frame, texts, crop_scores,
                       f"C  {title}", x + pw + gap, y, font_scale, bar_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    if not args.no_video and args.video is None:
        sys.exit("--video is required unless --no-video is set.")

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required.\n"
                 "Install with:  pip install opencv-python-headless")

    sys.path.insert(0, str(Path(__file__).parent))
    from pe_position_approach1 import build_patch_grid, PositionCrossAttention
    from pe_track_query import _head_forward_batch, _align_track_to_features

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
    # 1. Load pre-saved patch tokens  (head method)
    # ------------------------------------------------------------------
    print(f"Loading patch features from {args.features} …")
    feat_data = torch.load(args.features, map_location="cpu", weights_only=True)

    all_patch_tokens: torch.Tensor = feat_data["patch_tokens"]          # [T, N, D]
    proj: Optional[torch.Tensor]   = feat_data.get("proj")
    feat_keys: List[str]           = feat_data.get("frame_keys") or feat_data["frame_paths"]
    model_name: str                = feat_data["model_name"]
    enc_image_size: int            = feat_data["image_size"]
    patch_size: int                = feat_data["patch_size"]
    width: int                     = feat_data["width"]

    print(f"  {all_patch_tokens.shape[0]} frames  "
          f"patch_tokens {tuple(all_patch_tokens.shape)}")

    if proj is not None:
        proj = proj.to(device)

    # FPS
    fps: float = (args.fps
                  or feat_data.get("video_fps")
                  or feat_data.get("sample_fps"))
    if fps is None:
        cap_probe = cv2.VideoCapture(args.video)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        cap_probe.release()
    window_frames = max(1, int(args.window_sec * fps))
    print(f"  fps={fps:.3f}  window={args.window_sec}s  ({window_frames} frames/window)")

    # ------------------------------------------------------------------
    # 2. Load cross-attention head
    # ------------------------------------------------------------------
    mname = args.model or model_name
    print(f"Loading cross-attention head …")
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)
    if args.head_checkpoint is not None:
        state = torch.load(args.head_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: missing head keys (random init): {missing}")
        if unexpected:
            print(f"  Warning: unexpected checkpoint keys (ignored): {unexpected}")
        print(f"  Head weights loaded ← {args.head_checkpoint}")
    else:
        print("  Warning: --head-checkpoint not provided — using random head weights.")
    head.eval()
    patch_grid = build_patch_grid(enc_image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 3. Load PE model (full: visual + text encoder for crop method)
    # ------------------------------------------------------------------
    print(f"Loading PE model ({mname}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    pretrained = not args.no_pretrained
    pe_model = CLIP.from_config(
        mname, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()

    img_transform = get_image_transform(pe_model.visual.image_size)

    # ------------------------------------------------------------------
    # 4. Encode text descriptions  (shared by both methods)
    # ------------------------------------------------------------------
    tokenizer = get_text_tokenizer(pe_model.context_length)
    with torch.no_grad():
        text_feats = pe_model.encode_text(
            tokenizer(texts).to(device), normalize=True
        )  # [Q, E]
    print(f"  {len(texts)} text descriptions encoded.")

    # ------------------------------------------------------------------
    # 5. Load all tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    all_tracks = _load_all_tracks(args.track_file)
    print(f"  {len(all_tracks)} identities.  context_scale={args.context_scale}")

    # frame_idx → list of (identity, cx, cy, w, h)  — for the crop pass
    frame_to_crops: Dict[int, List] = defaultdict(list)
    for identity, entries in all_tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            frame_to_crops[int(fidx)].append((identity, cx, cy, w, h))

    # ------------------------------------------------------------------
    # 6. Video pass — encode crops  (crop method)
    # ------------------------------------------------------------------
    crop_frame_feats: Dict[str, Dict[int, torch.Tensor]] = {
        identity: {} for identity in all_tracks
    }

    cap_probe2 = cv2.VideoCapture(args.video)
    vid_total = int(cap_probe2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe2.release()

    print(f"\nEncoding crops (C method) from {args.video} …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    frame_idx = 0
    batch_crops: List[torch.Tensor] = []
    batch_keys:  List[Tuple[str, int]] = []

    def _flush_crop_batch() -> None:
        if not batch_crops:
            return
        imgs = torch.stack(batch_crops).to(device)
        with torch.no_grad():
            feats = pe_model.encode_image(imgs, normalize=True)   # [B, E]
        for (id_, fidx), feat in zip(batch_keys, feats):
            crop_frame_feats[id_][fidx] = feat.cpu()
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
                crop = _crop_bbox(pil_frame, cx, cy, w, h,
                                  args.context_scale, frame_w, frame_h)
                batch_crops.append(img_transform(crop))
                batch_keys.append((identity, frame_idx))
                if len(batch_crops) >= args.batch_size:
                    _flush_crop_batch()
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == vid_total:
            print(f"\r  {frame_idx}/{vid_total} frames", end="", flush=True)

    _flush_crop_batch()
    cap.release()
    print()

    # ------------------------------------------------------------------
    # 7. Head method — compute per-frame features from patch tokens
    # ------------------------------------------------------------------
    print("Computing head (H) features from patch tokens …")

    # identity → {frame_idx: [E] embedding}
    head_frame_feats: Dict[str, Dict[int, torch.Tensor]] = {}

    for identity, entries in all_tracks.items():
        try:
            fkeys = [f"{int(e[0]):06d}" for e in entries]
            feat_indices = _align_track_to_features(fkeys, feat_keys)
        except ValueError as e:
            print(f"  Skipping {identity} (head): {e}")
            head_frame_feats[identity] = {}
            continue

        id_feats: Dict[int, torch.Tensor] = {}
        # Build BBoxPrompt list for this identity (with context_scale expansion)
        from pe_position_approach1 import BBoxPrompt
        bboxes = []
        for entry in entries:
            _, cx, cy, w, h = entry
            ew = w * args.context_scale
            eh = h * args.context_scale
            x1 = int(cx - ew / 2)
            y1 = int(cy - eh / 2)
            bboxes.append(BBoxPrompt(
                pixel_coords=(x1, y1, int(ew), int(eh)),
                image_size=(frame_w, frame_h),
            ))

        with torch.no_grad():
            for start in range(0, len(feat_indices), args.batch_size):
                end      = min(start + args.batch_size, len(feat_indices))
                b_tokens = all_patch_tokens[feat_indices[start:end]]
                feats    = _head_forward_batch(
                    head, b_tokens, bboxes[start:end], patch_grid, proj, device
                )                                                   # [B, E]
                for i, feat in enumerate(feats):
                    orig_fidx = int(entries[start + i][0])
                    id_feats[orig_fidx] = feat.cpu()

        head_frame_feats[identity] = id_feats

    # ------------------------------------------------------------------
    # 8. Windowed similarity — per identity, both methods
    # ------------------------------------------------------------------
    # identity → {"head_windows": …, "crop_windows": …, "color": …,
    #             "window_frames": …}
    identity_info: Dict[str, Dict] = {}

    for identity, entries in all_tracks.items():
        hf = head_frame_feats.get(identity, {})
        cf = crop_frame_feats.get(identity, {})
        if not hf and not cf:
            print(f"  Skipping {identity}: no features.")
            continue

        head_wins = _compute_windows(entries, hf, text_feats, texts,
                                     window_frames, fps, device, args.softmax)
        crop_wins = _compute_windows(entries, cf, text_feats, texts,
                                     window_frames, fps, device, args.softmax)

        identity_info[identity] = {
            "head_windows":  head_wins,
            "crop_windows":  crop_wins,
            "color":         _identity_color(identity),
            "window_frames": window_frames,
        }

        for wid in sorted(set(head_wins) | set(crop_wins)):
            hw = head_wins.get(wid, {})
            cw = crop_wins.get(wid, {})
            t_str = (f"{hw.get('start_sec', cw.get('start_sec', 0)):5.1f}s"
                     f" – {hw.get('end_sec', cw.get('end_sec', 0)):5.1f}s")
            print(f"  {identity:>10}  [{t_str}]"
                  f"  H→ {hw.get('label','—')!r:<35} ({hw.get('score', 0):.3f})"
                  f"  C→ {cw.get('label','—')!r:<35} ({cw.get('score', 0):.3f})")

    # ------------------------------------------------------------------
    # 9. Print scores table
    # ------------------------------------------------------------------
    col_w = 20
    header = (f"  {'Identity':<10}  {'Window':<14}  "
              + "  ".join(f"H:{t[:col_w-3]:<{col_w}} C:{t[:col_w-3]:<{col_w}}"
                          for t in texts))
    print("\n" + "─" * min(len(header), 160))
    for identity, info in identity_info.items():
        all_wids = sorted(set(info["head_windows"]) | set(info["crop_windows"]))
        for wid in all_wids:
            hw = info["head_windows"].get(wid, {})
            cw = info["crop_windows"].get(wid, {})
            t_str = (f"{hw.get('start_sec', cw.get('start_sec',0)):.1f}"
                     f"–{hw.get('end_sec', cw.get('end_sec',0)):.1f}s")
            print(f"  {identity:<10}  {t_str:<14}", end="")
            for i, txt in enumerate(texts):
                hs = hw.get("all_scores", [0] * len(texts))[i]
                cs = cw.get("all_scores", [0] * len(texts))[i]
                hm = "◆" if i == (hw.get("all_scores", []) or [0]*len(texts)).index(
                    max(hw.get("all_scores", [0]*len(texts)))) else " "
                cm = "◆" if i == (cw.get("all_scores", []) or [0]*len(texts)).index(
                    max(cw.get("all_scores", [0]*len(texts)))) else " "
                print(f"  H{hm}{hs:+.2f} C{cm}{cs:+.2f}{'':<{col_w-12}}", end="")
            print()
    print("─" * min(len(header), 160))

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 10. Build frame annotations (expanded bboxes for drawing)
    # ------------------------------------------------------------------
    frame_annotations = _build_frame_annotations(
        args.track_file, args.context_scale, frame_w, frame_h
    )

    # ------------------------------------------------------------------
    # 11. Render annotated video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot re-open video: {args.video}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        for ann_i, (identity, x1, y1, x2, y2) in enumerate(annotations):
            if identity not in identity_info:
                continue
            info  = identity_info[identity]
            color = info["color"]
            wid   = frame_idx // info["window_frames"]

            hw = _get_window(info["head_windows"], wid)
            cw = _get_window(info["crop_windows"],  wid)
            if hw is None and cw is None:
                continue

            head_label = hw["label"] if hw else "—"
            head_score = hw["score"] if hw else 0.0
            crop_label = cw["label"] if cw else "—"
            crop_score = cw["score"] if cw else 0.0

            _draw_combined_box_labels(
                frame, x1, y1, x2, y2,
                head_label, head_score,
                crop_label, crop_score,
                color,
                font_scale=args.font_scale,
            )

            # Dual scores overlay for the first identity in the frame
            if args.scores_overlay and ann_i == 0:
                head_scores = hw["all_scores"] if hw else [0.0] * len(texts)
                crop_scores = cw["all_scores"] if cw else [0.0] * len(texts)
                t_str = (f"{(hw or cw)['start_sec']:.0f}"
                         f"–{(hw or cw)['end_sec']:.0f}s")
                _draw_dual_scores_overlay(
                    frame,
                    texts=texts,
                    head_scores=head_scores,
                    crop_scores=crop_scores,
                    title=f"{identity}  {t_str}",
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
