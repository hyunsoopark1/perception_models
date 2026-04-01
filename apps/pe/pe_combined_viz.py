"""
PE Combined Visualizer
======================
Runs *both* methods on the same video and overlays their per-frame language
similarity predictions on each tracked bounding box.

Methods
-------
  H (Head)  : patch tokens ──► cross-attention head ──► embedding
  C (Crop)  : crop bbox from raw frame ──► pe.encode_image(crop) ──► embedding

Both embeddings are compared with text descriptions via cosine similarity
**every frame** (no temporal windowing).  For each tracked identity the video
shows two stacked label rows above the bounding box:

    ┌──────────────────────────────┐
    │ H: <head label>    score     │  ← identity colour
    │ C: <crop label>    score     │  ← darker shade
    └──────────────────────────────┘

Processing loop (single pass, same structure as pe_feature_similarity_viz.py):

    load head + patch features
    load CLIP
    encode text descriptions once
    for each frame (up to --max-frames):
        im = imread(frame)
        for each identity / bbox in this frame:
            head_feat  = head(patch_tokens[frame], bbox)
            crop       = crop_image(im, bbox)
            crop_feat  = CLIP.encode_image(crop)
            head_scores = cosine_sim(head_feat, text_feats)   ← per frame
            crop_scores = cosine_sim(crop_feat, text_feats)   ← per frame
            draw H+C labels on frame
        save frame

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

Usage
-----
    python apps/pe/pe_combined_viz.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --features      patch_features.pt \\
        --head-checkpoint head.pt \\
        --out           annotated_combined.mp4

    python apps/pe/pe_combined_viz.py ... \\
        --text "a child running" "a child walking" \\
        --scores-overlay \\
        --max-frames 200
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
        description="Per-frame H+C language similarity overlay on a tracked video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, metavar="PATH")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) from "
                        "pe_extract_patch_features.py — used by the head method.")
    # --- head ---
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention checkpoint (.pt).")
    p.add_argument("--num-heads", type=int, default=8)
    # --- PE model ---
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name (inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path.")
    p.add_argument("--no-pretrained", action="store_true")
    # --- text labels ---
    p.add_argument("--text", nargs="+", default=None, metavar="PHRASE")
    p.add_argument("--text-file", default=None, metavar="PATH",
                   help="One description per line; overrides --text and defaults.")
    # --- output ---
    p.add_argument("--out", default="annotated_combined.mp4", metavar="PATH")
    # --- runtime ---
    p.add_argument("--max-frames", type=int, default=100, metavar="N",
                   help="Stop after this many frames (default: 100).")
    p.add_argument("--context-scale", type=float, default=1.0,
                   help="BBox expansion factor (default: 1.0).")
    p.add_argument("--softmax", action="store_true",
                   help="Display softmax-normalised scores instead of raw cosine sims.")
    # --- visual ---
    p.add_argument("--scores-overlay", action="store_true",
                   help="Draw dual H/C bar-chart panels in the top-left corner.")
    p.add_argument("--font-scale", type=float, default=0.55)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------

def _load_frame_to_tracks(path: str) -> Dict[int, List[Tuple]]:
    """Return frame_idx → list of (identity, cx, cy, w, h)."""
    with open(path) as f:
        data = json.load(f)
    frame_map: Dict[int, List] = defaultdict(list)
    for identity, entries in data.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            frame_map[int(fidx)].append((identity, cx, cy, w, h))
    return dict(frame_map)


# ---------------------------------------------------------------------------
# Crop helpers
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


def _expanded_box(
    cx: float, cy: float, w: float, h: float,
    context_scale: float,
    frame_w: int, frame_h: int,
) -> Tuple[int, int, int, int]:
    ew = w * context_scale
    eh = h * context_scale
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(frame_w, int(cx + ew / 2))
    y2 = min(frame_h, int(cy + eh / 2))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> Tuple[int, int, int]:
    import cv2
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _darken(color: Tuple[int, int, int], factor: float = 0.55) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)  # type: ignore[return-value]


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
    """Draw one filled label row; returns the y of its top edge."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, ft)
    pad = 3
    y_top = y_bottom - th - 2 * pad
    cv2.rectangle(frame, (x, y_top), (x + tw + 2 * pad, y_bottom), color, -1)
    cv2.putText(frame, text, (x + pad, y_bottom - pad - baseline),
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
    import cv2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    crop_color = _darken(color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    (_, th), _ = cv2.getTextSize("A", font, font_scale, ft)
    row_h = th + 2 * 3
    row2_bottom = max(y1, 2 * row_h)
    row2_top = _draw_label_row(
        frame, x1, row2_bottom,
        f"C: {crop_label}  {crop_score:.2f}",
        crop_color, font_scale,
    )
    row1_bottom = max(row2_top, row_h)
    _draw_label_row(
        frame, x1, row1_bottom,
        f"H: {head_label}  {head_score:.2f}",
        color, font_scale,
    )


def _draw_scores_panel(
    frame: np.ndarray,
    texts: List[str],
    scores: List[float],
    title: str,
    x: int, y: int,
    font_scale: float = 0.45,
    bar_w: int = 120,
) -> None:
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
    identity: str,
    x: int = 10, y: int = 10,
    font_scale: float = 0.45,
) -> None:
    import cv2
    bar_w = 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_w = max(cv2.getTextSize(t, font, font_scale, 1)[0][0] for t in texts) + 6
    pw = col_w + bar_w + 60 + 8
    gap = 8
    _draw_scores_panel(frame, texts, head_scores,
                       f"H  {identity}", x, y, font_scale, bar_w)
    _draw_scores_panel(frame, texts, crop_scores,
                       f"C  {identity}", x + pw + gap, y, font_scale, bar_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required:  pip install opencv-python-headless")

    sys.path.insert(0, str(Path(__file__).parent))
    from pe_position_approach1 import build_patch_grid, PositionCrossAttention, BBoxPrompt
    from pe_track_query import _head_forward_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_w, frame_h = args.image_size

    # ------------------------------------------------------------------
    # 1. Text labels
    # ------------------------------------------------------------------
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
    # 2. Load pre-saved patch tokens
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

    # frame_idx (int) → row index into all_patch_tokens
    frame_to_feat_idx: Dict[int, int] = {
        int(k): i for i, k in enumerate(feat_keys)
    }

    if proj is not None:
        proj = proj.to(device)

    print(f"  {all_patch_tokens.shape[0]} frames in patch features")

    # ------------------------------------------------------------------
    # 3. Load cross-attention head
    # ------------------------------------------------------------------
    mname = args.model or model_name
    print("Loading cross-attention head …")
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)
    if args.head_checkpoint is not None:
        state = torch.load(args.head_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning — missing keys (random init): {missing}")
        if unexpected:
            print(f"  Warning — unexpected keys (ignored): {unexpected}")
        print(f"  Loaded ← {args.head_checkpoint}")
    else:
        print("  Warning: no --head-checkpoint — using random weights.")
    head.eval()
    patch_grid = build_patch_grid(enc_image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 4. Load CLIP / PE model
    # ------------------------------------------------------------------
    print(f"Loading PE model ({mname}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    pe_model = CLIP.from_config(
        mname,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()
    img_transform = get_image_transform(pe_model.visual.image_size)

    # ------------------------------------------------------------------
    # 5. Encode text descriptions once (shared by H and C)
    # ------------------------------------------------------------------
    tokenizer = get_text_tokenizer(pe_model.context_length)
    with torch.no_grad():
        text_feats = pe_model.encode_text(
            tokenizer(texts).to(device), normalize=True
        )  # [Q, E]
    print(f"  {len(texts)} text descriptions encoded.")

    # ------------------------------------------------------------------
    # 6. Load tracks:  frame_idx → [(identity, cx, cy, w, h), ...]
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    frame_to_tracks = _load_frame_to_tracks(args.track_file)
    print(f"  {sum(len(v) for v in frame_to_tracks.values())} track entries "
          f"across {len(frame_to_tracks)} frames")

    # ------------------------------------------------------------------
    # 7. Open video + writer
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        vid_fps, (vid_w, vid_h),
    )

    frames_to_process = min(args.max_frames, vid_total)
    print(f"\nProcessing {args.video}  →  {out_path}")
    print(f"  {vid_w}×{vid_h}  {vid_fps:.2f} fps  "
          f"processing {frames_to_process}/{vid_total} frames  "
          f"context_scale={args.context_scale}")
    print("─" * 60)

    # ------------------------------------------------------------------
    # 8. Main loop: single pass, one frame at a time
    # ------------------------------------------------------------------
    frame_idx = 0
    while frame_idx < frames_to_process:
        ret, bgr = cap.read()
        if not ret:
            break

        tracks_this_frame = frame_to_tracks.get(frame_idx, [])

        if tracks_this_frame:
            pil_im = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            feat_idx = frame_to_feat_idx.get(frame_idx)
            patch_tokens_frame = (
                all_patch_tokens[feat_idx].unsqueeze(0)   # [1, N, D]
                if feat_idx is not None else None
            )

            first_identity_in_frame = True
            for identity, cx, cy, w, h in tracks_this_frame:
                x1, y1, x2, y2 = _expanded_box(
                    cx, cy, w, h, args.context_scale, frame_w, frame_h
                )
                color = _identity_color(identity)

                # ── Head feature ──────────────────────────────────────
                head_feat: Optional[torch.Tensor] = None
                if patch_tokens_frame is not None:
                    bbox = BBoxPrompt(
                        pixel_coords=(x1, y1, x2 - x1, y2 - y1),
                        image_size=(frame_w, frame_h),
                    )
                    with torch.no_grad():
                        head_feat = _head_forward_batch(
                            head,
                            patch_tokens_frame,   # [1, N, D]
                            [bbox],
                            patch_grid, proj, device,
                        )[0].cpu()               # [E]

                # ── Crop feature (CLIP) ───────────────────────────────
                crop = _crop_bbox(pil_im, cx, cy, w, h,
                                  args.context_scale, frame_w, frame_h)
                with torch.no_grad():
                    crop_feat = pe_model.encode_image(
                        img_transform(crop).unsqueeze(0).to(device),
                        normalize=True,
                    )[0].cpu()                   # [E]

                # ── Per-frame language similarity ─────────────────────
                # Crop scores (always available)
                crop_sims = (crop_feat.unsqueeze(0).to(device) @ text_feats.T).cpu().squeeze(0)
                crop_display = F.softmax(crop_sims, dim=0) if args.softmax else crop_sims
                crop_best = int(crop_display.argmax())
                crop_label = texts[crop_best]
                crop_score = crop_display[crop_best].item()

                # Head scores (when patch tokens are available)
                if head_feat is not None:
                    head_sims = (head_feat.unsqueeze(0).to(device) @ text_feats.T).cpu().squeeze(0)
                    head_display = F.softmax(head_sims, dim=0) if args.softmax else head_sims
                    head_best = int(head_display.argmax())
                    head_label = texts[head_best]
                    head_score = head_display[head_best].item()

                    _draw_combined_box_labels(
                        bgr, x1, y1, x2, y2,
                        head_label, head_score,
                        crop_label, crop_score,
                        color,
                        font_scale=args.font_scale,
                    )

                    if args.scores_overlay and first_identity_in_frame:
                        _draw_dual_scores_overlay(
                            bgr,
                            texts=texts,
                            head_scores=head_display.tolist(),
                            crop_scores=crop_display.tolist(),
                            identity=identity,
                            x=10, y=10,
                            font_scale=args.font_scale * 0.85,
                        )
                        first_identity_in_frame = False
                else:
                    # No head features — draw crop-only label
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 1)
                    _draw_label_row(
                        bgr, x1, max(y1, 20),
                        f"C: {crop_label}  {crop_score:.2f}",
                        _darken(color), args.font_scale,
                    )

        writer.write(bgr)
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == frames_to_process:
            print(f"\r  {frame_idx}/{frames_to_process} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")
