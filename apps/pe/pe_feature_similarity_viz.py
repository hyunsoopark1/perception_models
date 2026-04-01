"""
PE Feature Similarity Visualizer
=================================
Computes the **frame-level cosine similarity** between:

  H (Head)  : patch tokens  ──► cross-attention head  ──► embedding
  C (Crop)  : raw video crop ──► pe.encode_image(crop) ──► embedding

Processing loop (per-frame, no pre-computation passes):

    load head + patch features
    load CLIP
    for each frame:
        im = imread(frame)
        for each identity / bbox visible in this frame:
            head_feat = head(patch_tokens[frame], bbox)
            crop      = crop_image(im, bbox)
            crop_feat = CLIP.encode_image(crop)
            sim       = cosine_similarity(head_feat, crop_feat)
            draw sim  on frame
        save frame to output video

Usage
-----
    python apps/pe/pe_feature_similarity_viz.py \\
        --video          input.mp4 \\
        --track-file     tracks.json \\
        --image-size     1920 1080 \\
        --features       patch_features.pt \\
        --head-checkpoint head.pt \\
        --out            similarity_viz.mp4

    # Show per-identity similarity timeline in corner:
    python apps/pe/pe_feature_similarity_viz.py ... --timeline-overlay
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Overlay per-frame H↔C feature cosine similarity on a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, metavar="PATH")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) — used by the head method.")
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention checkpoint (.pt).")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name (inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out", default="similarity_viz.mp4", metavar="PATH")
    p.add_argument("--context-scale", type=float, default=4.0,
                   help="BBox expansion factor (default: 4.0).")
    p.add_argument("--timeline-overlay", action="store_true",
                   help="Draw per-identity similarity bar timelines in the corner.")
    p.add_argument("--font-scale", type=float, default=0.55)
    p.add_argument("--sim-low",  type=float, default=0.0,
                   help="Similarity value mapped to red   (default: 0.0).")
    p.add_argument("--sim-high", type=float, default=1.0,
                   help="Similarity value mapped to green (default: 1.0).")
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

def _sim_color(sim: float, low: float = 0.0, high: float = 1.0) -> Tuple[int, int, int]:
    """Map similarity in [low, high] to BGR: red → yellow → green."""
    t = max(0.0, min(1.0, (sim - low) / max(high - low, 1e-6)))
    if t < 0.5:
        r, g, b = 0, int(255 * 2 * t), 255        # red → yellow
    else:
        r, g, b = 0, 255, int(255 * 2 * (1.0 - t))  # yellow → green
    return (r, g, b)  # BGR


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_similarity_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    identity: str,
    sim: float,
    sim_low: float,
    sim_high: float,
    font_scale: float = 0.55,
) -> None:
    """Box border colour encodes similarity; badge shows identity + score."""
    import cv2
    color = _sim_color(sim, sim_low, sim_high)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{identity}  sim={sim:.3f}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    ft    = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, ft)
    pad = 3
    badge_y2 = max(y1, th + 2 * pad + 2)
    badge_y1 = badge_y2 - th - 2 * pad
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 2 * pad, badge_y2), color, -1)
    cv2.putText(frame, label, (x1 + pad, badge_y2 - pad - baseline),
                font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)


def _draw_timeline_panel(
    frame: np.ndarray,
    identity: str,
    sim_history: List[float],
    sim_low: float,
    sim_high: float,
    x: int, y: int,
    max_bars: int = 60,
    bar_w: int = 4,
    bar_max_h: int = 40,
    font_scale: float = 0.40,
) -> int:
    """Bar-chart timeline of recent similarity values. Returns bottom y."""
    import cv2
    recent = sim_history[-max_bars:]
    n = len(recent)
    if n == 0:
        return y

    panel_h = bar_max_h + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + n * bar_w + 62, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{identity} sim", (x + 2, y + 12),
                font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    for i, s in enumerate(recent):
        h = max(1, int(bar_max_h * max(0.0, min(1.0,
                       (s - sim_low) / max(sim_high - sim_low, 1e-6)))))
        bx       = x + i * bar_w
        by_bot   = y + panel_h - 2
        cv2.rectangle(frame, (bx, by_bot - h), (bx + bar_w - 1, by_bot),
                      _sim_color(s, sim_low, sim_high), -1)

    cv2.putText(frame, f"{recent[-1]:.2f}", (x + n * bar_w + 4, y + panel_h - 6),
                font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return y + panel_h + 4


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

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_w, frame_h = args.image_size

    # ------------------------------------------------------------------
    # 1. Load pre-saved patch tokens
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

    # frame_idx (int) → index into all_patch_tokens
    frame_to_feat_idx: Dict[int, int] = {
        int(k): i for i, k in enumerate(feat_keys)
    }

    if proj is not None:
        proj = proj.to(device)

    print(f"  {all_patch_tokens.shape[0]} frames in patch features")

    # ------------------------------------------------------------------
    # 2. Load cross-attention head
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
        print("  Warning: no --head-checkpoint provided — using random weights.")
    head.eval()
    patch_grid = build_patch_grid(enc_image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 3. Load CLIP / PE model
    # ------------------------------------------------------------------
    print(f"Loading PE model ({mname}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform

    pe_model = CLIP.from_config(
        mname,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()
    img_transform = get_image_transform(pe_model.visual.image_size)
    print("  PE model ready.")

    # ------------------------------------------------------------------
    # 4. Load tracks:  frame_idx → [(identity, cx, cy, w, h), ...]
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    frame_to_tracks = _load_frame_to_tracks(args.track_file)
    print(f"  {sum(len(v) for v in frame_to_tracks.values())} track entries "
          f"across {len(frame_to_tracks)} frames")

    # ------------------------------------------------------------------
    # 5. Open video + writer
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

    print(f"\nProcessing {args.video}  →  {out_path}")
    print(f"  {vid_w}×{vid_h}  {vid_fps:.2f} fps  {vid_total} frames")
    print("─" * 60)

    # similarity history per identity (for timeline overlay)
    sim_history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 6. Main loop: one pass over the video
    # ------------------------------------------------------------------
    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        tracks_this_frame = frame_to_tracks.get(frame_idx, [])

        if tracks_this_frame:
            # Convert frame to PIL once (used for all crops this frame)
            pil_im = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            # Patch tokens for this frame (if available)
            feat_idx = frame_to_feat_idx.get(frame_idx)
            patch_tokens_frame = (
                all_patch_tokens[feat_idx].unsqueeze(0)   # [1, N, D]
                if feat_idx is not None else None
            )

            for identity, cx, cy, w, h in tracks_this_frame:
                x1, y1, x2, y2 = _expanded_box(
                    cx, cy, w, h, args.context_scale, frame_w, frame_h
                )

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

                # ── Similarity & draw ─────────────────────────────────
                if head_feat is not None:
                    sim = float(F.cosine_similarity(
                        head_feat.unsqueeze(0),
                        crop_feat.unsqueeze(0),
                    ).item())
                    sim_history[identity].append(sim)

                    _draw_similarity_box(
                        bgr, x1, y1, x2, y2,
                        identity, sim,
                        args.sim_low, args.sim_high,
                        font_scale=args.font_scale,
                    )
                else:
                    # No patch tokens for this frame — grey box only
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # Timeline panels stacked top-left
        if args.timeline_overlay:
            panel_y = 10
            for identity in sorted(sim_history):
                panel_y = _draw_timeline_panel(
                    bgr, identity, sim_history[identity],
                    args.sim_low, args.sim_high,
                    x=10, y=panel_y,
                    font_scale=args.font_scale * 0.75,
                )

        # Save frame
        writer.write(bgr)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == vid_total:
            print(f"\r  {frame_idx}/{vid_total} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")
