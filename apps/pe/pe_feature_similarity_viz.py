"""
PE Feature Similarity Visualizer
=================================
Computes the **frame-level cosine similarity** between:

  H (Head)  : patch tokens  ──► cross-attention head  ──► embedding
  C (Crop)  : raw video crop ──► pe.encode_image(crop) ──► embedding

and overlays the similarity score directly on each tracked bounding box in
the output video.

Visual output per box
---------------------
  • Box border colour  : green (sim≈1) → red (sim≈0) gradient
  • Score badge above  : "id  sim=0.87"
  • Optional corner panel: per-identity similarity timeline (sparkline bar)

This is a diagnostic / ablation tool — no text queries are needed.

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
    python apps/pe/pe_feature_similarity_viz.py \\
        --video          input.mp4 \\
        --track-file     tracks.json \\
        --image-size     1920 1080 \\
        --features       patch_features.pt \\
        --head-checkpoint head.pt \\
        --out            similarity_viz.mp4 \\
        --timeline-overlay
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
    # --- inputs ---
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format track JSON: "
                        "{id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels (must match the tracked video).")
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) from "
                        "pe_extract_patch_features.py — used by the head method.")
    # --- head method ---
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention checkpoint (.pt).")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Cross-attention heads (default: 8, must match training).")
    # --- PE model (crop method) ---
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name (inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    # --- output ---
    p.add_argument("--out", default="similarity_viz.mp4", metavar="PATH",
                   help="Output video path (default: similarity_viz.mp4).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering — only print the per-frame CSV table.")
    # --- runtime ---
    p.add_argument("--batch-size", type=int, default=16, metavar="N",
                   help="Frames per encoding batch (default: 16).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Frame rate — inferred from video/features when omitted.")
    p.add_argument("--context-scale", type=float, default=4.0, metavar="S",
                   help="BBox expansion factor for crop extraction and drawing "
                        "(default: 4.0).")
    # --- visual ---
    p.add_argument("--timeline-overlay", action="store_true",
                   help="Draw per-identity similarity bar timelines in the top-left "
                        "corner of the video.")
    p.add_argument("--font-scale", type=float, default=0.55,
                   help="cv2 font scale for labels (default: 0.55).")
    p.add_argument("--sim-low", type=float, default=0.0,
                   help="Similarity value mapped to red   (default: 0.0).")
    p.add_argument("--sim-high", type=float, default=1.0,
                   help="Similarity value mapped to green (default: 1.0).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track / annotation helpers  (unchanged from pe_combined_viz.py)
# ---------------------------------------------------------------------------

def _load_all_tracks(path: str) -> Dict[str, List]:
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
    """frame_index → list of (identity, x1, y1, x2, y2)."""
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
# Colour helpers
# ---------------------------------------------------------------------------

def _sim_color(sim: float, low: float = 0.0, high: float = 1.0) -> Tuple[int, int, int]:
    """
    Map a similarity score in [low, high] to a BGR colour:
      low  → red   (0, 0, 255)
      mid  → yellow (0, 255, 255)
      high → green (0, 255, 0)
    """
    t = max(0.0, min(1.0, (sim - low) / max(high - low, 1e-6)))
    if t < 0.5:
        # red → yellow
        r, g, b = 0, int(255 * 2 * t), 255
    else:
        # yellow → green
        r, g, b = 0, 255, int(255 * 2 * (1.0 - t))
    return (r, g, b)  # BGR


def _identity_color(identity: str) -> Tuple[int, int, int]:
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    import cv2
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


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
    """
    Draw bounding box with:
      • Border colour  = sim gradient (red→green)
      • Label badge    = "<identity>  sim=<value>"  above the box
    """
    import cv2
    color = _sim_color(sim, sim_low, sim_high)

    # Draw box border (thickness 2, colour by similarity)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Badge text
    label = f"{identity}  sim={sim:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, ft)
    pad = 3
    badge_y2 = max(y1, th + 2 * pad + 2)
    badge_y1 = badge_y2 - th - 2 * pad
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 2 * pad, badge_y2), color, -1)
    cv2.putText(
        frame, label,
        (x1 + pad, badge_y2 - pad - baseline),
        font, font_scale, (255, 255, 255), ft, cv2.LINE_AA,
    )


def _draw_timeline_panel(
    frame: np.ndarray,
    identity: str,
    sim_history: List[float],   # chronological, most-recent last
    sim_low: float,
    sim_high: float,
    x: int, y: int,
    max_bars: int = 60,
    bar_w: int = 4,
    bar_max_h: int = 40,
    font_scale: float = 0.40,
) -> int:
    """
    Draw a compact bar-chart timeline of the last *max_bars* similarity values
    for *identity* at position (x, y).  Returns the y-coordinate of the
    panel's bottom edge so the caller can stack panels.
    """
    import cv2
    recent = sim_history[-max_bars:]
    n = len(recent)
    if n == 0:
        return y

    panel_w = n * bar_w + 2
    panel_h = bar_max_h + 20  # bars + label row
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w + 60, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{identity} sim", (x + 2, y + 12),
                font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    for i, s in enumerate(recent):
        h = max(1, int(bar_max_h * max(0.0, min(1.0, (s - sim_low) / max(sim_high - sim_low, 1e-6)))))
        bx = x + i * bar_w
        by_bottom = y + panel_h - 2
        color = _sim_color(s, sim_low, sim_high)
        cv2.rectangle(frame, (bx, by_bottom - h), (bx + bar_w - 1, by_bottom), color, -1)

    # Current value text
    cur = recent[-1]
    cv2.putText(frame, f"{cur:.2f}", (x + n * bar_w + 4, y + panel_h - 6),
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
        sys.exit("opencv-python is required.\n"
                 "Install with:  pip install opencv-python-headless")

    sys.path.insert(0, str(Path(__file__).parent))
    from pe_position_approach1 import build_patch_grid, PositionCrossAttention, BBoxPrompt
    from pe_track_query import _head_forward_batch, _align_track_to_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_w, frame_h = args.image_size

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

    print(f"  {all_patch_tokens.shape[0]} frames, "
          f"patch_tokens shape: {tuple(all_patch_tokens.shape)}")

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
    print(f"  fps={fps:.3f}")

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
            print(f"  Warning: missing head keys (random init): {missing}")
        if unexpected:
            print(f"  Warning: unexpected checkpoint keys (ignored): {unexpected}")
        print(f"  Head weights loaded ← {args.head_checkpoint}")
    else:
        print("  Warning: --head-checkpoint not provided — using random head weights.")
    head.eval()
    patch_grid = build_patch_grid(enc_image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 3. Load PE model  (crop method)
    # ------------------------------------------------------------------
    print(f"Loading PE model ({mname}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform

    pretrained = not args.no_pretrained
    pe_model = CLIP.from_config(
        mname, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()

    img_transform = get_image_transform(pe_model.visual.image_size)
    print("  PE model ready.")

    # ------------------------------------------------------------------
    # 4. Load all tracks
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
    # 5. Video pass — encode crops  (C method), per frame per identity
    # ------------------------------------------------------------------
    # identity → {frame_idx: normalised embedding [E]}
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
    # 6. Head method — compute per-frame embeddings from patch tokens
    # ------------------------------------------------------------------
    print("Computing head (H) features from patch tokens …")

    # identity → {frame_idx: normalised embedding [E]}
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
    # 7. Compute per-frame cosine similarity  H↔C  for each identity
    # ------------------------------------------------------------------
    # identity → {frame_idx: float similarity}
    frame_sim: Dict[str, Dict[int, float]] = {}

    print("\nPer-frame H↔C cosine similarity:")
    print("─" * 60)

    for identity in all_tracks:
        hf = head_frame_feats.get(identity, {})
        cf = crop_frame_feats.get(identity, {})
        common_frames = sorted(set(hf) & set(cf))
        if not common_frames:
            print(f"  {identity}: no overlapping frames — skipping.")
            frame_sim[identity] = {}
            continue

        sims: Dict[int, float] = {}
        for fidx in common_frames:
            h_vec = hf[fidx].unsqueeze(0)   # [1, E]  already normalised
            c_vec = cf[fidx].unsqueeze(0)   # [1, E]  already normalised
            sim = float(F.cosine_similarity(h_vec, c_vec).item())
            sims[fidx] = sim

        frame_sim[identity] = sims
        mean_sim = float(np.mean(list(sims.values())))
        min_sim  = float(np.min(list(sims.values())))
        max_sim  = float(np.max(list(sims.values())))
        print(f"  {identity:>12}  frames={len(sims):4d}  "
              f"mean={mean_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}")

    print("─" * 60)

    # ------------------------------------------------------------------
    # 8. Print CSV table
    # ------------------------------------------------------------------
    print("\nframe_idx,identity,sim_head_crop")
    for identity in sorted(frame_sim):
        for fidx in sorted(frame_sim[identity]):
            print(f"{fidx},{identity},{frame_sim[identity][fidx]:.6f}")

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 9. Build frame annotations (expanded bboxes for drawing)
    # ------------------------------------------------------------------
    frame_annotations = _build_frame_annotations(
        args.track_file, args.context_scale, frame_w, frame_h
    )

    # Chronological similarity history per identity (for timeline panel)
    # identity → list of (frame_idx, sim) in frame order
    sim_history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 10. Render annotated video
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

        for identity, x1, y1, x2, y2 in annotations:
            id_sims = frame_sim.get(identity, {})
            if frame_idx not in id_sims:
                # No similarity for this frame — draw grey box only
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            sim = id_sims[frame_idx]
            sim_history[identity].append(sim)

            _draw_similarity_box(
                frame, x1, y1, x2, y2,
                identity, sim,
                args.sim_low, args.sim_high,
                font_scale=args.font_scale,
            )

        # Timeline panels — stacked top-left
        if args.timeline_overlay:
            panel_y = 10
            for identity in sorted(sim_history):
                if sim_history[identity]:
                    panel_y = _draw_timeline_panel(
                        frame, identity, sim_history[identity],
                        args.sim_low, args.sim_high,
                        x=10, y=panel_y,
                        font_scale=args.font_scale * 0.75,
                    )

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == vid_total:
            print(f"\r  {frame_idx}/{vid_total} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")
