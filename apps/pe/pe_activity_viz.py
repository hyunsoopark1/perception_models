"""
PE Activity Visualizer

Given pre-saved patch features and a multi-identity track file, assigns the
best-matching activity description to every tracked person and renders an
annotated video with coloured bounding boxes and labels.

Default activity labels (used when --text is omitted):
  • a child running
  • a child walking
  • a child playing with blocks
  • a child playing with a house toy
  • a child reading a book
  • a child talking to friends
  • a child playing musical instruments
  • a child talking to a teacher
  • a child doing hand manipulation

Pipeline
--------
    patch_features.pt ──► cross-attn head ──► mean feat [D]  (per identity)
    track.json        ──► bboxes per frame
    text descriptions ──► text encoder    ──► text feats [Q, D]
                              │
                     cosine similarity
                              │
                    best label per identity
                              │
    original video   ──► annotated frames ──► output video

Usage
-----
    # Full pipeline — annotated video output:
    python apps/pe/pe_activity_viz.py \\
        --features        patch_features.pt \\
        --track-file      tracks.json \\
        --video           input.mp4 \\
        --image-size      1920 1080 \\
        --head-checkpoint head.pt \\
        --out             annotated.mp4

    # Scores table only, no video rendering:
    python apps/pe/pe_activity_viz.py \\
        --features        patch_features.pt \\
        --track-file      tracks.json \\
        --image-size      1920 1080 \\
        --head-checkpoint head.pt \\
        --no-video

    # Show scores bar chart overlay on the video:
    python apps/pe/pe_activity_viz.py \\
        --features        patch_features.pt \\
        --track-file      tracks.json \\
        --video           input.mp4 \\
        --image-size      1920 1080 \\
        --head-checkpoint head.pt \\
        --out             annotated.mp4 \\
        --scores-overlay

    # Custom activity labels:
    python apps/pe/pe_activity_viz.py \\
        --features        patch_features.pt \\
        --track-file      tracks.json \\
        --video           input.mp4 \\
        --image-size      1920 1080 \\
        --head-checkpoint head.pt \\
        --out             annotated.mp4 \\
        --text "a child running" "a child walking" "a child reading a book"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
        description="Annotate a video with per-identity activity labels from PE features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) from pe_extract_patch_features.py.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format track JSON: "
                        "{id: [[frame,x,y,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels.")
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention weights (.pt).")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Cross-attention heads — must match training (default: 8).")
    p.add_argument("--text", nargs="+", default=None, metavar="PHRASE",
                   help="Activity descriptions (default: 9 child-activity labels).")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name (inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path for the text encoder.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--video", default=None, metavar="PATH",
                   help="Original video file for rendering. Required unless --no-video.")
    p.add_argument("--out", default="annotated.mp4", metavar="PATH",
                   help="Output annotated video path (default: annotated.mp4).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering — only print the scores table.")
    p.add_argument("--batch-size", type=int, default=32, metavar="N",
                   help="Frames per head-forward batch (default: 32).")
    p.add_argument("--scores-overlay", action="store_true",
                   help="Draw a similarity bar chart in the video corner.")
    p.add_argument("--font-scale", type=float, default=0.55,
                   help="cv2 font scale for labels (default: 0.55).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track loading — all identities at once
# ---------------------------------------------------------------------------

def _load_all_tracks(
    path: str,
    image_size: Tuple[int, int],
) -> "dict[str, tuple[list[str], list]]":
    """
    Load every identity from an identity-format track file.

    Returns
    -------
    dict mapping identity → (frame_keys, bboxes)
        frame_keys : zero-padded frame indices, e.g. ["000000", "000003"]
        bboxes     : list of BBoxPrompt
    """
    from pe_position_approach1 import BBoxPrompt

    with open(path) as f:
        data = json.load(f)

    result: dict = {}
    for identity, entries in data.items():
        entries = sorted(entries, key=lambda e: e[0])
        fkeys:  list[str] = []
        bboxes: list      = []
        for entry in entries:
            frame_idx, x, y, w, h = entry
            fkeys.append(f"{int(frame_idx):06d}")
            bboxes.append(BBoxPrompt(
                pixel_coords=(int(x), int(y), int(w), int(h)),
                image_size=image_size,
            ))
        result[identity] = (fkeys, bboxes)
    return result


def _build_frame_annotations(path: str) -> "dict[int, list[tuple]]":
    """
    Build frame_index → list of (identity, x1, y1, x2, y2) for bbox drawing.
    Uses raw pixel values from the track file (no BBoxPrompt needed).
    """
    with open(path) as f:
        data = json.load(f)

    frame_anns: dict[int, list] = defaultdict(list)
    for identity, entries in data.items():
        for entry in entries:
            frame_idx, x, y, w, h = entry
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            frame_anns[int(frame_idx)].append((identity, x1, y1, x2, y2))
    return dict(frame_anns)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> "tuple[int, int, int]":
    """Deterministic, visually distinct BGR colour for an identity string."""
    # Spread hues evenly using the hash; high saturation + value for visibility
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = __import__("cv2").cvtColor(hsv, __import__("cv2").COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_box_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    score: float,
    color: "tuple[int, int, int]",
    font_scale: float = 0.55,
) -> None:
    """Draw a filled-label bounding box on frame (in-place, BGR)."""
    import cv2
    thickness = 2
    font      = cv2.FONT_HERSHEY_SIMPLEX
    ft        = 1  # font thickness

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}  {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, ft)
    pad = 3
    # Place label above the box; clamp to image top
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
    texts: list[str],
    scores: list[float],
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

    font  = cv2.FONT_HERSHEY_SIMPLEX
    ft    = 1
    lh    = int(cv2.getTextSize("A", font, font_scale, ft)[0][1] * 2.2)
    bar_w = 120
    col_w = max(cv2.getTextSize(t, font, font_scale, ft)[0][0] for t in texts) + 6
    panel_w = col_w + bar_w + 60 + 8
    rows    = len(texts) + (1 if title else 0)
    panel_h = rows * lh + 8

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cy = y + 6
    if title:
        cv2.putText(frame, title, (x + 4, cy + lh - 4),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy += lh

    max_score = max(scores) if scores else 1.0
    for text, score in zip(texts, scores):
        # Text label
        cv2.putText(frame, text, (x + 4, cy + lh - 6),
                    font, font_scale, (200, 200, 200), ft, cv2.LINE_AA)
        # Bar
        filled = int(bar_w * score / max(max_score, 1e-6))
        bx = x + col_w
        cv2.rectangle(frame, (bx, cy + 4), (bx + filled, cy + lh - 4),
                      (100, 200, 100), -1)
        cv2.rectangle(frame, (bx, cy + 4), (bx + bar_w, cy + lh - 4),
                      (120, 120, 120), 1)
        # Score value
        sv = f"{score:.2f}"
        cv2.putText(frame, sv, (bx + bar_w + 4, cy + lh - 6),
                    font, font_scale, (220, 220, 220), ft, cv2.LINE_AA)
        cy += lh


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

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size: Tuple[int, int] = tuple(args.image_size)   # type: ignore[assignment]
    texts = args.text if args.text is not None else DEFAULT_TEXTS

    # ------------------------------------------------------------------
    # 1. Load pre-saved patch tokens
    # ------------------------------------------------------------------
    print(f"Loading patch features from {args.features} …")
    feat_data = torch.load(args.features, map_location="cpu", weights_only=True)

    all_patch_tokens: torch.Tensor = feat_data["patch_tokens"]   # [T_all, N, D]
    proj: Optional[torch.Tensor]   = feat_data.get("proj")
    feat_keys: list[str]           = feat_data.get("frame_keys") or feat_data["frame_paths"]
    model_name: str                = feat_data["model_name"]
    enc_image_size: int            = feat_data["image_size"]
    patch_size: int                = feat_data["patch_size"]
    width: int                     = feat_data["width"]

    T_all = all_patch_tokens.shape[0]
    print(f"  {T_all} frames  patch_tokens {tuple(all_patch_tokens.shape)}")

    if proj is not None:
        proj = proj.to(device)

    # ------------------------------------------------------------------
    # 2. Load cross-attention head
    # ------------------------------------------------------------------
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)
    if args.head_checkpoint is not None:
        state = torch.load(args.head_checkpoint, map_location=device, weights_only=True)
        head.load_state_dict(state)
        print(f"  Head weights loaded ← {args.head_checkpoint}")
    else:
        print("  Warning: --head-checkpoint not provided — using random head.")
    head.eval()

    patch_grid = build_patch_grid(enc_image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 3. Load text encoder + encode activity descriptions
    # ------------------------------------------------------------------
    mname      = args.model or model_name
    pretrained = not args.no_pretrained

    print(f"Loading PE text encoder ({mname}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_text_tokenizer

    pe_model = CLIP.from_config(
        mname, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()
    del pe_model.visual
    torch.cuda.empty_cache()

    tokenizer = get_text_tokenizer(pe_model.context_length)
    with torch.no_grad():
        text_feats = pe_model.encode_text(
            tokenizer(texts).to(device), normalize=True
        )  # [Q, E]
    logit_scale = pe_model.logit_scale.exp().item()

    print(f"  {len(texts)} text descriptions encoded.")

    # ------------------------------------------------------------------
    # 4. Load all identity tracks + compute mean embedding per identity
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    all_tracks = _load_all_tracks(args.track_file, image_size)
    print(f"  {len(all_tracks)} identities found.")

    # identity → {"mean_feat", "label", "score", "all_scores", "color"}
    identity_info: dict[str, dict] = {}

    for identity, (track_fkeys, track_bboxes) in all_tracks.items():
        # Align to feature file
        try:
            feat_indices = _align_track_to_features(track_fkeys, feat_keys)
        except ValueError as e:
            print(f"  Skipping {identity}: {e}")
            continue

        # Per-frame features (head only, batched)
        frame_feats: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(feat_indices), args.batch_size):
                end      = min(start + args.batch_size, len(feat_indices))
                b_tokens = all_patch_tokens[feat_indices[start:end]]
                b_bboxes = track_bboxes[start:end]
                feats    = _head_forward_batch(
                    head, b_tokens, b_bboxes, patch_grid, proj, device
                )
                frame_feats.append(feats.cpu())

        per_frame  = torch.cat(frame_feats, dim=0)               # [T, E]
        mean_feat  = F.normalize(per_frame.mean(dim=0), dim=-1)  # [E]

        # Similarity against all text descriptions
        sims       = (mean_feat.to(device) @ text_feats.T).cpu()  # [Q]
        best_idx   = sims.argmax().item()
        best_score = sims[best_idx].item()
        best_label = texts[best_idx]

        identity_info[identity] = {
            "mean_feat":  mean_feat,
            "label":      best_label,
            "score":      best_score,
            "all_scores": sims.tolist(),
            "color":      _identity_color(identity),
        }

        print(f"  {identity:>10}  [{len(feat_indices):4d} frames]  "
              f"→  {best_label!r}  ({best_score:.3f})")

    # ------------------------------------------------------------------
    # 5. Print full scores table
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(f"  {'Identity':<12}", end="")
    for t in texts:
        short = t[:18].ljust(20)
        print(f"  {short}", end="")
    print()
    print("─" * 80)
    for identity, info in identity_info.items():
        print(f"  {identity:<12}", end="")
        for sc in info["all_scores"]:
            marker = "◆" if sc == max(info["all_scores"]) else " "
            print(f"  {marker}{sc:+.3f}            ", end="")
        print()
    print("─" * 80)

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 6. Build frame → annotation lookup from raw track data
    # ------------------------------------------------------------------
    frame_annotations = _build_frame_annotations(args.track_file)

    # ------------------------------------------------------------------
    # 7. Read original video + render annotated output
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

            _draw_box_label(
                frame, x1, y1, x2, y2,
                label=info["label"],
                score=info["score"],
                color=color,
                font_scale=args.font_scale,
            )

            # Scores overlay: show for the first annotated identity per frame
            if args.scores_overlay and identity == annotations[0][0]:
                _draw_scores_overlay(
                    frame,
                    texts=texts,
                    scores=info["all_scores"],
                    title=identity,
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
