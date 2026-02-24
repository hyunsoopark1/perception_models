#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video-text similarity inference using a fine-tuned PE checkpoint.

Loads a checkpoint produced by train_video_text.py, encodes one or more
videos and one or more text strings, then prints the full similarity matrix
together with top-k rankings in both directions.

Usage examples
--------------
# Single video vs. single caption
python infer_video_text.py \
    --checkpoint runs/pe_video_text/checkpoint_final.pt \
    --videos clip.mp4 \
    --texts "A person rides a bicycle through a park."

# Multiple videos vs. multiple captions (full cross-similarity matrix)
python infer_video_text.py \
    --checkpoint runs/pe_video_text/checkpoint_final.pt \
    --videos v1.mp4 v2.mp4 v3.mp4 \
    --texts "caption one" "caption two" "caption three"

# Read texts and video paths from a JSONL file (conversations format)
python infer_video_text.py \
    --checkpoint runs/pe_video_text/checkpoint_final.pt \
    --jsonl data/val.jsonl \
    --video_root /datasets/vatex/videos \
    --topk 3

# Override model config (must match the checkpoint's architecture)
python infer_video_text.py \
    --checkpoint runs/pe_video_text/checkpoint_final.pt \
    --pe_config PE-Core-G14-448 \
    --num_temporal_layers 4 \
    --videos clip.mp4 \
    --texts "A woman opens a book and starts reading."
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from core.vision_encoder.pe import CLIP, TemporalTransformer
from core.vision_encoder.config import PE_VISION_CONFIG
from core.vision_encoder.transforms import get_text_tokenizer
from core.transforms.video_transform import VideoTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str,
    pe_config: str = "PE-Core-B16-224",
    num_temporal_layers: int = 2,
    num_temporal_heads: int = 8,
    device: torch.device = torch.device("cpu"),
) -> CLIP:
    """
    Reconstruct the CLIP model with TemporalTransformer and load fine-tuned weights.

    The architecture is rebuilt from ``pe_config`` / ``num_temporal_*`` so that
    it matches exactly what was used during training.  No pretrained hub weights
    are downloaded — all weights come from the checkpoint.
    """
    # Build skeleton (pretrained=False; weights come entirely from the checkpoint)
    model = CLIP.from_config(name=pe_config, pretrained=False)

    # Attach TemporalTransformer with the requested depth
    output_dim = PE_VISION_CONFIG[pe_config].output_dim
    if output_dim is not None:
        model.temporal_encoder = TemporalTransformer(
            width=output_dim,
            layers=num_temporal_layers,
            heads=num_temporal_heads,
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint:\n  %s", "\n  ".join(missing))
    if unexpected:
        logger.warning("Unexpected keys in checkpoint:\n  %s", "\n  ".join(unexpected))

    step = ckpt.get("step", "unknown")
    logger.info("Loaded checkpoint from '%s' (step=%s)", checkpoint, step)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Video & text encoding helpers
# ---------------------------------------------------------------------------

def encode_videos(
    model: CLIP,
    video_paths: List[str],
    num_frames: int,
    image_size: int,
    sampling_fps: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Encode a list of video files into L2-normalised feature vectors.

    Returns:
        Float tensor (N, D) on CPU.
    """
    transform = VideoTransform(size=image_size)
    all_feats: List[torch.Tensor] = []

    for i in range(0, len(video_paths), batch_size):
        batch_paths = video_paths[i : i + batch_size]
        frames_list = []
        for path in batch_paths:
            frames = _load_frames(path, transform, num_frames, sampling_fps)
            frames_list.append(frames)

        # (B, N, 3, H, W)
        videos = torch.stack(frames_list, dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_video(videos, normalize=True)  # (B, D)
        all_feats.append(feats.cpu())
        logger.info("Encoded videos %d–%d / %d", i + 1, i + len(batch_paths), len(video_paths))

    return torch.cat(all_feats, dim=0)


def encode_texts(
    model: CLIP,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a list of text strings into L2-normalised feature vectors.

    Returns:
        Float tensor (N, D) on CPU.
    """
    all_feats: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # tokenizer returns (B, context_length)
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens, normalize=True)  # (B, D)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


def _load_frames(
    video_path: str,
    transform: VideoTransform,
    num_frames: int,
    sampling_fps: int,
) -> torch.Tensor:
    """Load, decode, and pad/truncate a single video to ``num_frames`` frames."""
    video_info = (video_path, num_frames, None, None, None)
    try:
        frames, _ = transform(video_info, sampling_fps=sampling_fps)
    except Exception as exc:
        logger.warning("Could not load '%s': %s — using blank frames", video_path, exc)
        h = w = transform.size
        return torch.zeros(num_frames, 3, h, w)

    if frames.shape[0] < num_frames:
        pad = torch.zeros(num_frames - frames.shape[0], *frames.shape[1:], dtype=frames.dtype)
        frames = torch.cat([frames, pad], dim=0)
    return frames[:num_frames]


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(
    jsonl_path: str,
    video_root: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Parse a conversations-format JSONL file and return parallel lists of
    (video_paths, captions).  Caption is taken from the last "assistant" turn.
    """
    video_paths, captions = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_file = item["video"]
            path = str(Path(video_root) / video_file) if video_root else video_file
            caption = ""
            for turn in reversed(item.get("conversations", [])):
                if turn.get("from") == "assistant":
                    caption = turn.get("value", "")
                    break
            video_paths.append(path)
            captions.append(caption)
    logger.info("Loaded %d samples from '%s'", len(video_paths), jsonl_path)
    return video_paths, captions


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def print_similarity_matrix(
    sim: torch.Tensor,
    video_labels: List[str],
    text_labels: List[str],
    topk: int = 5,
) -> None:
    """
    Pretty-print the similarity matrix and top-k rankings.

    sim: (V, T) float tensor, values in [-1, 1].
    """
    V, T = sim.shape
    topk = min(topk, T, V)

    # ---- Raw matrix (truncated label widths) ----
    col_w = 12
    vid_w = 30
    header = f"{'':>{vid_w}}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in text_labels)
    print("\n=== Similarity matrix (cosine, ×100) ===")
    print(header)
    print("-" * len(header))
    for i, vlabel in enumerate(video_labels):
        row = f"{vlabel[:vid_w]:>{vid_w}}"
        for j in range(T):
            row += f"{sim[i, j].item() * 100:>{col_w}.1f}"
        print(row)

    # ---- Video → Text top-k ----
    print(f"\n=== Video → Text  (top-{topk}) ===")
    for i, vlabel in enumerate(video_labels):
        scores, indices = sim[i].topk(topk)
        print(f"\n  Video : {vlabel}")
        for rank, (idx, s) in enumerate(zip(indices.tolist(), scores.tolist()), 1):
            if idx == i:
                print(f"    #{rank} (correct)  [{s * 100:+.1f}]  {text_labels[idx]}")
            else:
                print(f"    #{rank}  [{s * 100:+.1f}]  {text_labels[idx]}")

    # ---- Text → Video top-k ----
    print(f"\n=== Text → Video  (top-{topk}) ===")
    for j, tlabel in enumerate(text_labels):
        scores, indices = sim[:, j].topk(topk)
        print(f"\n  Text  : {tlabel}")
        for rank, (idx, s) in enumerate(zip(indices.tolist(), scores.tolist()), 1):
            
            if idx == j:
                print(f"    #{rank} (correct)  [{s * 100:+.1f}]  {video_labels[idx]}")
            else:
                print(f"    #{rank}  [{s * 100:+.1f}]  {video_labels[idx]}")

    # ---- Retrieval metrics (only when V == T, i.e. paired evaluation) ----
    if V == T:
        gt = torch.arange(V)
        r1_v2t = (sim.argmax(dim=1) == gt).float().mean().item()
        r1_t2v = (sim.argmax(dim=0) == gt).float().mean().item()

        topk_v2t = {k: 0.0 for k in (1, 5, 10) if k <= T}
        topk_t2v = {k: 0.0 for k in (1, 5, 10) if k <= V}
        for k in topk_v2t:
            _, topk_idx = sim.topk(k, dim=1)
            topk_v2t[k] = (topk_idx == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        for k in topk_t2v:
            _, topk_idx = sim.topk(k, dim=0)
            topk_t2v[k] = (topk_idx == gt.unsqueeze(0)).any(dim=0).float().mean().item()

        print("\n=== Retrieval metrics (paired, V == T) ===")
        header_m = f"  {'metric':<18}" + "".join(f"R@{k:<5}" for k in sorted(topk_v2t))
        print(header_m)
        row_v2t = f"  {'Video → Text':<18}" + "".join(
            f"{topk_v2t[k] * 100:>6.1f}" for k in sorted(topk_v2t)
        )
        row_t2v = f"  {'Text → Video':<18}" + "".join(
            f"{topk_t2v[k] * 100:>6.1f}" for k in sorted(topk_t2v)
        )
        print(row_v2t)
        print(row_t2v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video-text similarity inference with a fine-tuned PE checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required ---
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pt file saved by train_video_text.py.",
    )

    # --- Input: direct ---
    p.add_argument(
        "--videos", nargs="+", default=[],
        metavar="PATH",
        help="One or more video file paths.",
    )
    p.add_argument(
        "--texts", nargs="+", default=[],
        metavar="TEXT",
        help='One or more text strings (quote each one).',
    )

    # --- Input: JSONL ---
    p.add_argument(
        "--jsonl",
        help="Path to a conversations-format JSONL file. "
             "Overrides --videos / --texts if provided.",
    )
    p.add_argument(
        "--video_root", default="",
        help="Root directory prepended to video filenames in the JSONL.",
    )

    # --- Model architecture (must match training config) ---
    p.add_argument("--pe_config", default="PE-Core-B16-224",
                   help="PE vision config name (default: PE-Core-B16-224).")
    p.add_argument("--num_temporal_layers", type=int, default=2,
                   help="TemporalTransformer depth used during training (default: 2).")
    p.add_argument("--num_temporal_heads", type=int, default=8,
                   help="TemporalTransformer heads used during training (default: 8).")

    # --- Video sampling ---
    p.add_argument("--num_frames", type=int, default=8,
                   help="Frames uniformly sampled per video (default: 8).")
    p.add_argument("--image_size", type=int, default=224,
                   help="Spatial resolution for video frames (default: 224).")
    p.add_argument("--sampling_fps", type=int, default=1,
                   help="Target FPS when sampling frames (default: 1).")

    # --- Inference ---
    p.add_argument("--batch_size", type=int, default=8,
                   help="Videos encoded per forward pass (default: 8).")
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top matches to display (default: 5).")
    p.add_argument("--device", default="",
                   help="'cuda', 'cpu', or '' for auto-detect (default: auto).")
    p.add_argument("--output_json", default="",
                   help="If set, write the full similarity matrix to this JSON file.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Inputs
    if args.jsonl:
        video_paths, texts = load_jsonl(args.jsonl, args.video_root)
    else:
        video_paths = args.videos
        texts = args.texts

    if not video_paths:
        sys.exit("No videos provided. Use --videos or --jsonl.")
    if not texts:
        sys.exit("No texts provided. Use --texts or --jsonl.")

    # Model
    model = load_model(
        checkpoint=args.checkpoint,
        pe_config=args.pe_config,
        num_temporal_layers=args.num_temporal_layers,
        num_temporal_heads=args.num_temporal_heads,
        device=device,
    )
    tokenizer = get_text_tokenizer(model.context_length)

    # Encode
    logger.info("Encoding %d video(s)…", len(video_paths))
    video_feats = encode_videos(
        model, video_paths,
        num_frames=args.num_frames,
        image_size=args.image_size,
        sampling_fps=args.sampling_fps,
        device=device,
        batch_size=args.batch_size,
    )

    logger.info("Encoding %d text(s)…", len(texts))
    text_feats = encode_texts(model, texts, tokenizer, device=device)

    # Similarity matrix  (V, T)
    sim = video_feats @ text_feats.T

    # Display
    video_labels = [Path(p).name for p in video_paths]
    text_labels = [t[:80] for t in texts]
    print_similarity_matrix(sim, video_labels, text_labels, topk=args.topk)

    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    plt.imshow(sim, cmap='viridis') # 'viridis' is a good default colormap
    plt.colorbar() # Add a color bar to show the scale
    plt.title("Matrix Visualization with imshow()")
    plt.xlabel("Video feature")
    plt.ylabel("Language feature")
    plt.savefig("output_base.png", dpi=150, bbox_inches="tight")

    # Optional JSON dump
    if args.output_json:
        import json as _json
        out = {
            "video_paths": video_paths,
            "texts": texts,
            "similarity": sim.tolist(),
        }
        with open(args.output_json, "w") as f:
            _json.dump(out, f, indent=2)
        logger.info("Similarity matrix written to '%s'", args.output_json)


if __name__ == "__main__":
    main()
