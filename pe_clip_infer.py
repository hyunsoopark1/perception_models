#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Compare one video and one text using a pretrained PE-CLIP model.

Usage:
    python pe_clip_infer.py --video clip.mp4 --text "A person rides a bicycle."

    # Choose a different model size
    python pe_clip_infer.py \
        --model PE-Core-G14-448 \
        --video clip.mp4 \
        --text "A person rides a bicycle."

    # Adjust frame sampling
    python pe_clip_infer.py \
        --video clip.mp4 \
        --text "A person rides a bicycle." \
        --num_frames 16 \
        --sampling_fps 2

Available models:
    PE-Core-T16-384, PE-Core-S16-384, PE-Core-B16-224 (default),
    PE-Core-L14-336, PE-Core-G14-448
"""

import argparse
import logging
import sys

import torch

from core.vision_encoder.pe import CLIP
from core.vision_encoder.config import PE_VISION_CONFIG
from core.vision_encoder.transforms import get_text_tokenizer
from core.transforms.video_transform import VideoTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_video(
    path: str,
    transform: VideoTransform,
    num_frames: int,
    sampling_fps: int,
) -> torch.Tensor:
    """Decode a video file and return a (1, N, 3, H, W) tensor."""
    video_info = (path, num_frames, None, None, None)
    try:
        frames, _ = transform(video_info, sampling_fps=sampling_fps)
    except Exception as exc:
        logger.error("Failed to load video '%s': %s", path, exc)
        sys.exit(1)

    # Pad short clips with black frames
    if frames.shape[0] < num_frames:
        pad = torch.zeros(num_frames - frames.shape[0], *frames.shape[1:], dtype=frames.dtype)
        frames = torch.cat([frames, pad], dim=0)

    return frames[:num_frames].unsqueeze(0)  # (1, N, 3, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PE-CLIP video-text similarity inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--text",  required=True, help="Text string to compare against.")
    parser.add_argument(
        "--model", default="PE-Core-B16-224",
        choices=list(PE_VISION_CONFIG.keys()),
        help="PE-CLIP model variant (default: PE-Core-B16-224).",
    )
    parser.add_argument("--num_frames",  type=int, default=8,
                        help="Frames uniformly sampled from the video (default: 8).")
    parser.add_argument("--sampling_fps", type=int, default=1,
                        help="Target FPS when sampling frames (default: 1).")
    parser.add_argument("--device", default="",
                        help="'cuda' or 'cpu'. Default: auto-detect.")
    args = parser.parse_args()

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device : %s", device)

    # ---- Model ----
    logger.info("Loading pretrained PE-CLIP model '%s'…", args.model)
    model = CLIP.from_config(name=args.model, pretrained=True)
    model.to(device).eval()

    image_size = PE_VISION_CONFIG[args.model].image_size
    # image_size may be stored as a 1-tuple in some configs
    if isinstance(image_size, tuple):
        image_size = image_size[0]

    # ---- Encode video ----
    logger.info("Encoding video '%s' (%d frames @ %d fps)…",
                args.video, args.num_frames, args.sampling_fps)
    transform = VideoTransform(size=image_size)
    video = load_video(args.video, transform, args.num_frames, args.sampling_fps)
    video = video.to(device)

    with torch.no_grad():
        video_feat = model.encode_video(video, normalize=True)  # (1, D)

    # ---- Encode text ----
    logger.info("Encoding text: \"%s\"", args.text)
    tokenizer = get_text_tokenizer(model.context_length)
    tokens = tokenizer(args.text).to(device)  # (1, context_length)

    with torch.no_grad():
        text_feat = model.encode_text(tokens, normalize=True)   # (1, D)

    # ---- Similarity ----
    cosine_sim = (video_feat @ text_feat.T).item()   # scalar in [-1, 1]
    logit_scale = model.logit_scale.exp().item()
    scaled_sim   = cosine_sim * logit_scale          # same scale used during training

    print()
    print(f"  Video : {args.video}")
    print(f"  Text  : {args.text}")
    print()
    print(f"  Cosine similarity : {cosine_sim:+.4f}  (range -1 to +1)")
    print(f"  Scaled similarity : {scaled_sim:+.2f}   (cosine × logit_scale={logit_scale:.2f})")
    print()


if __name__ == "__main__":
    main()
