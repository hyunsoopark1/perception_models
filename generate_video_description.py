# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video Description Generator using PerceptionLM (PLM).

Generates natural language descriptions of video content using Meta's
Perception Language Model built on top of the Perception Encoder.

Usage:
    # Basic usage - describe a video
    python generate_video_description.py --video path/to/video.mp4

    # Use a specific model checkpoint
    python generate_video_description.py --video path/to/video.mp4 \
        --ckpt facebook/Perception-LM-8B

    # Control generation with sampling parameters
    python generate_video_description.py --video path/to/video.mp4 \
        --num_frames 16 --temperature 0.7 --top_p 0.9

    # Ask a specific question about the video
    python generate_video_description.py --video path/to/video.mp4 \
        --prompt "What actions are being performed in this video?"

    # Process multiple videos from a directory
    python generate_video_description.py --video_dir path/to/videos/ \
        --output results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from apps.plm.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from core.args import dataclass_from_dict
from core.transforms.video_transform import get_video_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

DEFAULT_PROMPT = "Describe what is happening in this video in detail."


def load_model(ckpt: str):
    """Load the PLM model, tokenizer, and config from a checkpoint.

    Args:
        ckpt: Path to a local checkpoint directory or a HuggingFace model ID
              (e.g., "facebook/Perception-LM-3B").

    Returns:
        Tuple of (model, tokenizer, config).
    """
    logger.info(f"Loading model from: {ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)
    logger.info("Model loaded successfully.")
    return model, tokenizer, config


def generate_description(
    video_path: str,
    model,
    tokenizer,
    config,
    prompt: str = DEFAULT_PROMPT,
    num_frames: int = 16,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_gen_len: int = 512,
) -> dict:
    """Generate a text description for a single video.

    Args:
        video_path: Path to the video file.
        model: The loaded PLM model.
        tokenizer: The PLM tokenizer.
        config: The model configuration.
        prompt: The question/prompt to condition generation on.
        num_frames: Number of frames to sample from the video.
        temperature: Sampling temperature (0.0 for greedy decoding).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        max_gen_len: Maximum number of tokens to generate.

    Returns:
        A dict with keys: "video_path", "prompt", "description",
        "tokens_per_second", "num_frames".
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Build the video transform and load frames
    transform = get_video_transform(image_res=model.vision_model.image_size)
    video_info = (video_path, num_frames, None, None, None)
    frames, _ = transform(video_info)

    # Build the prompt tuple expected by the generator
    prompts = [(prompt, frames)]

    # Create the generator with the specified parameters
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs,
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_gen_len": max_gen_len,
        },
        strict=False,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Run generation and measure speed
    start_time = time.time()
    generation, _, _ = generator.generate(prompts)
    elapsed = time.time() - start_time

    description = generation[0]
    total_tokens = len(tokenizer.encode(description, False, False))
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "video_path": video_path,
        "prompt": prompt,
        "description": description,
        "tokens_per_second": round(tokens_per_second, 2),
        "num_frames": num_frames,
    }


def collect_videos(video_dir: str) -> list:
    """Recursively collect all supported video files from a directory."""
    videos = []
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Generate natural language descriptions for videos using PerceptionLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --ckpt facebook/Perception-LM-8B --num_frames 16
  %(prog)s --video_dir ./videos/ --output results.json
        """,
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", type=str, help="Path to a single video file."
    )
    input_group.add_argument(
        "--video_dir", type=str, help="Path to a directory of video files."
    )

    # Model arguments
    parser.add_argument(
        "--ckpt",
        type=str,
        default="facebook/Perception-LM-3B",
        help="Model checkpoint path or HuggingFace ID (default: facebook/Perception-LM-3B).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt/question to ask about the video.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample from the video (default: 16).",
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 for greedy decoding (default: 0.0).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling threshold (default: None).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter (default: None).",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512).",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON. If not set, prints to stdout.",
    )

    args = parser.parse_args()

    # Load the model once
    model, tokenizer, config = load_model(args.ckpt)

    # Collect video paths
    if args.video:
        video_paths = [args.video]
    else:
        video_paths = collect_videos(args.video_dir)
        if not video_paths:
            logger.error(f"No supported video files found in {args.video_dir}")
            sys.exit(1)
        logger.info(f"Found {len(video_paths)} video(s) in {args.video_dir}")

    # Generate descriptions
    results = []
    for i, video_path in enumerate(video_paths):
        logger.info(f"[{i + 1}/{len(video_paths)}] Processing: {video_path}")
        try:
            result = generate_description(
                video_path=video_path,
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt=args.prompt,
                num_frames=args.num_frames,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_gen_len=args.max_gen_len,
            )
            results.append(result)
            print(f"\n{'=' * 60}")
            print(f"Video: {result['video_path']}")
            print(f"Prompt: {result['prompt']}")
            print(f"Frames sampled: {result['num_frames']}")
            print(f"-" * 60)
            print(f"Description:\n{result['description']}")
            print(f"-" * 60)
            print(f"Tokens/sec: {result['tokens_per_second']}")
            print(f"{'=' * 60}\n")
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            results.append({
                "video_path": video_path,
                "error": str(e),
            })

    # Save results to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
