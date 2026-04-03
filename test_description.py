# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Quick test: run generate_descriptions.py on a single video and print the full output.

Usage:
    python test_description.py --video path/to/video.mp4
    python test_description.py --video path/to/video.MOV --ckpt facebook/Perception-LM-3B
    python test_description.py --video path/to/video.mp4 --num_frames 16 --max_gen_len 512
"""

import argparse

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_descriptions import DEFAULT_PROMPT
from generate_video_description import generate_description


def main():
    parser = argparse.ArgumentParser(
        description="Test behavioral description on a single video."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to a video file.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="facebook/Perception-LM-3B",
        help="PLM checkpoint or HuggingFace ID (default: facebook/Perception-LM-3B).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=8,
        help="Frames to sample from the video (default: 8).",
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=1024,
        help="Max tokens to generate (default: 1024).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature; 0.0 = greedy (default: 0.0).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    print(f"\nPrompt:\n{'-' * 60}\n{DEFAULT_PROMPT}\n{'-' * 60}\n")

    result = generate_description(
        video_path=args.video,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=DEFAULT_PROMPT,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
    )

    print(f"Video : {result['video_path']}")
    print(f"Frames: {result['num_frames']}")
    print(f"Speed : {result['tokens_per_second']} tok/s")
    print(f"\n{'=' * 60}")
    print(result["description"])
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
