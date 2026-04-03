# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Convenience wrapper: run split_videos + generate_descriptions in one command.

For more control — especially to re-run description generation with a
different prompt or model without re-splitting — use the two stages directly:

    python split_videos.py          --video_dir ./data/ --output_dir ./output/
    python generate_descriptions.py --clips ./output/clips.json --output_dir ./output/ \\
                                    --prompt "Describe the mood of this scene."

Usage:
    python preprocess_videos.py --video_dir data/202503_a/ --output_dir ./output/
    python preprocess_videos.py --video input.MOV --output_dir ./output/ \\
        --prompt "What objects are visible in this clip?" \\
        --ckpt facebook/Perception-LM-8B
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_descriptions import DEFAULT_PROMPT, describe_clips
from generate_video_description import collect_videos
from split_videos import split_video_into_clips

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split videos into clips and generate PLM descriptions in one step. "
            "Produces clips.json and descriptions.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video input.MOV --output_dir ./output/
  %(prog)s --video_dir data/202503_a/ --output_dir ./output/
  %(prog)s --video_dir ./videos/ --output_dir ./output/ \\
           --prompt "Describe the mood and setting." --ckpt facebook/Perception-LM-8B
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to a single input video.")
    src.add_argument(
        "--video_dir", type=str, help="Directory of input videos (recursive)."
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root output directory (default: ./output/).",
    )
    parser.add_argument(
        "--descriptions_file",
        type=str,
        default=None,
        help="Explicit path for descriptions.json (default: <output_dir>/descriptions.json).",
    )

    # ── Clipping ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=2.0,
        help="Duration of each clip in seconds (default: 2.0).",
    )

    # ── Prompt ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Instruction sent to PLM for every clip.",
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ckpt",
        type=str,
        default="facebook/Perception-LM-3B",
        help="PLM checkpoint or HuggingFace ID (default: facebook/Perception-LM-3B).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Frames to sample per clip for PLM (default: 8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 = greedy (default: 0.0).",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="Max tokens to generate per description (default: 200).",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    clips_dir = os.path.join(output_dir, "clips")
    clips_file = os.path.join(output_dir, "clips.json")
    descriptions_file = args.descriptions_file or os.path.join(
        output_dir, "descriptions.json"
    )

    # ── Collect source videos ──────────────────────────────────────────────────
    if args.video:
        source_videos = [args.video]
    else:
        source_videos = collect_videos(args.video_dir)
        if not source_videos:
            logger.error(f"No supported video files found in {args.video_dir}")
            sys.exit(1)
        logger.info(f"Found {len(source_videos)} source video(s).")

    # ── Stage 1: Split ─────────────────────────────────────────────────────────
    logger.info("\n=== Stage 1/2: Splitting videos into clips ===")
    all_clips: List[str] = []
    for video_path in source_videos:
        video_clips_dir = os.path.join(clips_dir, Path(video_path).stem)
        clips = split_video_into_clips(video_path, video_clips_dir, args.clip_duration)
        all_clips.extend(clips)

    if not all_clips:
        logger.error("No clips generated. Exiting.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    with open(clips_file, "w") as f:
        json.dump(all_clips, f, indent=2)
    logger.info(f"clips.json → {clips_file}")

    # ── Stage 2: Describe ──────────────────────────────────────────────────────
    logger.info(f"\n=== Stage 2/2: Generating PLM descriptions ({args.ckpt}) ===")
    logger.info(f'Prompt: "{args.prompt}"')
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    clips_with_desc = describe_clips(
        clip_paths=all_clips,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
    )

    with open(descriptions_file, "w") as f:
        json.dump(clips_with_desc, f, indent=2)

    logger.info(f"\nDone.")
    logger.info(f"  clips.json       → {clips_file}")
    logger.info(f"  descriptions.json → {descriptions_file}")
    logger.info(
        "\nTo re-describe with a different prompt (no re-splitting needed):\n"
        f"  python generate_descriptions.py --clips {clips_file} "
        f"--output_dir {output_dir} --prompt \"your new prompt\""
    )


if __name__ == "__main__":
    main()
