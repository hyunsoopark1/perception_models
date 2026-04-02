# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
End-to-end video story compilation pipeline using PerceptionLM (PLM).

Runs both pipeline stages in sequence:
  1. preprocess_videos.py  — split clips + generate PLM descriptions
  2. compile_story_video.py — order clips into a narrative arc + assemble video

For more control, run the two stages separately:
  python preprocess_videos.py  --video_dir ./data/ --output_dir ./output/
  python compile_story_video.py --descriptions ./output/descriptions.json --output story.mp4

Usage:
    python video_story_compilation.py --video input.mp4 --output story.mp4
    python video_story_compilation.py --video_dir data/202503_a/ --output story.mp4
    python video_story_compilation.py \\
        --video_dir ./videos/ \\
        --output story.mp4 \\
        --ckpt facebook/Perception-LM-8B \\
        --max_clips 10 \\
        --clip_duration 2.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from apps.plm.generate import load_consolidated_model_and_tokenizer
from compile_story_video import create_compilation_video, order_clips_for_story
from generate_video_description import collect_videos
from preprocess_videos import describe_clips, split_video_into_clips

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: split videos → PLM descriptions → story compilation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video input.mp4 --output story.mp4
  %(prog)s --video_dir data/202503_a/ --output story.mp4
  %(prog)s --video_dir ./videos/ --output story.mp4 --max_clips 8 --save_descriptions
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to a single input video.")
    src.add_argument(
        "--video_dir", type=str, help="Directory of input videos (searched recursively)."
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default="story_compilation.mp4",
        help="Output compilation video path (default: story_compilation.mp4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for intermediate clips and descriptions.json "
            "(default: same directory as --output)."
        ),
    )
    parser.add_argument(
        "--save_descriptions",
        action="store_true",
        help="Save descriptions.json alongside the output video.",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable text overlay on the output video.",
    )

    # ── Clipping ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=2.0,
        help="Duration of each clip in seconds (default: 2.0).",
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
        help="Sampling temperature; 0.0 = greedy decoding (default: 0.0).",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="Max tokens to generate per description (default: 200).",
    )

    # ── Story ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_clips",
        type=int,
        default=10,
        help="Max clips to include in the story compilation (default: 10).",
    )

    args = parser.parse_args()

    output_abs = os.path.abspath(args.output)
    output_dir = args.output_dir or os.path.dirname(output_abs)
    clips_dir = os.path.join(output_dir, "clips")

    # ── Collect source videos ──────────────────────────────────────────────────
    if args.video:
        source_videos = [args.video]
    else:
        source_videos = collect_videos(args.video_dir)
        if not source_videos:
            logger.error(f"No supported video files found in {args.video_dir}")
            sys.exit(1)
        logger.info(f"Found {len(source_videos)} source video(s).")

    # ── Stage 1a: Split into clips ─────────────────────────────────────────────
    logger.info("\n=== Stage 1/3: Splitting videos into clips ===")
    all_clips: List[str] = []
    for video_path in source_videos:
        video_clips_dir = os.path.join(clips_dir, Path(video_path).stem)
        clips = split_video_into_clips(video_path, video_clips_dir, args.clip_duration)
        all_clips.extend(clips)

    if not all_clips:
        logger.error("No clips were generated. Exiting.")
        sys.exit(1)
    logger.info(f"Total clips: {len(all_clips)}")

    # ── Stage 1b: Generate PLM descriptions ───────────────────────────────────
    logger.info(f"\n=== Stage 2/3: Generating PLM descriptions ({args.ckpt}) ===")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    clips_with_desc = describe_clips(
        clip_paths=all_clips,
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
    )

    if args.save_descriptions:
        desc_path = os.path.join(output_dir, "descriptions.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(desc_path, "w") as f:
            json.dump(clips_with_desc, f, indent=2)
        logger.info(f"Descriptions saved to: {desc_path}")

    # ── Stage 2: Order clips into narrative arc ────────────────────────────────
    logger.info("\n=== Stage 3/3: Ordering clips and assembling video ===")
    ordered_clips = order_clips_for_story(clips_with_desc, max_clips=args.max_clips)

    # ── Stage 3: Assemble compilation video ───────────────────────────────────
    create_compilation_video(
        ordered_clips=ordered_clips,
        output_path=output_abs,
        overlay_text=not args.no_overlay,
    )

    print(f"\n{'=' * 60}")
    print(f"Story compilation: {output_abs}")
    print(f"Clips included   : {len(ordered_clips)}")
    print(f"{'=' * 60}")
    for i, item in enumerate(ordered_clips, 1):
        score = round(item.get("_narrative_score", 0), 2)
        desc = item["description"]
        print(f"  {i:2d}. [score={score}] {Path(item['clip_path']).name}")
        print(f"       {desc[:80]}{'...' if len(desc) > 80 else ''}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
