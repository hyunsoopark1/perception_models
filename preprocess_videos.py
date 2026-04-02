# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Preprocessing stage for the video story compilation pipeline.

Performs two steps:
  1. Split each input video into fixed-duration clips (default: 2 seconds)
  2. Generate a natural language description for each clip using PLM

Outputs a JSON file (descriptions.json) that is consumed by
compile_story_video.py to assemble the final story compilation.

Usage:
    # Single video
    python preprocess_videos.py --video input.mp4 --output_dir ./output/

    # Directory of videos (recursive)
    python preprocess_videos.py --video_dir data/202503_a/ --output_dir ./output/

    # Larger model, 3-second clips, more sampled frames
    python preprocess_videos.py \\
        --video_dir ./videos/ \\
        --output_dir ./output/ \\
        --ckpt facebook/Perception-LM-8B \\
        --clip_duration 3.0 \\
        --num_frames 12
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_video_description import collect_videos, generate_description

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

DEFAULT_PROMPT = (
    "Describe what is happening in this video clip in one or two sentences."
)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Split videos into fixed-duration clips
# ──────────────────────────────────────────────────────────────────────────────


def split_video_into_clips(
    video_path: str,
    output_dir: str,
    clip_duration: float = 2.0,
) -> List[str]:
    """Split a video into fixed-duration clips using OpenCV.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to save the generated clip files.
        clip_duration: Duration in seconds for each clip (default: 2.0).

    Returns:
        List of clip file paths in temporal order.

    Raises:
        ValueError: If the video cannot be opened.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        logger.warning(f"Could not detect FPS for {video_path}, defaulting to {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = max(1, int(round(fps * clip_duration)))
    estimated_clips = max(1, total_frames // frames_per_clip)

    logger.info(
        f"  {Path(video_path).name}: {total_frames} frames @ {fps:.1f} fps → "
        f"~{estimated_clips} clips of {clip_duration}s ({frames_per_clip} frames each)"
    )

    clips: List[str] = []
    clip_idx = 0
    stem = Path(video_path).stem

    while True:
        frames_buffer: List[np.ndarray] = []
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)

        if not frames_buffer:
            break

        # Drop very short trailing clips (< half the target duration)
        if len(frames_buffer) < frames_per_clip // 2:
            logger.debug(
                f"Skipping short trailing clip: {len(frames_buffer)} frames "
                f"(min required: {frames_per_clip // 2})"
            )
            break

        clip_path = os.path.join(output_dir, f"{stem}_clip_{clip_idx:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        for frame in frames_buffer:
            writer.write(frame)
        writer.release()

        clips.append(clip_path)
        clip_idx += 1

    cap.release()
    logger.info(f"  → {len(clips)} clips saved to: {output_dir}")
    return clips


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Generate PLM descriptions
# ──────────────────────────────────────────────────────────────────────────────


def describe_clips(
    clip_paths: List[str],
    model,
    tokenizer,
    config,
    num_frames: int = 8,
    temperature: float = 0.0,
    max_gen_len: int = 200,
    prompt: str = DEFAULT_PROMPT,
) -> List[dict]:
    """Generate a PLM description for each clip.

    Args:
        clip_paths: List of clip file paths.
        model: Loaded PLM model.
        tokenizer: PLM tokenizer.
        config: Model configuration.
        num_frames: Frames to sample per clip for PLM.
        temperature: Sampling temperature (0.0 = greedy decoding).
        max_gen_len: Maximum tokens to generate per description.
        prompt: Text question/prompt for the model.

    Returns:
        List of dicts with keys: ``clip_path``, ``description``.
    """
    results: List[dict] = []
    for i, clip_path in enumerate(clip_paths):
        logger.info(f"  [{i + 1}/{len(clip_paths)}] {Path(clip_path).name}")
        try:
            result = generate_description(
                video_path=clip_path,
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt=prompt,
                num_frames=num_frames,
                temperature=temperature,
                max_gen_len=max_gen_len,
            )
            description = result["description"]
        except Exception as exc:
            logger.warning(f"    PLM failed: {exc}")
            description = ""

        logger.info(
            f"    → {description[:100]}{'...' if len(description) > 100 else ''}"
        )
        results.append({"clip_path": clip_path, "description": description})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split videos into clips and generate PLM descriptions. "
            "Outputs a descriptions.json consumed by compile_story_video.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video input.mp4 --output_dir ./output/
  %(prog)s --video_dir data/202503_a/ --output_dir ./output/
  %(prog)s --video_dir ./videos/ --output_dir ./output/ --ckpt facebook/Perception-LM-8B
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
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for clips and descriptions.json (default: ./output/).",
    )
    parser.add_argument(
        "--descriptions_file",
        type=str,
        default=None,
        help=(
            "Path for the output JSON file "
            "(default: <output_dir>/descriptions.json)."
        ),
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
        help="PLM checkpoint path or HuggingFace ID (default: facebook/Perception-LM-3B).",
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt to ask PLM about each clip.",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    clips_dir = os.path.join(output_dir, "clips")
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

    # ── Step 1: Split into clips ───────────────────────────────────────────────
    logger.info("\n=== Step 1/2: Splitting videos into clips ===")
    all_clips: List[str] = []
    for video_path in source_videos:
        video_clips_dir = os.path.join(clips_dir, Path(video_path).stem)
        clips = split_video_into_clips(video_path, video_clips_dir, args.clip_duration)
        all_clips.extend(clips)

    if not all_clips:
        logger.error("No clips were generated. Exiting.")
        sys.exit(1)

    logger.info(f"Total clips: {len(all_clips)}")

    # ── Step 2: Generate PLM descriptions ─────────────────────────────────────
    logger.info(f"\n=== Step 2/2: Generating PLM descriptions ({args.ckpt}) ===")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    clips_with_desc = describe_clips(
        clip_paths=all_clips,
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
        prompt=args.prompt,
    )

    # ── Save descriptions JSON ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(descriptions_file), exist_ok=True)
    with open(descriptions_file, "w") as f:
        json.dump(clips_with_desc, f, indent=2)

    logger.info(f"\nDone. {len(clips_with_desc)} clips described.")
    logger.info(f"Descriptions saved to: {descriptions_file}")
    logger.info(
        "Next step:\n"
        f"  python compile_story_video.py "
        f"--descriptions {descriptions_file} --output story.mp4"
    )


if __name__ == "__main__":
    main()
