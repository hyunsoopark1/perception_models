# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 1: Split videos into fixed-duration clips.

Detects container-level rotation (portrait phone videos, etc.) and bakes it
into each saved clip so downstream stages need no rotation metadata.

Outputs clips.json — a list of clip paths consumed by generate_descriptions.py.

Usage:
    python split_videos.py --video input.MOV --output_dir ./output/
    python split_videos.py --video_dir data/202503_a/ --output_dir ./output/
    python split_videos.py --video_dir ./videos/ --output_dir ./output/ --clip_duration 3.0
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

from generate_video_description import collect_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Orientation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_rotation(cap: cv2.VideoCapture) -> int:
    """Return the clockwise rotation in degrees stored in the video container.

    Uses OpenCV's CAP_PROP_ORIENTATION_META (available in OpenCV 4.x+ with the
    FFMPEG backend). Returns 0 when the property is unavailable or absent.
    """
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except Exception:
        rotation = 0
    return rotation % 360  # normalise to {0, 90, 180, 270}


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a frame to correct for container-level orientation metadata.

    Args:
        frame: BGR image array.
        rotation: Clockwise rotation in degrees (0, 90, 180, or 270).

    Returns:
        Rotated frame; width and height are swapped for 90° and 270°.
    """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Clip splitting
# ──────────────────────────────────────────────────────────────────────────────


def split_video_into_clips(
    video_path: str,
    output_dir: str,
    clip_duration: float = 2.0,
) -> List[str]:
    """Split a video into fixed-duration clips with orientation correction.

    Reads the container rotation flag and applies it to every frame before
    writing, so each saved clip is upright with the correct aspect ratio.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to save the generated clip files.
        clip_duration: Duration in seconds per clip (default: 2.0).

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

    rotation = _get_rotation(cap)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = max(1, int(round(fps * clip_duration)))
    estimated = max(1, total_frames // frames_per_clip)

    logger.info(
        f"  {Path(video_path).name}: {total_frames} frames @ {fps:.1f} fps, "
        f"rotation={rotation}°, output {out_w}×{out_h} → ~{estimated} clips"
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
            frames_buffer.append(_apply_rotation(frame, rotation))

        if not frames_buffer:
            break
        if len(frames_buffer) < frames_per_clip // 2:
            logger.debug(f"Skipping short trailing clip ({len(frames_buffer)} frames)")
            break

        clip_path = os.path.join(output_dir, f"{stem}_clip_{clip_idx:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (out_w, out_h))
        for frame in frames_buffer:
            writer.write(frame)
        writer.release()

        clips.append(clip_path)
        clip_idx += 1

    cap.release()
    logger.info(f"  → {len(clips)} clips saved to: {output_dir}")
    return clips


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Split videos into fixed-duration clips. Outputs clips.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video input.MOV --output_dir ./output/
  %(prog)s --video_dir data/202503_a/ --output_dir ./output/
  %(prog)s --video_dir ./videos/ --output_dir ./output/ --clip_duration 3.0
        """,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to a single input video.")
    src.add_argument(
        "--video_dir", type=str, help="Directory of input videos (recursive)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root output directory (default: ./output/).",
    )
    parser.add_argument(
        "--clips_file",
        type=str,
        default=None,
        help="Path for clips.json (default: <output_dir>/clips.json).",
    )
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=2.0,
        help="Duration of each clip in seconds (default: 2.0).",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    clips_dir = os.path.join(output_dir, "clips")
    clips_file = args.clips_file or os.path.join(output_dir, "clips.json")

    # Collect source videos
    if args.video:
        source_videos = [args.video]
    else:
        source_videos = collect_videos(args.video_dir)
        if not source_videos:
            logger.error(f"No supported video files found in {args.video_dir}")
            sys.exit(1)
        logger.info(f"Found {len(source_videos)} source video(s).")

    # Split
    logger.info("\n=== Splitting videos into clips ===")
    all_clips: List[str] = []
    for video_path in source_videos:
        video_clips_dir = os.path.join(clips_dir, Path(video_path).stem)
        clips = split_video_into_clips(video_path, video_clips_dir, args.clip_duration)
        all_clips.extend(clips)

    if not all_clips:
        logger.error("No clips were generated. Exiting.")
        sys.exit(1)

    # Save clips.json
    os.makedirs(os.path.dirname(clips_file), exist_ok=True)
    with open(clips_file, "w") as f:
        json.dump(all_clips, f, indent=2)

    logger.info(f"\nDone. {len(all_clips)} clips → {clips_file}")
    logger.info(
        "Next step:\n"
        f"  python generate_descriptions.py --clips {clips_file} --output_dir {output_dir}"
    )


if __name__ == "__main__":
    main()
