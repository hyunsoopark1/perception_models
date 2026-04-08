# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Split MOV (or any video) files in a folder into fixed-duration clips.

Bakes container-level rotation into every clip so downstream tools need
no special metadata handling.  Output clips are H.264-compatible mp4v files.

Usage:
    python generate_clip.py --input_dir data/videos/
    python generate_clip.py --input_dir data/videos/ --output_dir data/clips/ --clip_duration 5
    python generate_clip.py --input_dir data/videos/ --ext .mp4 --overwrite
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from split_videos import _apply_rotation, _get_rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def split_video(
    video_path: str,
    output_dir: str,
    clip_duration: float = 5.0,
    overwrite: bool = False,
) -> list:
    """Split one video into fixed-duration clips.

    Returns a list of dicts:
        {"clip_path": str, "source_file": str, "year_month": str,
         "clip_index": int, "t_start": float, "t_end": float}
    """
    src = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation     = _get_rotation(cap)
    raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    frames_per_clip = max(1, int(fps * clip_duration))
    n_clips = max(1, (total_frames + frames_per_clip - 1) // frames_per_clip)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")

    # Derive year_month from filename (e.g. "2025-06-15.MOV" → "2025-06")
    year_month = ""
    import re
    m = re.match(r"(\d{4}-\d{2})", src.stem)
    if m:
        year_month = m.group(1)

    clips = []
    writer = None
    clip_idx = 0
    clip_frame = 0
    clip_path = None

    def _open_writer(idx):
        p = out_dir / f"{src.stem}_clip{idx:03d}.mp4"
        return cv2.VideoWriter(str(p), fourcc, fps, (out_w, out_h)), str(p)

    frame_total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if clip_frame == 0:
            if writer is not None:
                writer.release()
                clips[-1]["t_end"] = round(frame_total / fps, 2)

            if clip_idx >= n_clips and total_frames > 0:
                # safety: shouldn't happen, but avoid infinite open
                break

            clip_path_str_candidate = str(out_dir / f"{src.stem}_clip{clip_idx:03d}.mp4")
            if not overwrite and Path(clip_path_str_candidate).exists():
                logger.info(f"  Skip existing: {clip_path_str_candidate}")
                # fast-forward frames
                for _ in range(frames_per_clip - 1):
                    cap.read()
                    frame_total += 1
                clip_idx += 1
                frame_total += 1
                continue

            writer, clip_path = _open_writer(clip_idx)
            clips.append({
                "clip_path":   clip_path,
                "source_file": src.name,
                "year_month":  year_month,
                "clip_index":  clip_idx,
                "t_start":     round(frame_total / fps, 2),
                "t_end":       None,
            })
            clip_idx += 1

        writer.write(_apply_rotation(frame, rotation))
        clip_frame += 1
        frame_total += 1

        if clip_frame >= frames_per_clip:
            clip_frame = 0

    if writer is not None:
        writer.release()
        if clips and clips[-1]["t_end"] is None:
            clips[-1]["t_end"] = round(frame_total / fps, 2)

    cap.release()
    logger.info(
        f"  {src.name}: {clip_idx} clip(s), "
        f"{total_frames} frames @ {fps:.1f} fps, rotation={rotation}°"
    )
    return clips


def main():
    parser = argparse.ArgumentParser(
        description="Split video files into fixed-duration clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input_dir data/videos/
  %(prog)s --input_dir data/videos/ --output_dir data/clips/ --clip_duration 5
  %(prog)s --input_dir data/videos/ --overwrite
        """,
    )
    parser.add_argument("--input_dir",     type=str, required=True,
                        help="Folder containing source videos.")
    parser.add_argument("--output_dir",    type=str, default=None,
                        help="Where to write clips (default: <input_dir>/clips).")
    parser.add_argument("--clip_duration", type=float, default=5.0,
                        help="Clip length in seconds (default: 5).")
    parser.add_argument("--ext",           type=str, default=None,
                        help="Filter by extension, e.g. .MOV (default: all video types).")
    parser.add_argument("--overwrite",     action="store_true",
                        help="Re-generate clips that already exist.")
    parser.add_argument("--save",          type=str, default=None,
                        help="Save clip metadata list as JSON to this path.")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of videos to process in parallel (default: 1).")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "clips"

    # Collect source videos
    exts = {args.ext.lower()} if args.ext else VIDEO_EXTENSIONS
    sources = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not sources:
        logger.error(f"No matching video files found in {input_dir}")
        return

    logger.info(f"Found {len(sources)} video(s) in {input_dir}")
    logger.info(f"Output dir: {output_dir}  |  clip_duration: {args.clip_duration}s  |  workers: {args.workers}")

    all_clips = []
    if args.workers > 1 and len(sources) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(split_video, str(src), str(output_dir), args.clip_duration, args.overwrite): src
                for src in sources
            }
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    clips = fut.result()
                    all_clips.extend(clips)
                except Exception as exc:
                    logger.error(f"Failed to process {src.name}: {exc}")
    else:
        for src in sources:
            logger.info(f"Processing: {src.name}")
            clips = split_video(str(src), str(output_dir), args.clip_duration, args.overwrite)
            all_clips.extend(clips)

    logger.info(f"Total clips generated: {len(all_clips)}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_clips, f, indent=2)
        logger.info(f"Clip metadata saved to: {args.save}")
    else:
        # default: save alongside output_dir
        default_json = output_dir / "clips.json"
        with open(default_json, "w") as f:
            json.dump(all_clips, f, indent=2)
        logger.info(f"Clip metadata saved to: {default_json}")


if __name__ == "__main__":
    main()
