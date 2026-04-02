# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video Story Compilation using PerceptionLM (PLM).

Pipeline:
  1. Split input video(s) into 2-second clips using OpenCV
  2. Generate text descriptions for each clip using PLM
  3. Order clips into a narrative arc and select the best subset
  4. Create a compilation video with text description overlays

Usage:
    # From a single video
    python video_story_compilation.py --video input.mp4 --output story.mp4

    # From a directory of videos
    python video_story_compilation.py --video_dir ./videos/ --output story.mp4

    # Use a larger model, limit to 8 story clips, save descriptions
    python video_story_compilation.py \\
        --video_dir ./videos/ \\
        --output story.mp4 \\
        --ckpt facebook/Perception-LM-8B \\
        --max_clips 8 \\
        --save_descriptions
"""

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

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

# Keyword lists for scoring narrative position (0 = opening, 3 = closing)
_NARRATIVE_KEYWORDS = {
    0: [  # Establishing / Opening
        "standing", "sitting", "scene", "landscape", "building", "street", "road",
        "outside", "inside", "room", "area", "location", "background", "setting",
        "morning", "daytime", "night", "weather", "empty", "quiet", "still",
    ],
    1: [  # Rising action / Development
        "walking", "moving", "approaching", "preparing", "reaching", "working",
        "talking", "looking", "carrying", "holding", "using", "watching", "showing",
        "beginning", "starting", "opening",
    ],
    2: [  # Action / Climax
        "running", "jumping", "racing", "fighting", "dancing", "performing",
        "competing", "playing", "throwing", "catching", "climbing", "driving",
        "pushing", "pulling", "lifting", "shouting", "falling",
    ],
    3: [  # Resolution / Closing
        "finishing", "completing", "stopping", "leaving", "departing", "ending",
        "returning", "waving", "celebrating", "resting", "smiling", "laughing",
        "final", "last", "conclusion",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Split videos into 2-second clips
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
        List of paths to the generated clip files, in temporal order.

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

        # Skip very short trailing clips (less than half the target duration)
        min_frames = frames_per_clip // 2
        if len(frames_buffer) < min_frames:
            logger.debug(
                f"Skipping short trailing clip: {len(frames_buffer)} frames "
                f"(min required: {min_frames})"
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
# Step 2: PLM description generation (delegates to generate_video_description)
# ──────────────────────────────────────────────────────────────────────────────


def describe_clips(
    clip_paths: List[str],
    model,
    tokenizer,
    config,
    num_frames: int = 8,
    temperature: float = 0.0,
    max_gen_len: int = 200,
    prompt: str = (
        "Describe what is happening in this video clip in one or two sentences."
    ),
) -> List[dict]:
    """Generate PLM descriptions for a list of video clips.

    Args:
        clip_paths: List of paths to video clip files.
        model: Loaded PLM model.
        tokenizer: PLM tokenizer.
        config: Model configuration.
        num_frames: Number of frames to sample per clip.
        temperature: Sampling temperature (0.0 = greedy).
        max_gen_len: Maximum tokens to generate per description.
        prompt: Text prompt/question for the model.

    Returns:
        List of dicts with keys: "clip_path", "description".
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
            logger.warning(f"    Failed: {exc}")
            description = ""

        logger.info(f"    → {description[:100]}{'...' if len(description) > 100 else ''}")
        results.append({"clip_path": clip_path, "description": description})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Order clips into a narrative arc
# ──────────────────────────────────────────────────────────────────────────────


def _narrative_score(description: str) -> float:
    """Return a float in [0, 3] representing the narrative position of a clip.

    Scores based on keyword frequency across four narrative phases:
      0 = establishing/opening, 1 = development, 2 = climax, 3 = resolution.
    """
    desc_lower = description.lower()
    phase_hits = {phase: 0 for phase in _NARRATIVE_KEYWORDS}
    for phase, keywords in _NARRATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                phase_hits[phase] += 1

    total = sum(phase_hits.values())
    if total == 0:
        return 1.5  # No signal → place in the middle of the arc
    return sum(phase * hits for phase, hits in phase_hits.items()) / total


def order_clips_for_story(
    clips_with_descriptions: List[dict],
    max_clips: int = 10,
) -> List[dict]:
    """Select and order clips to form a coherent narrative arc.

    Scores each clip for its narrative position (establishing → climax →
    resolution), sorts by score, then samples evenly across the arc so the
    final compilation has a clear beginning, middle, and end.

    Args:
        clips_with_descriptions: List of dicts with "clip_path" and "description".
        max_clips: Maximum number of clips to include in the output.

    Returns:
        Ordered list of clip dicts forming a coherent story.
    """
    if not clips_with_descriptions:
        return []

    # Score and sort by narrative position
    for item in clips_with_descriptions:
        item["_narrative_score"] = _narrative_score(item["description"])

    sorted_clips = sorted(clips_with_descriptions, key=lambda x: x["_narrative_score"])

    # Sample evenly across the sorted list to preserve the arc
    if len(sorted_clips) > max_clips:
        indices = np.linspace(0, len(sorted_clips) - 1, max_clips, dtype=int)
        selected = [sorted_clips[i] for i in indices]
    else:
        selected = sorted_clips

    scores = [round(c["_narrative_score"], 2) for c in selected]
    logger.info(
        f"Selected {len(selected)}/{len(clips_with_descriptions)} clips "
        f"(narrative scores: {scores})"
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Create compilation video with text overlay
# ──────────────────────────────────────────────────────────────────────────────


def _wrap_text(text: str, max_chars: int = 58) -> List[str]:
    """Wrap text to lines of at most max_chars characters."""
    return textwrap.wrap(text, width=max_chars) or [""]


def _add_text_overlay(
    frame: np.ndarray,
    text: str,
    max_chars: int = 58,
    font_scale: float = 0.55,
    thickness: int = 1,
    padding: int = 8,
    bg_alpha: float = 0.6,
) -> np.ndarray:
    """Draw a wrapped text caption at the bottom of a frame.

    Uses a semi-transparent dark background box for readability, then white
    text with a thin black outline on top.

    Args:
        frame: BGR image array (modified in-place on a copy).
        text: Caption text to display.
        max_chars: Max characters per wrapped line.
        font_scale: OpenCV font scale factor.
        thickness: Text stroke thickness.
        padding: Pixels of padding around the text block.
        bg_alpha: Background box opacity (0 = transparent, 1 = opaque).

    Returns:
        New frame array with the text overlay applied.
    """
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_text(text, max_chars)

    (_, line_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_step = line_h + baseline + 4
    total_text_h = line_step * len(lines) + padding * 2

    h, w = frame.shape[:2]
    box_y1 = max(0, h - total_text_h - padding)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, box_y1), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

    # Draw each line: black outline then white fill
    for i, line in enumerate(lines):
        y = box_y1 + padding + line_h + i * line_step
        cv2.putText(
            frame, line, (padding + 1, y + 1),
            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (padding, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return frame


def create_compilation_video(
    ordered_clips: List[dict],
    output_path: str,
    overlay_text: bool = True,
) -> str:
    """Concatenate ordered clips into a single compilation video.

    Each clip is written frame-by-frame with the PLM description overlaid
    at the bottom. Clips with different resolutions are resized to match
    the first clip.

    Args:
        ordered_clips: Ordered list of dicts with "clip_path" and "description".
        output_path: Destination path for the compilation video.
        overlay_text: Whether to burn in text descriptions.

    Returns:
        Path to the written compilation video.
    """
    if not ordered_clips:
        raise ValueError("No clips provided for compilation.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Read dimensions from the first clip
    first_cap = cv2.VideoCapture(ordered_clips[0]["clip_path"])
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    total_frames = 0
    for idx, item in enumerate(ordered_clips):
        clip_path = item["clip_path"]
        description = item.get("description", "")

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open clip {clip_path}, skipping.")
            continue

        clip_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        clip_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        clip_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if clip_w != out_w or clip_h != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            if overlay_text and description:
                frame = _add_text_overlay(frame, description)
            writer.write(frame)
            clip_frames += 1
            total_frames += 1

        cap.release()
        logger.info(
            f"  [{idx + 1}/{len(ordered_clips)}] {Path(clip_path).name} "
            f"({clip_frames} frames)"
        )

    writer.release()
    duration = total_frames / fps
    logger.info(
        f"Compilation complete: {output_path} "
        f"({total_frames} frames, {duration:.1f}s, {len(ordered_clips)} clips)"
    )
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Create a story compilation video from input videos using PLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video input.mp4 --output story.mp4
  %(prog)s --video_dir ./videos/ --output story.mp4
  %(prog)s --video_dir ./videos/ --output story.mp4 --max_clips 8 --save_descriptions
  %(prog)s --video input.mp4 --output story.mp4 --ckpt facebook/Perception-LM-8B
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to a single input video.")
    src.add_argument(
        "--video_dir", type=str, help="Directory containing input video files."
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default="story_compilation.mp4",
        help="Output compilation video path (default: story_compilation.mp4).",
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        default=None,
        help=(
            "Directory to save 2-second clips "
            "(default: <output_dir>/clips/)."
        ),
    )
    parser.add_argument(
        "--save_descriptions",
        action="store_true",
        help="Save clip descriptions to a JSON file next to the output video.",
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
        help="PLM checkpoint path or HuggingFace ID (default: facebook/Perception-LM-3B).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Frames to sample per clip for PLM description (default: 8).",
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
    output_dir = os.path.dirname(output_abs)
    clips_dir = args.clips_dir or os.path.join(output_dir, "clips")

    # ── Step 1: Collect source videos ─────────────────────────────────────────
    if args.video:
        source_videos = [args.video]
    else:
        source_videos = collect_videos(args.video_dir)
        if not source_videos:
            logger.error(f"No supported video files found in {args.video_dir}")
            sys.exit(1)
        logger.info(f"Found {len(source_videos)} source video(s).")

    # ── Step 2: Split into 2-second clips ─────────────────────────────────────
    logger.info("\n=== Step 1/4: Splitting videos into 2-second clips ===")
    all_clips: List[str] = []
    for video_path in source_videos:
        video_clips_dir = os.path.join(clips_dir, Path(video_path).stem)
        clips = split_video_into_clips(video_path, video_clips_dir, args.clip_duration)
        all_clips.extend(clips)

    if not all_clips:
        logger.error("No clips were generated. Exiting.")
        sys.exit(1)

    logger.info(f"Total clips: {len(all_clips)}")

    # ── Step 3: Generate PLM descriptions ─────────────────────────────────────
    logger.info(f"\n=== Step 2/4: Generating PLM descriptions ({args.ckpt}) ===")
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
        desc_path = output_abs.rsplit(".", 1)[0] + "_descriptions.json"
        with open(desc_path, "w") as f:
            json.dump(clips_with_desc, f, indent=2)
        logger.info(f"Descriptions saved to: {desc_path}")

    # ── Step 4: Order clips into a story arc ──────────────────────────────────
    logger.info("\n=== Step 3/4: Ordering clips into a narrative story arc ===")
    ordered_clips = order_clips_for_story(clips_with_desc, max_clips=args.max_clips)

    # ── Step 5: Create compilation video ──────────────────────────────────────
    logger.info("\n=== Step 4/4: Creating compilation video ===")
    create_compilation_video(
        ordered_clips=ordered_clips,
        output_path=output_abs,
        overlay_text=not args.no_overlay,
    )

    print(f"\n{'=' * 60}")
    print(f"Story compilation: {output_abs}")
    print(f"Clips included: {len(ordered_clips)}")
    print(f"{'=' * 60}")
    for i, item in enumerate(ordered_clips, 1):
        score = round(item.get("_narrative_score", 0), 2)
        desc = item["description"][:80]
        print(f"  {i:2d}. [score={score}] {Path(item['clip_path']).name}")
        print(f"       {desc}{'...' if len(item['description']) > 80 else ''}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
