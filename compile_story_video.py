# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video assembly stage for the story compilation pipeline.

Reads the descriptions.json produced by preprocess_videos.py, orders the clips
into a coherent narrative arc, and concatenates them into a single video with
text description overlays burned in.

Run preprocess_videos.py first to generate the descriptions JSON, then:

Usage:
    python compile_story_video.py \\
        --descriptions ./output/descriptions.json \\
        --output story.mp4

    # Limit to 8 clips, skip text overlay
    python compile_story_video.py \\
        --descriptions ./output/descriptions.json \\
        --output story.mp4 \\
        --max_clips 8 \\
        --no_overlay

    # Override ordering: supply a manually curated JSON list of clip paths
    python compile_story_video.py \\
        --descriptions ./output/descriptions.json \\
        --output story.mp4 \\
        --order 3,0,7,1,5
"""

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import List

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keyword lists for scoring each clip's narrative position.
# Phase 0 = establishing/opening, 3 = resolution/closing.
_NARRATIVE_KEYWORDS: dict = {
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
# Story ordering
# ──────────────────────────────────────────────────────────────────────────────


def _narrative_score(description: str) -> float:
    """Return a float in [0, 3] representing the narrative position of a clip.

    Counts keyword hits across four phases and returns the weighted average.
    Clips with no keyword signal default to 1.5 (middle of the arc).
    """
    desc_lower = description.lower()
    phase_hits = {phase: 0 for phase in _NARRATIVE_KEYWORDS}
    for phase, keywords in _NARRATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                phase_hits[phase] += 1

    total = sum(phase_hits.values())
    if total == 0:
        return 1.5
    return sum(phase * hits for phase, hits in phase_hits.items()) / total


def order_clips_for_story(
    clips_with_descriptions: List[dict],
    max_clips: int = 10,
) -> List[dict]:
    """Select and order clips to form a coherent narrative arc.

    Sorts clips by narrative score (0 = establishing → 3 = resolution),
    then samples evenly across the sorted list so the compilation always has
    a clear beginning, middle, and end.

    Args:
        clips_with_descriptions: List of dicts with ``clip_path`` and
            ``description`` keys (as produced by preprocess_videos.py).
        max_clips: Maximum number of clips to include.

    Returns:
        Ordered list of clip dicts with an added ``_narrative_score`` key.
    """
    if not clips_with_descriptions:
        return []

    for item in clips_with_descriptions:
        item["_narrative_score"] = _narrative_score(item["description"])

    sorted_clips = sorted(clips_with_descriptions, key=lambda x: x["_narrative_score"])

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


def apply_manual_order(
    clips_with_descriptions: List[dict],
    order: List[int],
) -> List[dict]:
    """Return clips re-ordered by a manually supplied index list.

    Args:
        clips_with_descriptions: Full list of clip dicts.
        order: List of 0-based indices into clips_with_descriptions.

    Returns:
        Subset of clip dicts in the specified order.
    """
    n = len(clips_with_descriptions)
    result = []
    for idx in order:
        if 0 <= idx < n:
            result.append(clips_with_descriptions[idx])
        else:
            logger.warning(f"Index {idx} out of range (0–{n-1}), skipping.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Text overlay helpers
# ──────────────────────────────────────────────────────────────────────────────


def _wrap_text(text: str, max_chars: int = 58) -> List[str]:
    """Wrap text into lines of at most max_chars characters."""
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
    """Burn a wrapped text caption into the bottom of a video frame.

    Draws a semi-transparent dark background box, then white text with a
    thin black outline on top for readability against any background.

    Args:
        frame: BGR image array.
        text: Caption text to display.
        max_chars: Max characters per wrapped line.
        font_scale: OpenCV font scale factor.
        thickness: Text stroke thickness in pixels.
        padding: Pixels of padding around the text block.
        bg_alpha: Background box opacity (0 = transparent, 1 = opaque).

    Returns:
        New frame array with the overlay applied.
    """
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_text(text, max_chars)

    (_, line_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_step = line_h + baseline + 4
    total_text_h = line_step * len(lines) + padding * 2

    h, w = frame.shape[:2]
    box_y1 = max(0, h - total_text_h - padding)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, box_y1), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

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


# ──────────────────────────────────────────────────────────────────────────────
# Video assembly
# ──────────────────────────────────────────────────────────────────────────────


def create_compilation_video(
    ordered_clips: List[dict],
    output_path: str,
    overlay_text: bool = True,
) -> str:
    """Concatenate ordered clips into a single compilation video.

    Clips with different resolutions are resized to match the first clip.

    Args:
        ordered_clips: Ordered list of dicts with ``clip_path`` and
            ``description`` keys.
        output_path: Destination path for the output video.
        overlay_text: Whether to burn description captions into frames.

    Returns:
        Path to the written compilation video.

    Raises:
        ValueError: If ordered_clips is empty.
    """
    if not ordered_clips:
        raise ValueError("No clips provided for compilation.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Read dimensions from the first valid clip
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
            logger.warning(f"Cannot open clip: {clip_path} — skipping.")
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
        f"Compilation saved: {output_path} "
        f"({len(ordered_clips)} clips, {total_frames} frames, {duration:.1f}s)"
    )
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a story compilation video from clip descriptions JSON. "
            "Run preprocess_videos.py first to generate the descriptions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --descriptions ./output/descriptions.json --output story.mp4
  %(prog)s --descriptions ./output/descriptions.json --output story.mp4 --max_clips 8
  %(prog)s --descriptions ./output/descriptions.json --output story.mp4 --order 3,0,7,1,5
  %(prog)s --descriptions ./output/descriptions.json --output story.mp4 --no_overlay
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--descriptions",
        type=str,
        required=True,
        help="Path to descriptions.json produced by preprocess_videos.py.",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default="story.mp4",
        help="Output compilation video path (default: story.mp4).",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable text overlay on the output video.",
    )

    # ── Story ordering ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_clips",
        type=int,
        default=10,
        help="Max clips to include in the compilation (default: 10).",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help=(
            "Manually specify clip order as comma-separated 0-based indices "
            "into the descriptions JSON (e.g. --order 3,0,7,1,5). "
            "Overrides automatic narrative ordering."
        ),
    )

    args = parser.parse_args()

    # ── Load descriptions ──────────────────────────────────────────────────────
    if not os.path.exists(args.descriptions):
        logger.error(f"Descriptions file not found: {args.descriptions}")
        raise SystemExit(1)

    with open(args.descriptions) as f:
        clips_with_desc = json.load(f)

    logger.info(f"Loaded {len(clips_with_desc)} clip descriptions from {args.descriptions}")

    # ── Order clips ────────────────────────────────────────────────────────────
    if args.order:
        try:
            indices = [int(x.strip()) for x in args.order.split(",")]
        except ValueError:
            logger.error("--order must be comma-separated integers, e.g. --order 3,0,7")
            raise SystemExit(1)
        logger.info(f"Using manual order: {indices}")
        ordered_clips = apply_manual_order(clips_with_desc, indices)
    else:
        logger.info("Ordering clips by narrative arc...")
        ordered_clips = order_clips_for_story(clips_with_desc, max_clips=args.max_clips)

    if not ordered_clips:
        logger.error("No clips selected. Exiting.")
        raise SystemExit(1)

    # ── Assemble video ─────────────────────────────────────────────────────────
    logger.info(f"\nAssembling {len(ordered_clips)} clips into: {args.output}")
    create_compilation_video(
        ordered_clips=ordered_clips,
        output_path=args.output,
        overlay_text=not args.no_overlay,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Story compilation: {os.path.abspath(args.output)}")
    print(f"Clips included   : {len(ordered_clips)}")
    print(f"{'=' * 60}")
    for i, item in enumerate(ordered_clips, 1):
        score = item.get("_narrative_score")
        score_str = f"score={score:.2f}" if score is not None else "manual"
        desc = item["description"]
        print(f"  {i:2d}. [{score_str}] {Path(item['clip_path']).name}")
        print(f"       {desc[:80]}{'...' if len(desc) > 80 else ''}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
