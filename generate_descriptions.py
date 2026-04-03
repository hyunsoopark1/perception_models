# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 2: Generate PLM descriptions for video clips.

Reads a clips.json produced by split_videos.py, runs each clip through
PerceptionLM, and writes descriptions.json consumed by compile_story_video.py
or query_video_compilation.py.

Run this stage as often as needed — changing the prompt, model, number of
sampled frames, or temperature — without re-splitting the source videos.

Usage:
    # Basic
    python generate_descriptions.py \\
        --clips ./output/clips.json \\
        --output_dir ./output/

    # Custom scene-description prompt
    python generate_descriptions.py \\
        --clips ./output/clips.json \\
        --output_dir ./output/ \\
        --prompt "In one sentence, describe the mood and setting of this scene."

    # Different model
    python generate_descriptions.py \\
        --clips ./output/clips.json \\
        --output_dir ./output/ \\
        --ckpt facebook/Perception-LM-8B \\
        --num_frames 12

    # Save under a different name to keep multiple description sets
    python generate_descriptions.py \\
        --clips ./output/clips.json \\
        --output_dir ./output/ \\
        --descriptions_file ./output/descriptions_mood.json \\
        --prompt "Describe the emotional atmosphere of this clip."
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_video_description import generate_description

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """\
You are a developmental behavior analyst.

Your task is to estimate a child's developmental level from a video using observable behavior.

CRITICAL RULE:
- If a behavior is NOT OBSERVED, DO NOT estimate it.
- Output null for features and age when evidence is missing.
- Never infer ability from absence of evidence.

---

## Step 1: Determine observability

For each category, first decide:

- motor_observed (true/false)
- autonomy_observed (true/false)
- attention_observed (true/false)
- interaction_observed (true/false)
- language_observed (true/false)

---

## Step 2: Extract features ONLY if observed

If observed = false:
- set ALL fields in that category to null

If observed = true:
- estimate features in [0,1]

---

## Step 3: Estimate category age ONLY if observed

If observed = false:
- age = null

---

## Step 4: Output JSON

{
  "observability": {
    "motor": true,
    "autonomy": true,
    "attention": true,
    "interaction": true,
    "language": false
  },

  "category_ages": {
    "motor_age_months": null,
    "autonomy_age_months": 0.0,
    "attention_age_months": 0.0,
    "interaction_age_months": 0.0,
    "language_age_months": null
  },

  "behavioral_features": {
    "motor": null,
    "autonomy": {...},
    "attention": {...},
    "interaction": {...},
    "language": null
  },

  "stage_distribution": {
    "S0": 0.0,
    "S1": 0.0,
    "S2": 0.0,
    "S3": 0.0
  },

  "evidence": [],

  "uncertainty": {
    "sources": [],
    "confidence": 0.0
  }
}\
"""


# ──────────────────────────────────────────────────────────────────────────────
# Description generation
# ──────────────────────────────────────────────────────────────────────────────


def describe_clips(
    clip_paths: List[str],
    model,
    tokenizer,
    config,
    prompt: str = DEFAULT_PROMPT,
    num_frames: int = 8,
    temperature: float = 0.0,
    max_gen_len: int = 200,
) -> List[dict]:
    """Generate a PLM description for each clip.

    Args:
        clip_paths: Ordered list of clip file paths.
        model: Loaded PLM model.
        tokenizer: PLM tokenizer.
        config: Model configuration.
        prompt: Text question/instruction sent to the model for every clip.
            Change this to reshape what the model focuses on — action,
            mood, setting, objects, etc.
        num_frames: Frames to sample per clip for PLM input.
        temperature: Sampling temperature (0.0 = greedy / deterministic).
        max_gen_len: Maximum tokens to generate per description.

    Returns:
        List of dicts with keys: ``clip_path``, ``description``, ``prompt``.
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
        results.append({
            "clip_path": clip_path,
            "description": description,
            "prompt": prompt,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate PLM descriptions for clips listed in clips.json. "
            "Re-run freely with different prompts or models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --clips ./output/clips.json --output_dir ./output/
  %(prog)s --clips ./output/clips.json --output_dir ./output/ \\
           --prompt "Describe the mood and setting of this scene."
  %(prog)s --clips ./output/clips.json --output_dir ./output/ \\
           --ckpt facebook/Perception-LM-8B --num_frames 12 \\
           --descriptions_file ./output/descriptions_8B.json
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--clips",
        type=str,
        required=True,
        help="Path to clips.json produced by split_videos.py.",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for descriptions.json (default: ./output/).",
    )
    parser.add_argument(
        "--descriptions_file",
        type=str,
        default=None,
        help=(
            "Explicit output path for the descriptions JSON. "
            "Useful for keeping multiple description sets side by side "
            "(default: <output_dir>/descriptions.json)."
        ),
    )

    # ── Prompt ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=(
            "Instruction sent to PLM for every clip. "
            'Default: "Describe what is happening in this video clip in one or two sentences." '
            "Change to focus on mood, setting, objects, actions, etc."
        ),
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
        help="Sampling temperature; 0.0 = greedy / deterministic (default: 0.0).",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="Max tokens to generate per description (default: 200).",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    descriptions_file = args.descriptions_file or os.path.join(
        output_dir, "descriptions.json"
    )

    # ── Load clip list ─────────────────────────────────────────────────────────
    if not os.path.exists(args.clips):
        logger.error(f"clips.json not found: {args.clips}")
        sys.exit(1)

    with open(args.clips) as f:
        clip_paths = json.load(f)

    # clips.json is a plain list of paths
    if clip_paths and isinstance(clip_paths[0], dict):
        clip_paths = [c["clip_path"] for c in clip_paths]

    missing = [p for p in clip_paths if not os.path.exists(p)]
    if missing:
        logger.warning(f"{len(missing)} clip(s) not found on disk — they will be skipped.")
        clip_paths = [p for p in clip_paths if os.path.exists(p)]

    if not clip_paths:
        logger.error("No valid clips found. Run split_videos.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(clip_paths)} clips from {args.clips}")
    logger.info(f'Prompt: "{args.prompt}"')

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info(f"\nLoading model: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    # ── Generate descriptions ──────────────────────────────────────────────────
    logger.info(f"\n=== Generating descriptions for {len(clip_paths)} clips ===")
    clips_with_desc = describe_clips(
        clip_paths=clip_paths,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(descriptions_file), exist_ok=True)
    with open(descriptions_file, "w") as f:
        json.dump(clips_with_desc, f, indent=2)

    logger.info(f"\nDone. {len(clips_with_desc)} descriptions → {descriptions_file}")
    logger.info(
        "Next steps:\n"
        f"  python compile_story_video.py --descriptions {descriptions_file} --output story.mp4\n"
        f"  python query_video_compilation.py --descriptions {descriptions_file} "
        f'--query "your query here" --output result.mp4'
    )


if __name__ == "__main__":
    main()
