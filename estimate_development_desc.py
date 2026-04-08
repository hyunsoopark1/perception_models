# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator — description-first pipeline.

Pipeline (per chunk):
  1. PLM Call 1 : Detect whether a child is visible (yes / no).
  2. PLM Call 2 : Generate a free-form behavioral description of the child.
  3. PLM Calls 3+: For each domain, feed the description back into the prompt
                   alongside the video and ask PLM to select ONE vocabulary
                   level.  The description grounds the matching (less ambiguity,
                   no majority voting needed).
  4. Parse, aggregate across chunks, inverse-interpolate ages, print report.

Compared with estimate_development.py:
  - The description (step 2) is generated first and embedded in every
    subsequent domain prompt as explicit context.
  - num_runs is dropped: the grounded description makes a single call
    reliable enough that majority voting adds little value.
  - PLM calls per chunk: N_domains × N_runs  →  1 + N_domains

Usage:
    python estimate_development_desc.py --video clip.mp4
    python estimate_development_desc.py --video clip.mp4 --num_frames 16 --chunk_duration 10
    python estimate_development_desc.py --video clip.mp4 --json_only > result.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from apps.plm.generate import load_consolidated_model_and_tokenizer

# Shared constants, CDC helpers, report rendering — all live in estimate_development.
from estimate_development import (
    # vocabulary / CDC data
    _FEATURE_LEVELS,
    _DECREASING_FEATURES,
    _DOMAIN_DESCRIPTIONS,
    _DOMAIN_FEATURES,
    DOMAINS,
    CDC_ANCHORS,
    STAGE_BOUNDS,
    # small helpers
    _feat_label,
    _score_to_age,
    domain_age,
    stage_distribution,
    # PLM output parsing & aggregation
    _parse_domain_output,
    _aggregate_domain_scores,
    # video utilities
    _split_into_chunks,
    _transcode_to_h264,
    _run_plm_text,
    # report / overlay
    print_report,
    render_assessment_video,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# PLM prompts
# ------------------------------------------------------------------------------

_PROMPT_CHILD = (
    "Is there a child (infant or toddler under 4 years old) clearly visible "
    "and active in this video? Answer only: yes or no."
)

# Asks for a rich behavioral description covering all developmental domains.
_PROMPT_DESCRIBE_BEHAVIOR = (
    "Describe this child's behavior in detail. "
    "Focus on: how they move (crawling, walking, running, climbing), "
    "what their hands do (grasping, stacking, drawing), "
    "how long they stay focused on one activity, "
    "whether they pursue a goal or solve a problem, "
    "how they interact with people around them, "
    "how dependent they seem on a caregiver, "
    "and how they communicate (sounds, words, gestures)."
)


# ------------------------------------------------------------------------------
# Description-grounded domain prompt builder
# ------------------------------------------------------------------------------


def _build_domain_match_prompt(domain: str, description: str) -> str:
    """Build a domain-matching prompt that embeds the pre-generated description.

    The model receives the behavioral description as explicit context, then
    selects the ONE vocabulary level per feature that best matches.
    Including the description reduces ambiguity so a single call is reliable.
    """
    features   = _DOMAIN_FEATURES[domain]
    desc_label = _DOMAIN_DESCRIPTIONS[domain]

    level_lines = []
    for feat in features:
        levels  = _FEATURE_LEVELS[feat]
        ordered = levels if feat in _DECREASING_FEATURES else sorted(levels, key=lambda x: x[0])
        option_str = " < ".join(phrase for _, phrase in ordered)
        level_lines.append(f"  {_feat_label(feat)}: {option_str}")

    answer_block = "\n".join(f"{_feat_label(feat)}: <option>" for feat in features)

    return (
        f"A child in this video has been described as:\n"
        f"\"{description}\"\n\n"
        f"Based on this description and what you observe, for each {desc_label} skill "
        f"below select the ONE option that BEST MATCHES the child's behavior.\n"
        f"Rules:\n"
        f"- Match what is described or directly visible.\n"
        f"- If the skill is not mentioned and not observable, write: not visible\n"
        f"Options run from youngest (left, ◄) to most developed (right, ►).\n\n"
        + "\n".join(level_lines)
        + f"\n\nReply in this exact format (one line per skill):\n{answer_block}"
    )


# ------------------------------------------------------------------------------
# Domain assessment using pre-generated description as context
# ------------------------------------------------------------------------------


def _assess_domain_from_description(
    seg: str,
    domain: str,
    description: str,
    model,
    tokenizer,
    config,
    num_frames: int,
    temperature: float,
    max_gen_len: int,
    debug: bool = False,
) -> dict:
    """Run one PLM call per domain, grounded by the behavioral description.

    Returns same shape as estimate_development._assess_domain() with one run.
    """
    prompt = _build_domain_match_prompt(domain, description)
    txt = _run_plm_text(
        seg, prompt, model, tokenizer, config,
        num_frames=num_frames, temperature=temperature, max_gen_len=max_gen_len,
    )
    parsed = _parse_domain_output(domain, txt)

    if debug:
        logger.info(f"    [{domain}] raw: {txt!r}")
        logger.info(f"    parsed: {parsed}")

    if not parsed:
        return {"features": {}, "phrases": {}, "age": None, "raw_outputs": [txt]}

    features = {feat: score for feat, (score, _) in parsed.items()}
    phrases  = {feat: phrase for feat, (_, phrase) in parsed.items()}
    return {
        "features":    features,
        "phrases":     phrases,
        "age":         domain_age(domain, features),
        "raw_outputs": [txt],
    }


# ------------------------------------------------------------------------------
# Full assessment — description-first pipeline
# ------------------------------------------------------------------------------


def assess_desc(
    video_path: str,
    model,
    tokenizer,
    config,
    num_frames: int = 8,
    temperature: float = 0.0,
    max_gen_len: int = 128,
    desc_max_gen_len: int = 256,
    chunk_duration: Optional[float] = None,
    debug: bool = False,
) -> dict:
    """Run the description-first developmental assessment pipeline.

    Per chunk:
      Call 1 (video)             : child detection (yes/no).
      Call 2 (video)             : free-form behavioral description.
      Calls 3 to 2+N_domains     : one domain-matching call each, with the
                                   description embedded in the prompt.

    Total PLM calls per chunk: 1 (description) + N_domains (matching).
    No majority voting — the description grounds the answer sufficiently.

    Args:
        desc_max_gen_len: Max tokens for the behavioral description (default 256).
        max_gen_len:      Max tokens for each domain matching call (default 128).
        chunk_duration:   Split into N-second chunks; None = full video as one chunk.
    """
    import os
    import tempfile

    tmp_path: Optional[str] = None
    chunk_paths: list = []
    plm_path = video_path

    try:
        # ---- Child detection (HEVC transcode fallback) -----------------------
        try:
            child_text = _run_plm_text(
                plm_path, _PROMPT_CHILD, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature, max_gen_len=10,
            )
        except RuntimeError as e:
            if "decoder" not in str(e).lower() and "NAL" not in str(e):
                raise
            logger.warning(f"HEVC decode failed ({e}). Transcoding to H.264 ...")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            _transcode_to_h264(video_path, tmp_path)
            plm_path = tmp_path
            child_text = _run_plm_text(
                plm_path, _PROMPT_CHILD, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature, max_gen_len=10,
            )

        child_present = bool(re.search(r"\byes\b", child_text, re.IGNORECASE))
        logger.info(f"Child present: {child_present!r}  (PLM raw: {child_text!r})")

        if not child_present:
            return {
                "video_path":         video_path,
                "child_present":      False,
                "plm_features":       {},
                "domain_ages":        {},
                "overall_age_months": None,
                "stage_distribution": None,
                "raw_plm_output":     child_text,
            }

        # ---- Determine segments ----------------------------------------------
        if chunk_duration:
            chunk_paths = _split_into_chunks(plm_path, chunk_duration)
            segments = chunk_paths
        else:
            segments = [plm_path]

        n_chunks  = len(segments)
        total_plm = n_chunks * (1 + len(DOMAINS))
        logger.info(
            f"Assessment: {n_chunks} chunk(s) × (1 description + {len(DOMAINS)} domain calls)"
            f" = {total_plm} PLM calls"
        )

        # ---- Assess each segment --------------------------------------------
        chunk_details: list = []
        for i, seg in enumerate(segments):
            t_start = i * chunk_duration if chunk_duration else 0
            t_end   = (i + 1) * chunk_duration if chunk_duration else None
            logger.info(f"Segment {i + 1}/{n_chunks}: {seg}")

            # Step 1: generate behavioral description (one PLM call with video)
            logger.info("  Generating behavioral description ...")
            description = _run_plm_text(
                seg, _PROMPT_DESCRIBE_BEHAVIOR, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature,
                max_gen_len=desc_max_gen_len,
            )
            logger.info(
                f"  Description: {description[:120]!r}"
                f"{'...' if len(description) > 120 else ''}"
            )

            # Step 2: for each domain, match description → vocabulary (1 PLM call each)
            domain_scores: dict = {}
            for domain in DOMAINS:
                logger.info(f"  Matching [{domain}] ...")
                ds = _assess_domain_from_description(
                    seg, domain, description, model, tokenizer, config,
                    num_frames=num_frames, temperature=temperature,
                    max_gen_len=max_gen_len, debug=debug,
                )
                domain_scores[domain] = ds
                if debug:
                    age_d = ds["age"]
                    print(f"  [{domain}] age={'%.1f' % age_d if age_d is not None else 'n/a'}"
                          f"  phrases={ds['phrases']}")

            raw_text = "\n".join(
                (f"{d}: " + ", ".join(f"{f}={p}" for f, p in domain_scores[d]["phrases"].items()))
                if domain_scores[d]["phrases"] else f"{d}: none"
                for d in DOMAINS
            )
            chunk_details.append({
                "index":         i + 1,
                "t_start":       t_start,
                "t_end":         t_end,
                "description":   description,
                "domain_scores": domain_scores,
                "domain_ages":   {d: domain_scores[d]["age"] for d in DOMAINS},
                "raw_text":      raw_text,
            })

        # ---- Aggregate across chunks -----------------------------------------
        plm_features: dict = {}
        domain_ages:  dict = {}

        for domain in DOMAINS:
            per_chunk = [c["domain_scores"][domain] for c in chunk_details]
            agg = _aggregate_domain_scores(per_chunk, domain)

            if agg["features"]:
                entry = dict(agg["features"])
                entry["evidence"] = ", ".join(
                    f"{f}={p}" for f, p in agg["phrases"].items()
                )
                entry["matched_keywords"] = dict(agg["phrases"])
                plm_features[domain] = entry
            else:
                plm_features[domain] = None
            domain_ages[domain] = agg["age"]

            if debug:
                print(f"\n--- Aggregated [{domain}] age={agg['age']} ---")
                for feat, score in agg["features"].items():
                    print(f"  {feat}: {score:.2f}  ({agg['phrases'].get(feat, '')})")

        observed_ages = [a for a in domain_ages.values() if a is not None]
        overall_age = sum(observed_ages) / len(observed_ages) if observed_ages else None
        stage_dist = stage_distribution(overall_age) if overall_age is not None else None

        return {
            "video_path":         video_path,
            "child_present":      True,
            "plm_features":       plm_features,
            "domain_ages":        domain_ages,
            "overall_age_months": round(overall_age, 1) if overall_age is not None else None,
            "stage_distribution": stage_dist,
            "raw_plm_output":     "\n\n".join(c["raw_text"] for c in chunk_details),
            "chunk_details":      chunk_details,
        }

    finally:
        for p in chunk_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Developmental assessment via description-grounded PLM matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --num_frames 16 --chunk_duration 10
  %(prog)s --video clip.mp4 --debug       # show raw PLM output per domain
  %(prog)s --video clip.mp4 --json_only > result.json
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the child's video.")
    parser.add_argument("--ckpt", type=str, default="facebook/Perception-LM-3B",
                        help="PLM checkpoint or HuggingFace ID.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample per PLM call (default: 8).")
    parser.add_argument("--max_gen_len", type=int, default=128,
                        help="Max tokens for domain matching calls (default: 128).")
    parser.add_argument("--desc_max_gen_len", type=int, default=256,
                        help="Max tokens for the behavioral description (default: 256).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 = greedy (default).")
    parser.add_argument("--chunk_duration", type=float, default=None,
                        help="Split video into chunks of this many seconds.")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Render keyword overlays onto the video and save here.")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw PLM outputs and parsed domain scores.")

    args = parser.parse_args()

    logger.info(f"Loading model: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    result = assess_desc(
        video_path=args.video,
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
        desc_max_gen_len=args.desc_max_gen_len,
        chunk_duration=args.chunk_duration,
        debug=args.debug,
    )

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
        print("--- Behavioral description ---")
        print(result["raw_plm_output"])

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to: {args.save}")

    if args.output_video:
        render_assessment_video(args.video, result, args.output_video)


if __name__ == "__main__":
    main()
