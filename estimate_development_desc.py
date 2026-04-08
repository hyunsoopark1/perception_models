# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator — description-first pipeline.

Pipeline (per chunk):
  1. PLM Call 1 : Detect whether a child is visible (yes / no).
  2. PLM Call 2 : Generate a free-form behavioral description of the child.
  3. Semantic match: for each developmental feature, find the vocabulary phrase
                     most semantically similar to the description.
                     No additional PLM calls — pure text similarity.
  4. Inverse-interpolation of CDC milestone curves yields age estimates.
  5. A soft S0-S3 stage distribution is computed and a formatted report printed.

Compared with estimate_development.py:
  PLM calls per chunk:  N_domains × N_runs  →  1 (description only)
  Domain scoring:       model-selected exact phrase → semantic similarity match

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
    # aggregation across chunks
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
# Semantic matching — description → vocabulary phrase
# ------------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "without", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "not", "no", "only", "just", "more", "most", "some", "any", "all",
    "their", "they", "them", "this", "that", "these", "those", "it",
    "its", "he", "she", "his", "her", "him", "who", "what", "which",
    "while", "when", "as", "if", "then", "than", "also", "very",
    "can", "cannot", "could", "would", "will", "shall", "may", "might",
    "own", "other", "each", "both", "few", "such", "one", "two",
})


def _tokenize(text: str) -> set:
    """Lower-case, split on word boundaries, remove stopwords and short tokens."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _semantic_match_feature(description: str, feat: str,
                             threshold: float = 0.10) -> Optional[tuple]:
    """Find the vocabulary phrase most semantically similar to the description.

    Computes Jaccard similarity between the (stopword-filtered) description tokens
    and each vocabulary phrase's tokens.  Returns (score, phrase) or None if the
    best match is below `threshold` (feature not observed / not mentioned).

    Args:
        description: Free-form behavioral description from PLM.
        feat:        Feature key in _FEATURE_LEVELS (e.g. "locomotion").
        threshold:   Minimum Jaccard similarity to accept a match (default 0.10).

    Returns:
        (cdc_score, phrase) of the best-matching level, or None.
    """
    desc_tokens = _tokenize(description)

    best_sim   = -1.0
    best_level: Optional[tuple] = None

    for score, phrase in _FEATURE_LEVELS[feat]:
        phrase_tokens = _tokenize(phrase)
        sim = _jaccard(desc_tokens, phrase_tokens)
        if sim > best_sim:
            best_sim   = sim
            best_level = (score, phrase)

    if best_level is not None and best_sim >= threshold:
        return best_level
    return None


def _match_description_to_domain(description: str, domain: str,
                                  threshold: float = 0.10,
                                  debug: bool = False) -> dict:
    """Match a behavioral description against every feature in a domain.

    Returns same shape as estimate_development._assess_domain():
        {
            "features":    {feat: score},
            "phrases":     {feat: phrase},
            "age":         float | None,
            "raw_outputs": [],          # no PLM output for this step
        }
    """
    features_scores: dict = {}
    features_phrases: dict = {}

    for feat in _DOMAIN_FEATURES[domain]:
        match = _semantic_match_feature(description, feat, threshold=threshold)
        if match is not None:
            score, phrase = match
            features_scores[feat]  = score
            features_phrases[feat] = phrase
            if debug:
                desc_tok = _tokenize(description)
                phrase_tok = _tokenize(phrase)
                sim = _jaccard(desc_tok, phrase_tok)
                logger.info(
                    f"    [{domain}] {feat}: sim={sim:.3f} → '{phrase}' (score={score:.2f})"
                )
        else:
            if debug:
                logger.info(f"    [{domain}] {feat}: below threshold — not matched")

    return {
        "features":    features_scores,
        "phrases":     features_phrases,
        "age":         domain_age(domain, features_scores),
        "raw_outputs": [],
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
    desc_max_gen_len: int = 256,
    sim_threshold: float = 0.10,
    chunk_duration: Optional[float] = None,
    debug: bool = False,
) -> dict:
    """Run the description-first developmental assessment pipeline.

    Per chunk:
      Call 1 (video) : child detection (yes/no).
      Call 2 (video) : free-form behavioral description.
      Step 3 (text)  : semantic match description → vocabulary per feature.

    Total PLM calls: 2 per chunk (+ 1 for initial child detection).

    Args:
        desc_max_gen_len: Max tokens for the behavioral description (default 256).
        sim_threshold:    Min Jaccard similarity to accept a vocabulary match (default 0.10).
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

        n_chunks = len(segments)
        logger.info(
            f"Assessment: {n_chunks} chunk(s) × 1 description call "
            f"= {n_chunks} PLM calls (semantic matching, no per-domain calls)"
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

            # Step 2: semantic match description → all domains (no PLM calls)
            logger.info("  Semantic matching to domain vocabulary ...")
            domain_scores: dict = {}
            for domain in DOMAINS:
                ds = _match_description_to_domain(
                    description, domain, threshold=sim_threshold, debug=debug
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
            "raw_plm_output":     "\n\n".join(c["description"] for c in chunk_details),
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
        description="Developmental assessment via description + semantic matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --num_frames 16 --chunk_duration 10
  %(prog)s --video clip.mp4 --debug       # show per-feature similarity scores
  %(prog)s --video clip.mp4 --json_only > result.json
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the child's video.")
    parser.add_argument("--ckpt", type=str, default="facebook/Perception-LM-3B",
                        help="PLM checkpoint or HuggingFace ID.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample per PLM call (default: 8).")
    parser.add_argument("--desc_max_gen_len", type=int, default=256,
                        help="Max tokens for the behavioral description (default: 256).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 = greedy (default).")
    parser.add_argument("--sim_threshold", type=float, default=0.10,
                        help="Min Jaccard similarity to accept a vocabulary match (default: 0.10).")
    parser.add_argument("--chunk_duration", type=float, default=None,
                        help="Split video into chunks of this many seconds.")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Render keyword overlays onto the video and save here.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-feature similarity scores and matched phrases.")

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
        desc_max_gen_len=args.desc_max_gen_len,
        sim_threshold=args.sim_threshold,
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
