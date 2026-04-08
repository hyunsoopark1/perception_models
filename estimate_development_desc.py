# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator — description-first pipeline.

Pipeline (per chunk):
  1. PLM Call 1 : Detect whether a child is visible (yes / no).
  2. PLM Call 2 : Generate a free-form behavioral description of the child.
  3. Semantic match (no PLM): for each developmental feature, embed the
     description and all vocabulary phrases with a sentence transformer and
     pick the phrase with the highest cosine similarity.
  4. Aggregate across chunks, inverse-interpolate ages, print report.

There are only 2 PLM calls per chunk regardless of the number of domains.
Domain scoring comes entirely from the description text — if the description
does not mention a feature the score for that feature is left unset.

Usage:
    python estimate_development_desc.py --video clip.mp4
    python estimate_development_desc.py --video clip.mp4 --num_frames 16 --chunk_duration 10
    python estimate_development_desc.py --video clip.mp4 --sim_threshold 0.25
    python estimate_development_desc.py --video clip.mp4 --json_only > result.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from apps.plm.generate import load_consolidated_model_and_tokenizer

# Shared constants, CDC helpers, report rendering — all live in estimate_development.
from estimate_development import (
    # vocabulary / CDC data
    _FEATURE_LEVELS,
    _DECREASING_FEATURES,
    _DOMAIN_DESCRIPTIONS,
    _DOMAIN_FEATURES,
    DOMAINS,
    # small helpers
    _feat_label,
    domain_age,
    stage_distribution,
    # aggregation across chunks
    _aggregate_domain_scores,
    # video utilities
    _split_into_chunks,
    _transcode_to_h264,
    _run_plm_text,
    # report helpers
    _STAGE_LABELS,
    _bar,
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

# Focused description prompt: elicits observable developmental behaviors.
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
# NLI cross-encoder semantic matcher
# ------------------------------------------------------------------------------
# Strategy: run each (description, hypothesis) pair through an NLI model.
# The hypothesis is "The child is: <vocab phrase>".
# The phrase with the LOWEST contradiction score is the best match.
# If even the best phrase has contradiction > threshold, the feature is
# considered unobserved in this description.
#
# This outperforms cosine-similarity approaches because the cross-encoder
# sees both texts simultaneously and understands that "climbing on the slide"
# does NOT contradict "walking well and beginning to run" while it DOES
# contradict "crawling on hands and knees".
# ------------------------------------------------------------------------------

_nli_model = None   # loaded once on first use


def _load_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI cross-encoder (nli-MiniLM2-L6-H768) ...")
        _nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        logger.info("NLI cross-encoder loaded.")
    return _nli_model


def _nli_match_feature(
    description: str,
    feat: str,
    threshold: float = 0.50,
    debug: bool = False,
) -> Optional[tuple]:
    """Match a description to the vocabulary phrase least contradicted by it.

    Args:
        description: Free-form behavioral description from PLM.
        feat:        Feature key (e.g. "locomotion").
        threshold:   Max allowed contradiction score (default 0.50).
                     If the best phrase exceeds this, the feature is unobserved.

    Returns:
        (cdc_score, phrase) of the best-matching level, or None if unobserved.
    """
    model = _load_nli_model()
    levels = _FEATURE_LEVELS[feat]   # [(score, phrase), ...]

    # Each hypothesis is "The child is: <phrase>"
    pairs = [(description, f"The child is: {phrase}") for _, phrase in levels]
    scores = model.predict(pairs, apply_softmax=True)  # shape (n_phrases, 3)
    # Label order: [contradiction=0, entailment=1, neutral=2]
    contradiction = scores[:, 0]

    best_idx         = int(np.argmin(contradiction))
    best_contra      = float(contradiction[best_idx])
    best_score, best_phrase = levels[best_idx]

    if debug:
        for (_, ph), c in zip(levels, contradiction):
            mark = " <<<" if ph == best_phrase else ""
            logger.info(f"    [{feat}] contra={c:.3f}  {ph}{mark}")

    if best_contra < threshold:
        return (best_score, best_phrase)
    return None


def _match_description_to_domain(
    description: str,
    domain: str,
    threshold: float = 0.50,
    debug: bool = False,
) -> dict:
    """NLI-match a description against every feature in one domain.

    Returns same shape as estimate_development._assess_domain():
        {"features": {feat: score}, "phrases": {feat: phrase},
         "age": float|None, "raw_outputs": []}
    """
    features_scores: dict = {}
    features_phrases: dict = {}

    for feat in _DOMAIN_FEATURES[domain]:
        match = _nli_match_feature(description, feat,
                                   threshold=threshold, debug=debug)
        if match is not None:
            score, phrase = match
            features_scores[feat]  = score
            features_phrases[feat] = phrase

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
    sim_threshold: float = 0.50,
    chunk_duration: Optional[float] = None,
    debug: bool = False,
) -> dict:
    """Run the description-first developmental assessment pipeline.

    Per chunk:
      PLM Call 1 (video) : child detection (yes / no).
      PLM Call 2 (video) : free-form behavioral description.
      Semantic matching  : sentence-embedding cosine similarity maps the
                           description to vocabulary phrases per feature.
                           No further PLM calls.

    Args:
        desc_max_gen_len: Max tokens for the behavioral description (default 256).
        sim_threshold:    Min cosine similarity to accept a vocabulary match (default 0.20).
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
            f"Assessment: {n_chunks} chunk(s) × 1 PLM description call "
            f"+ sentence-embedding matching (no per-domain PLM calls)"
        )

        # Pre-load NLI model before the chunk loop (avoids first-call overhead)
        _load_nli_model()

        # ---- Assess each segment --------------------------------------------
        chunk_details: list = []
        for i, seg in enumerate(segments):
            t_start = i * chunk_duration if chunk_duration else 0
            t_end   = (i + 1) * chunk_duration if chunk_duration else None
            logger.info(f"Segment {i + 1}/{n_chunks}: {seg}")

            # Step 1: generate behavioral description (the only PLM call)
            logger.info("  Generating behavioral description ...")
            description = _run_plm_text(
                seg, _PROMPT_DESCRIBE_BEHAVIOR, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature,
                max_gen_len=desc_max_gen_len,
            )
            logger.info(
                f"  Description ({len(description)} chars): "
                f"{description[:100]!r}{'...' if len(description) > 100 else ''}"
            )

            # Step 2: semantic match description → domain vocabulary (no PLM)
            domain_scores: dict = {}
            for domain in DOMAINS:
                ds = _match_description_to_domain(
                    description, domain,
                    threshold=sim_threshold, debug=debug,
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
# Custom report — shows description as PLM output, matched phrases per domain
# ------------------------------------------------------------------------------


def _print_chunk_timeline_desc(chunk_details: list) -> None:
    """Print per-chunk description (PLM output) and semantic-matched phrases."""
    if not chunk_details:
        return

    n    = len(chunk_details)
    sep  = "=" * 66
    thin = "-" * 66

    print(f"\n  Per-chunk output ({n} chunk{'s' if n > 1 else ''})")

    for c in chunk_details:
        t_s = int(c["t_start"])
        t_e = int(c["t_end"]) if c["t_end"] is not None else "?"
        time_str = f"{t_s}s – {t_e}s" if c["t_end"] is not None else "full video"

        print(f"\n  {sep}")
        print(f"  Chunk {c['index']}  [{time_str}]")
        print(f"  {sep}")

        # The description IS the PLM output
        description = c.get("description", "")
        print(f"\n  PLM output:")
        if description:
            for line in description.strip().splitlines():
                print(f"    {line}")
        else:
            print("    (description not generated)")

        domain_scores = c.get("domain_scores", {})
        domain_ages   = c.get("domain_ages",   {})

        for domain in DOMAINS:
            ds      = domain_scores.get(domain, {})
            phrases = ds.get("phrases",  {})
            age_d   = domain_ages.get(domain)
            age_tag = f"  [{age_d:.0f}mo]" if age_d is not None else "  [n/a]"

            print(f"\n  [{domain.upper()}]{age_tag}")
            print(f"  {thin}")

            if phrases:
                for feat, phrase in phrases.items():
                    score = ds.get("features", {}).get(feat)
                    score_str = f"  (score {score:.2f})" if score is not None else ""
                    print(f"    {_feat_label(feat)}: {phrase}{score_str}")
            else:
                print("    (no levels matched — below similarity threshold)")

        print()


def print_report_desc(result: dict) -> None:
    """Formatted report for the description-first pipeline."""
    sep  = "=" * 62
    thin = "-" * 62

    child_present = result.get("child_present", False)
    child_str = "YES" if child_present else "NO"

    print(f"\n{sep}")
    print(f"  Developmental Assessment  [description mode]")
    print(f"  {Path(result['video_path']).name}")
    print(f"  Child present: {child_str}")
    print(sep)

    if not child_present:
        print("\n  No child detected in this video.\n")
        print(f"{sep}\n")
        return

    chunks = result.get("chunk_details", [])
    if len(chunks) == 1:
        desc = chunks[0].get("description", "")
        print(f"\n  PLM output:")
        if desc:
            for line in desc.strip().splitlines():
                print(f"    {line}")
        else:
            print("    (description not generated)")
        print(f"  {thin}")
    elif len(chunks) > 1:
        _print_chunk_timeline_desc(chunks)

    print(f"\n{'Domain':<14} {'Feature scores':<30} {'Age est.'}")
    print(thin)

    for domain in DOMAINS:
        features = result["plm_features"].get(domain)
        age      = result["domain_ages"].get(domain)
        age_str  = f"{age:.1f} mo" if age is not None else "n/a"

        if features is None:
            print(f"  {domain:<12} not observed{'':<22} {age_str}")
            continue

        feature_names = _DOMAIN_FEATURES[domain]
        matched_kw    = features.get("matched_keywords", {})
        lines    = []
        kw_lines = []
        for fname in feature_names:
            val = features.get(fname)
            if val is not None:
                lines.append(f"{fname}: {val:.2f} {_bar(val, 10)}")
                kw_lines.append(f"  matched: {matched_kw.get(fname, '?')}")
            else:
                lines.append(f"{fname}: --")
                kw_lines.append("")

        print(f"  {domain:<12} {lines[0]:<32} {age_str}")
        if kw_lines[0]:
            print(f"  {'':<12} {kw_lines[0]}")
        for line, kw_line in zip(lines[1:], kw_lines[1:]):
            print(f"  {'':<12} {line}")
            if kw_line:
                print(f"  {'':<12} {kw_line}")

        evidence = features.get("evidence")
        if evidence:
            import textwrap
            wrapped = textwrap.wrap(str(evidence), width=54)
            print(f"  {'':<12} \033[3mEvidence: {wrapped[0]}\033[0m")
            for w in wrapped[1:]:
                print(f"  {'':<12}           {w}")

    overall = result["overall_age_months"]
    print(f"\n{thin}")
    print(f"  Overall estimated age: "
          f"{'%.1f months' % overall if overall is not None else 'insufficient data'}")

    dist = result["stage_distribution"]
    if dist:
        print(f"\n  Stage distribution:")
        for stage, prob in dist.items():
            label = _STAGE_LABELS.get(stage, stage)
            print(f"    {stage} ({label:<17}) {_bar(prob, 20)} {prob:.3f}")

    print(f"\n{sep}\n")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Developmental assessment: PLM description + sentence-embedding matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --num_frames 16 --chunk_duration 10
  %(prog)s --video clip.mp4 --sim_threshold 0.35   # stricter matching
  %(prog)s --video clip.mp4 --debug                # show per-feature NLI contradiction scores
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
    parser.add_argument("--sim_threshold", type=float, default=0.50,
                        help="Max NLI contradiction score to accept a match (default: 0.50). "
                             "Lower = stricter; raise if too few features matched.")
    parser.add_argument("--chunk_duration", type=float, default=None,
                        help="Split video into chunks of this many seconds.")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Render keyword overlays onto the video and save here.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-feature NLI contradiction scores.")

    args = parser.parse_args()

    logger.info(f"Loading PLM model: {args.ckpt}")
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
        print_report_desc(result)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to: {args.save}")

    if args.output_video:
        render_assessment_video(args.video, result, args.output_video)


if __name__ == "__main__":
    main()
