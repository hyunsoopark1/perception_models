# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator — description-first pipeline.

Pipeline (per chunk):
  1. PLM Call 1 : Detect whether a child is visible (yes / no).
  2. PLM Call 2 : Generate a free-form description of the child (the PLM output).
  3. PLM Calls 3+: For each domain, give PLM the description as context and an
                   ordered vocabulary list; PLM selects ONE phrase per feature.
  4. Aggregate across chunks, inverse-interpolate CDC ages, print report.

PLM calls per chunk: 1 (description) + N_domains (matching) = 6 total.
No majority voting — the description grounds the selection reliably.

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
from estimate_development import (
    _FEATURE_LEVELS, _DECREASING_FEATURES, _DOMAIN_DESCRIPTIONS, _DOMAIN_FEATURES,
    DOMAINS, _feat_label, domain_age, stage_distribution,
    _parse_domain_output, _aggregate_domain_scores,
    _split_into_chunks, _transcode_to_h264, _run_plm_text,
    _STAGE_LABELS, _bar, render_assessment_video,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------------------

_PROMPT_CHILD = (
    "Is there a child (infant or toddler under 4 years old) clearly visible "
    "and active in this video? Answer only: yes or no."
)

_PROMPT_DESCRIBE = "Describe what the child is doing in this video in detail."


# ------------------------------------------------------------------------------
# Domain prompt (description-grounded, numbered options to prevent echo)
# ------------------------------------------------------------------------------


def _build_domain_prompt(domain: str, description: str) -> str:
    """Ordered vocabulary prompt with the pre-generated description as context.

    Options are presented as a numbered list so the model copies the exact
    phrase rather than echoing the full option string.
    """
    features   = _DOMAIN_FEATURES[domain]
    desc_label = _DOMAIN_DESCRIPTIONS[domain]

    option_blocks = []
    for feat in features:
        levels  = _FEATURE_LEVELS[feat]
        ordered = levels if feat in _DECREASING_FEATURES else sorted(levels, key=lambda x: x[0])
        lines = [f"  {_feat_label(feat)} (youngest → most developed):"]
        for idx, (_, phrase) in enumerate(ordered, start=1):
            lines.append(f"    {idx}. {phrase}")
        option_blocks.append("\n".join(lines))

    answer_block = "\n".join(
        f"{_feat_label(feat)}: <copy exact phrase>" for feat in features
    )

    return (
        f"The child in this video has been described as:\n"
        f"\"{description}\"\n\n"
        f"Based on this description and what you observe, for each {desc_label} skill "
        f"copy the ONE phrase that BEST MATCHES the child's behavior.\n"
        f"- Copy the phrase verbatim from the numbered list.\n"
        f"- If the skill is not visible, write: not visible\n\n"
        + "\n\n".join(option_blocks)
        + f"\n\nReply (one line per skill, phrase copied verbatim):\n"
        + answer_block
    )


# ------------------------------------------------------------------------------
# Single domain assessment
# ------------------------------------------------------------------------------


def _assess_domain(
    seg: str, domain: str, description: str,
    model, tokenizer, config,
    num_frames: int, temperature: float, max_gen_len: int,
    debug: bool = False,
) -> dict:
    prompt = _build_domain_prompt(domain, description)
    txt = _run_plm_text(
        seg, prompt, model, tokenizer, config,
        num_frames=num_frames, temperature=temperature, max_gen_len=max_gen_len,
    )
    # Strip number prefix the model may add (e.g. "4. phrase" → "phrase")
    txt_clean = re.sub(r'(:\s*)\d+\.\s*', r'\1', txt)
    parsed = _parse_domain_output(domain, txt_clean)

    if debug:
        logger.info(f"    [{domain}] raw:    {txt!r}")
        logger.info(f"    [{domain}] parsed: {parsed}")

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
# Full assessment
# ------------------------------------------------------------------------------


def assess_desc(
    video_path: str, model, tokenizer, config,
    num_frames: int = 8, temperature: float = 0.0,
    max_gen_len: int = 128, desc_max_gen_len: int = 256,
    chunk_duration: Optional[float] = None,
    debug: bool = False,
) -> dict:
    import os, tempfile

    tmp_path: Optional[str] = None
    chunk_paths: list = []
    plm_path = video_path

    try:
        # ---- Child detection ------------------------------------------------
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
        logger.info(f"Child present: {child_present!r}  (PLM: {child_text!r})")

        if not child_present:
            return {
                "video_path": video_path, "child_present": False,
                "plm_features": {}, "domain_ages": {},
                "overall_age_months": None, "stage_distribution": None,
                "raw_plm_output": child_text,
            }

        # ---- Segments -------------------------------------------------------
        if chunk_duration:
            chunk_paths = _split_into_chunks(plm_path, chunk_duration)
            segments = chunk_paths
        else:
            segments = [plm_path]

        n_chunks = len(segments)
        logger.info(
            f"Assessment: {n_chunks} chunk(s) × "
            f"(1 description + {len(DOMAINS)} domain calls) = "
            f"{n_chunks * (1 + len(DOMAINS))} PLM calls"
        )

        # ---- Per-segment assessment -----------------------------------------
        chunk_details: list = []
        for i, seg in enumerate(segments):
            t_start = i * chunk_duration if chunk_duration else 0
            t_end   = (i + 1) * chunk_duration if chunk_duration else None
            logger.info(f"Segment {i + 1}/{n_chunks}")

            # Step 1: description (the PLM output shown to the user)
            description = _run_plm_text(
                seg, _PROMPT_DESCRIBE, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature,
                max_gen_len=desc_max_gen_len,
            )
            logger.info(f"  Description ({len(description)} chars): {description[:80]!r}")

            if not description:
                logger.warning("  Description empty — skipping domain assessment for this chunk.")

            # Step 2: domain selection using description as context
            domain_scores: dict = {}
            for domain in DOMAINS:
                if not description:
                    domain_scores[domain] = {
                        "features": {}, "phrases": {}, "age": None, "raw_outputs": []
                    }
                else:
                    domain_scores[domain] = _assess_domain(
                        seg, domain, description, model, tokenizer, config,
                        num_frames=num_frames, temperature=temperature,
                        max_gen_len=max_gen_len, debug=debug,
                    )

            chunk_details.append({
                "index":         i + 1,
                "t_start":       t_start,
                "t_end":         t_end,
                "description":   description,
                "domain_scores": domain_scores,
                "domain_ages":   {d: domain_scores[d]["age"] for d in DOMAINS},
                "raw_text":      "\n".join(
                    (f"{d}: " + ", ".join(f"{f}={p}" for f, p in domain_scores[d]["phrases"].items()))
                    if domain_scores[d]["phrases"] else f"{d}: none"
                    for d in DOMAINS
                ),
            })

        # ---- Aggregate ------------------------------------------------------
        plm_features: dict = {}
        domain_ages:  dict = {}

        for domain in DOMAINS:
            per_chunk = [c["domain_scores"][domain] for c in chunk_details]
            agg = _aggregate_domain_scores(per_chunk, domain)
            if agg["features"]:
                entry = dict(agg["features"])
                entry["evidence"]         = ", ".join(f"{f}={p}" for f, p in agg["phrases"].items())
                entry["matched_keywords"] = dict(agg["phrases"])
                plm_features[domain] = entry
            else:
                plm_features[domain] = None
            domain_ages[domain] = agg["age"]

        observed = [a for a in domain_ages.values() if a is not None]
        overall  = sum(observed) / len(observed) if observed else None
        stage_dist = stage_distribution(overall) if overall is not None else None

        return {
            "video_path":         video_path,
            "child_present":      True,
            "plm_features":       plm_features,
            "domain_ages":        domain_ages,
            "overall_age_months": round(overall, 1) if overall is not None else None,
            "stage_distribution": stage_dist,
            "raw_plm_output":     "\n\n".join(c["description"] for c in chunk_details),
            "chunk_details":      chunk_details,
        }

    finally:
        for p in chunk_paths:
            try: os.unlink(p)
            except OSError: pass
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass


# ------------------------------------------------------------------------------
# Report — description is "PLM output", domain matching is internal
# ------------------------------------------------------------------------------


def _print_chunk_timeline(chunk_details: list) -> None:
    n   = len(chunk_details)
    sep = "=" * 66
    thin= "-" * 66
    print(f"\n  Per-chunk output ({n} chunk{'s' if n > 1 else ''})")

    for c in chunk_details:
        t_s = int(c["t_start"])
        t_e = int(c["t_end"]) if c["t_end"] is not None else "?"
        time_str = f"{t_s}s – {t_e}s" if c["t_end"] is not None else "full video"

        print(f"\n  {sep}")
        print(f"  Chunk {c['index']}  [{time_str}]")
        print(f"  {sep}")

        desc = c.get("description", "")
        print(f"\n  PLM output:")
        if desc:
            for line in desc.strip().splitlines():
                print(f"    {line}")
        else:
            print("    (description not generated)")

        domain_scores = c.get("domain_scores", {})
        domain_ages   = c.get("domain_ages",   {})

        for domain in DOMAINS:
            ds      = domain_scores.get(domain, {})
            phrases = ds.get("phrases", {})
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
                print("    (no levels matched)")
        print()


def print_report(result: dict) -> None:
    sep  = "=" * 62
    thin = "-" * 62

    child_present = result.get("child_present", False)
    print(f"\n{sep}")
    print(f"  Developmental Assessment  [description mode]")
    print(f"  {Path(result['video_path']).name}")
    print(f"  Child present: {'YES' if child_present else 'NO'}")
    print(sep)

    if not child_present:
        print("\n  No child detected.\n")
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
        _print_chunk_timeline(chunks)

    print(f"\n{'Domain':<14} {'Feature scores':<30} {'Age est.'}")
    print(thin)

    for domain in DOMAINS:
        features = result["plm_features"].get(domain)
        age      = result["domain_ages"].get(domain)
        age_str  = f"{age:.1f} mo" if age is not None else "n/a"

        if features is None:
            print(f"  {domain:<12} not observed{'':<22} {age_str}")
            continue

        matched_kw = features.get("matched_keywords", {})
        lines, kw_lines = [], []
        for fname in _DOMAIN_FEATURES[domain]:
            val = features.get(fname)
            if val is not None:
                lines.append(f"{fname}: {val:.2f} {_bar(val, 10)}")
                kw_lines.append(f"  matched: {matched_kw.get(fname, '?')}")
            else:
                lines.append(f"{fname}: --")
                kw_lines.append("")

        print(f"  {domain:<12} {lines[0]:<32} {age_str}")
        if kw_lines[0]: print(f"  {'':<12} {kw_lines[0]}")
        for line, kw_line in zip(lines[1:], kw_lines[1:]):
            print(f"  {'':<12} {line}")
            if kw_line: print(f"  {'':<12} {kw_line}")

        evidence = features.get("evidence")
        if evidence:
            import textwrap
            for i, w in enumerate(textwrap.wrap(str(evidence), 54)):
                prefix = "Evidence: " if i == 0 else "          "
                print(f"  {'':<12} \033[3m{prefix}{w}\033[0m")

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
        description="Developmental assessment: PLM description then PLM domain matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --num_frames 16 --chunk_duration 10
  %(prog)s --video clip.mp4 --debug
        """,
    )
    parser.add_argument("--video",            type=str,   required=True)
    parser.add_argument("--ckpt",             type=str,   default="facebook/Perception-LM-3B")
    parser.add_argument("--num_frames",       type=int,   default=8)
    parser.add_argument("--max_gen_len",      type=int,   default=128,
                        help="Max tokens for domain selection calls (default: 128).")
    parser.add_argument("--desc_max_gen_len", type=int,   default=256,
                        help="Max tokens for description call (default: 256).")
    parser.add_argument("--temperature",      type=float, default=0.0)
    parser.add_argument("--chunk_duration",   type=float, default=None)
    parser.add_argument("--json_only",        action="store_true")
    parser.add_argument("--save",             type=str,   default=None)
    parser.add_argument("--output_video",     type=str,   default=None)
    parser.add_argument("--debug",            action="store_true")

    args = parser.parse_args()

    logger.info(f"Loading model: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    result = assess_desc(
        video_path=args.video, model=model, tokenizer=tokenizer, config=config,
        num_frames=args.num_frames, temperature=args.temperature,
        max_gen_len=args.max_gen_len, desc_max_gen_len=args.desc_max_gen_len,
        chunk_duration=args.chunk_duration, debug=args.debug,
    )

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to: {args.save}")

    if args.output_video:
        render_assessment_video(args.video, result, args.output_video)


if __name__ == "__main__":
    main()
