# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator for children's videos.

Pipeline:
  1. PLM observes behavioral features in 5 domains (motor, autonomy,
     attention, interaction, language) and outputs scores in [0, 1].
  2. Python maps each domain's feature scores to an estimated age in months
     using CDC milestone anchor points at 12 / 18 / 24 / 36 months.
  3. A soft stage distribution (S0–S3) is computed from the age estimates.
  4. A formatted report is printed.

CDC anchor reference:
  S0 → < 12 months   (pre-walker, single words / gestures beginning)
  S1 → 12–18 months  (independent walking, single words, caregiver-dependent)
  S2 → 18–24 months  (running, 2-word phrases, parallel play)
  S3 → 24–36 months  (complex motor, sentences, cooperative play)

Usage:
    python estimate_development.py --video data/202503_a/zdgaa.MOV
    python estimate_development.py --video clip.mp4 --ckpt facebook/Perception-LM-8B
    python estimate_development.py --video clip.mp4 --num_frames 16 --json_only
"""

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_video_description import generate_description

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PLM prompt
# ──────────────────────────────────────────────────────────────────────────────

# Kept short to avoid context-window echo issues.
# PLM outputs feature scores; CDC mapping is done in Python.
_PROMPT = """\
Analyze this child's video. For each domain set "observed" true/false. \
If observed, score each feature in [0,1] based on what you see. \
If not observed, set features to null. Output only JSON:

{"motor":{"observed":bool,"locomotion":float|null,"coordination":float|null,"stability":float|null},\
"autonomy":{"observed":bool,"independence":float|null,"initiative":float|null},\
"attention":{"observed":bool,"duration":float|null,"goal_directed":float|null},\
"interaction":{"observed":bool,"social_engagement":float|null,"caregiver_dependency":float|null},\
"language":{"observed":bool,"verbal":float|null,"gesture":float|null}}\
"""


# ──────────────────────────────────────────────────────────────────────────────
# CDC milestone anchor curves
# Feature score expected at each milestone age (months).
# Interpolating these gives a continuous age estimate from a feature score.
# ──────────────────────────────────────────────────────────────────────────────

# fmt: off
CDC_ANCHORS: dict = {
    "motor": {
        "locomotion":   {0: 0.00, 12: 0.35, 18: 0.60, 24: 0.80, 36: 1.00},
        "coordination": {0: 0.00, 12: 0.30, 18: 0.50, 24: 0.70, 36: 0.95},
        "stability":    {0: 0.00, 12: 0.40, 18: 0.65, 24: 0.85, 36: 1.00},
    },
    "autonomy": {
        "independence": {0: 0.00, 12: 0.10, 18: 0.30, 24: 0.55, 36: 0.85},
        "initiative":   {0: 0.00, 12: 0.15, 18: 0.35, 24: 0.60, 36: 0.90},
    },
    "attention": {
        "duration":      {0: 0.00, 12: 0.20, 18: 0.40, 24: 0.65, 36: 0.90},
        "goal_directed": {0: 0.00, 12: 0.25, 18: 0.45, 24: 0.70, 36: 0.95},
    },
    "interaction": {
        "social_engagement":    {0: 0.00, 12: 0.50, 18: 0.65, 24: 0.75, 36: 0.90},
        # Decreasing: high dependency at birth, low at 36m
        "caregiver_dependency": {0: 1.00, 12: 0.80, 18: 0.65, 24: 0.45, 36: 0.20},
    },
    "language": {
        "verbal":   {0: 0.00, 12: 0.20, 18: 0.40, 24: 0.70, 36: 0.95},
        "gesture":  {0: 0.00, 12: 0.50, 18: 0.75, 24: 0.85, 36: 0.90},
    },
}
# fmt: on

# Stage boundaries in months
STAGE_BOUNDS = {
    "S0": (0, 12),
    "S1": (12, 18),
    "S2": (18, 24),
    "S3": (24, 42),   # 42 = open upper bound approximation
}


# ──────────────────────────────────────────────────────────────────────────────
# CDC mapping helpers
# ──────────────────────────────────────────────────────────────────────────────


def _score_to_age(score: float, curve: dict) -> float:
    """Inverse-interpolate a feature score to an estimated age in months.

    Handles both increasing (most features) and decreasing curves
    (e.g. caregiver_dependency).

    Args:
        score: Observed feature value in [0, 1].
        curve: Dict mapping age_months → expected_score.

    Returns:
        Estimated age in months (float).
    """
    ages = sorted(curve.keys())
    scores = [curve[a] for a in ages]

    increasing = scores[-1] >= scores[0]

    if increasing:
        if score <= scores[0]:
            return float(ages[0])
        if score >= scores[-1]:
            return float(ages[-1])
        for i in range(len(ages) - 1):
            lo, hi = scores[i], scores[i + 1]
            if lo <= score <= hi:
                t = (score - lo) / (hi - lo) if hi != lo else 0.5
                return ages[i] + t * (ages[i + 1] - ages[i])
    else:  # decreasing
        if score >= scores[0]:
            return float(ages[0])
        if score <= scores[-1]:
            return float(ages[-1])
        for i in range(len(ages) - 1):
            hi, lo = scores[i], scores[i + 1]
            if lo <= score <= hi:
                t = (hi - score) / (hi - lo) if hi != lo else 0.5
                return ages[i] + t * (ages[i + 1] - ages[i])

    return float(ages[-1])


def domain_age(domain: str, features: dict) -> Optional[float]:
    """Estimate developmental age (months) for one domain.

    Averages the inverse-interpolated ages across all observed features
    that have a CDC anchor curve defined.

    Args:
        domain: One of motor / autonomy / attention / interaction / language.
        features: Dict of feature_name → score (or None).

    Returns:
        Estimated age in months, or None if no scorable features.
    """
    curves = CDC_ANCHORS.get(domain, {})
    ages = []
    for feature, score in features.items():
        if feature == "observed" or score is None:
            continue
        if feature not in curves:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        ages.append(_score_to_age(score, curves[feature]))

    return sum(ages) / len(ages) if ages else None


def stage_distribution(age_months: float, sigma: float = 4.0) -> dict:
    """Compute a soft S0–S3 stage distribution from a continuous age estimate.

    Uses a Gaussian centred at age_months with std=sigma, integrated over
    each stage's age range, then normalised.

    Args:
        age_months: Point estimate of developmental age.
        sigma: Uncertainty spread in months (default: 4.0).

    Returns:
        Dict mapping stage label → probability (sums to 1.0).
    """
    def _gauss_integral(a, b, mu, sig):
        """Integral of N(mu, sig) from a to b using erf."""
        from math import erf, sqrt
        z = lambda x: (x - mu) / (sig * sqrt(2))
        return 0.5 * (erf(z(b)) - erf(z(a)))

    raw = {}
    for stage, (lo, hi) in STAGE_BOUNDS.items():
        raw[stage] = _gauss_integral(lo, hi, age_months, sigma)

    total = sum(raw.values()) or 1.0
    return {s: round(v / total, 4) for s, v in raw.items()}


# ──────────────────────────────────────────────────────────────────────────────
# PLM inference + JSON extraction
# ──────────────────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from a (possibly noisy) PLM output."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def run_plm(video_path: str, model, tokenizer, config,
            num_frames: int = 8, temperature: float = 0.0,
            max_gen_len: int = 512) -> Optional[dict]:
    """Run PLM on a video and return the parsed domain feature dict."""
    result = generate_description(
        video_path=video_path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=_PROMPT,
        num_frames=num_frames,
        temperature=temperature,
        max_gen_len=max_gen_len,
    )
    raw_text = result["description"]
    parsed = _extract_json(raw_text)

    if parsed is None:
        logger.warning("Could not parse JSON from PLM output.")
        logger.debug(f"Raw output:\n{raw_text}")

    return parsed, raw_text


# ──────────────────────────────────────────────────────────────────────────────
# Full assessment
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS = ["motor", "autonomy", "attention", "interaction", "language"]


def assess(video_path: str, model, tokenizer, config,
           num_frames: int = 8, temperature: float = 0.0,
           max_gen_len: int = 512) -> dict:
    """Run the full developmental assessment pipeline on a video.

    Returns:
        Dict with keys: video_path, plm_features, domain_ages,
        overall_age_months, stage_distribution, raw_plm_output.
    """
    parsed, raw_text = run_plm(
        video_path, model, tokenizer, config,
        num_frames=num_frames, temperature=temperature, max_gen_len=max_gen_len,
    )

    domain_ages = {}
    plm_features = {}

    if parsed:
        for domain in DOMAINS:
            domain_data = parsed.get(domain, {})
            observed = domain_data.get("observed", False)
            if not observed:
                domain_ages[domain] = None
                plm_features[domain] = None
                continue
            # Extract feature scores (exclude the "observed" key)
            features = {k: v for k, v in domain_data.items() if k != "observed"}
            plm_features[domain] = features
            domain_ages[domain] = domain_age(domain, features)
    else:
        for domain in DOMAINS:
            domain_ages[domain] = None
            plm_features[domain] = None

    # Overall age: mean of observed domains
    observed_ages = [a for a in domain_ages.values() if a is not None]
    overall_age = sum(observed_ages) / len(observed_ages) if observed_ages else None

    stage_dist = stage_distribution(overall_age) if overall_age is not None else None

    return {
        "video_path": video_path,
        "plm_features": plm_features,
        "domain_ages": domain_ages,
        "overall_age_months": round(overall_age, 1) if overall_age else None,
        "stage_distribution": stage_dist,
        "raw_plm_output": raw_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Formatted report
# ──────────────────────────────────────────────────────────────────────────────

_STAGE_LABELS = {
    "S0": "< 12 months",
    "S1": "12–18 months",
    "S2": "18–24 months",
    "S3": "24–36+ months",
}

_DOMAIN_FEATURES = {
    "motor":       ["locomotion", "coordination", "stability"],
    "autonomy":    ["independence", "initiative"],
    "attention":   ["duration", "goal_directed"],
    "interaction": ["social_engagement", "caregiver_dependency"],
    "language":    ["verbal", "gesture"],
}


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(result: dict) -> None:
    sep = "=" * 62
    thin = "-" * 62

    print(f"\n{sep}")
    print(f"  Developmental Assessment")
    print(f"  {Path(result['video_path']).name}")
    print(sep)

    # Per-domain table
    print(f"\n{'Domain':<14} {'Features (PLM scores)':<30} {'Age est.'}")
    print(thin)

    for domain in DOMAINS:
        features = result["plm_features"].get(domain)
        age = result["domain_ages"].get(domain)
        age_str = f"{age:.1f} mo" if age is not None else "n/a"

        if features is None:
            print(f"  {domain:<12} not observed{'':<22} {age_str}")
        else:
            feature_names = _DOMAIN_FEATURES.get(domain, list(features.keys()))
            lines = []
            for fname in feature_names:
                val = features.get(fname)
                if val is not None:
                    lines.append(f"{fname}: {val:.2f} {_bar(val, 10)}")
                else:
                    lines.append(f"{fname}: null")

            # First feature line on same row as domain name
            print(f"  {domain:<12} {lines[0]:<32} {age_str}")
            for line in lines[1:]:
                print(f"  {'':<12} {line}")

    # Overall age
    overall = result["overall_age_months"]
    print(f"\n{thin}")
    print(f"  Overall estimated age: "
          f"{'%.1f months' % overall if overall else 'insufficient data'}")

    # Stage distribution
    dist = result["stage_distribution"]
    if dist:
        print(f"\n  Stage distribution:")
        for stage, prob in dist.items():
            label = _STAGE_LABELS.get(stage, stage)
            print(f"    {stage} ({label:<17}) {_bar(prob, 20)} {prob:.3f}")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a child's developmental stage from a video using PLM + CDC milestones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video data/202503_a/zdgaa.MOV
  %(prog)s --video clip.mp4 --ckpt facebook/Perception-LM-8B --num_frames 16
  %(prog)s --video clip.mp4 --json_only > result.json
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the child's video.")
    parser.add_argument("--ckpt", type=str, default="facebook/Perception-LM-3B",
                        help="PLM checkpoint or HuggingFace ID.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample from the video (default: 8).")
    parser.add_argument("--max_gen_len", type=int, default=512,
                        help="Max tokens to generate (default: 512).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 = greedy (default: 0.0).")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")

    args = parser.parse_args()

    logger.info(f"Loading model: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    result = assess(
        video_path=args.video,
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_frames=args.num_frames,
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
    )

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
        print("Raw PLM output:")
        print(result["raw_plm_output"])

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
