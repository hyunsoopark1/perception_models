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
Watch this child's video carefully and observe what they actually do.

For each domain, describe the specific action you see in *_evidence \
(e.g. "walks steadily across room", "reaches for toy alone", "points at dog"). \
Never write vague phrases like "child present" — describe the observed behavior. \
If a domain is not visible, set *_observed=false and all its fields to null.

Score anchors (CDC milestones): 0.0=not yet present, 0.35=12mo, 0.6=18mo, 0.8=24mo, 1.0=36mo+.

Motor — look for: walking, running, climbing, grasping, throwing.
Autonomy — look for: reaching/acting without help, exploring independently.
Attention — look for: sustained gaze, following objects, goal-directed play.
Interaction — look for: eye contact, responding to others, seeking caregiver.
Language — look for: babbling, words, pointing, waving, gestures.

Output only this flat JSON with real values — no placeholders, no extra text:

{"child_present":bool,\
"motor_observed":bool,"locomotion":float|null,"coordination":float|null,"stability":float|null,"motor_evidence":string|null,\
"autonomy_observed":bool,"independence":float|null,"initiative":float|null,"autonomy_evidence":string|null,\
"attention_observed":bool,"duration":float|null,"goal_directed":float|null,"attention_evidence":string|null,\
"interaction_observed":bool,"social_engagement":float|null,"caregiver_dependency":float|null,"interaction_evidence":string|null,\
"language_observed":bool,"verbal":float|null,"gesture":float|null,"language_evidence":string|null}\
"""

# Mapping from flat JSON keys back to (domain, feature) structure
_DOMAIN_MAP = {
    "motor":       {"observed_key": "motor_observed",       "evidence_key": "motor_evidence",       "features": ["locomotion", "coordination", "stability"]},
    "autonomy":    {"observed_key": "autonomy_observed",    "evidence_key": "autonomy_evidence",    "features": ["independence", "initiative"]},
    "attention":   {"observed_key": "attention_observed",   "evidence_key": "attention_evidence",   "features": ["duration", "goal_directed"]},
    "interaction": {"observed_key": "interaction_observed", "evidence_key": "interaction_evidence", "features": ["social_engagement", "caregiver_dependency"]},
    "language":    {"observed_key": "language_observed",    "evidence_key": "language_evidence",    "features": ["verbal", "gesture"]},
}


def _normalise_keys(flat: dict) -> dict:
    """Normalise PLM dict keys: replace spaces with underscores."""
    return {k.replace(" ", "_"): v for k, v in flat.items()}


def _flat_to_domains(flat: dict) -> dict:
    """Reconstruct nested domain dict from the flat PLM output."""
    flat = _normalise_keys(flat)
    domains = {}
    for domain, cfg in _DOMAIN_MAP.items():
        observed = flat.get(cfg["observed_key"], False)
        if isinstance(observed, (int, float)):
            observed = bool(observed)
        entry = {"observed": observed}
        for feat in cfg["features"]:
            entry[feat] = flat.get(feat) if observed else None
        entry["evidence"] = flat.get(cfg["evidence_key"]) if observed else None
        domains[domain] = entry
    return domains


def _regex_child_present(text: str) -> bool:
    """Regex fallback: extract child_present from raw text when JSON parse fails."""
    m = re.search(r"['\"]child_present['\"]\s*:\s*(true|false)", text, re.IGNORECASE)
    return m.group(1).lower() == "true" if m else False


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


def _transcode_to_h264(video_path: str, tmp_path: str) -> str:
    """Re-encode a video to H.264 MP4 using OpenCV so torchcodec can decode it.

    iPhone MOV files use HEVC/H.265 with non-standard NAL unit structures that
    torchcodec cannot decode. Reading with OpenCV (FFmpeg backend) and writing
    as plain H.264 produces a file that torchcodec handles correctly.

    Rotation metadata is baked into the frames during re-encoding (same logic
    as split_videos.py) so the output clip is already upright.

    Args:
        video_path: Source video path (any format OpenCV can read).
        tmp_path: Destination path for the re-encoded file.

    Returns:
        tmp_path on success.

    Raises:
        RuntimeError: If OpenCV cannot open the source video.
    """
    import cv2
    import numpy as np
    from split_videos import _apply_rotation, _get_rotation

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation = _get_rotation(cap)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(_apply_rotation(frame, rotation))

    cap.release()
    writer.release()
    logger.info(f"  Re-encoded {Path(video_path).name} → {Path(tmp_path).name} "
                f"({out_w}×{out_h}, rotation={rotation}°)")
    return tmp_path


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON/dict object from PLM output.

    LLMs often produce Python-dict syntax (single quotes, True/False/None)
    instead of strict JSON. This function tries multiple normalisation
    strategies before giving up.
    """
    import ast

    def _try_json(s):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    def _try_ast(s):
        # Normalise JSON literals → Python literals for ast.literal_eval
        s = re.sub(r':\s*true\b',  ': True',  s)
        s = re.sub(r':\s*false\b', ': False', s)
        s = re.sub(r':\s*null\b',  ': None',  s)
        try:
            result = ast.literal_eval(s)
            return result if isinstance(result, dict) else None
        except (ValueError, SyntaxError):
            return None

    # 1. Direct JSON parse (model produced valid JSON)
    result = _try_json(text.strip())
    if result is not None:
        return result

    # 2. Extract first {...} block, then try JSON
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    block = match.group()

    result = _try_json(block)
    if result is not None:
        return result

    # 3. ast.literal_eval on the block (handles single-quoted Python dicts)
    result = _try_ast(block)
    if result is not None:
        return result

    # 4. Replace single quotes → double quotes as last resort
    try:
        double_quoted = block.replace("'", '"')
        result = _try_json(double_quoted)
        if result is not None:
            return result
    except Exception:
        pass

    return None


def run_plm(video_path: str, model, tokenizer, config,
            num_frames: int = 8, temperature: float = 0.0,
            max_gen_len: int = 512) -> Optional[dict]:
    """Run PLM on a video and return the parsed domain feature dict.

    Automatically re-encodes the video to H.264 if torchcodec fails to decode
    the original (common with iPhone HEVC recordings).
    """
    import tempfile
    import os

    def _run(path):
        result = generate_description(
            video_path=path,
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompt=_PROMPT,
            num_frames=num_frames,
            temperature=temperature,
            max_gen_len=max_gen_len,
        )
        return result["description"]

    # First attempt with the original file
    try:
        raw_text = _run(video_path)
        return _extract_json(raw_text), raw_text
    except RuntimeError as e:
        if "decoder" not in str(e).lower() and "NAL" not in str(e):
            raise
        logger.warning(
            f"torchcodec could not decode {Path(video_path).name} "
            f"({e}). Re-encoding to H.264 and retrying..."
        )

    # Fallback: transcode to a temp H.264 file and retry
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _transcode_to_h264(video_path, tmp_path)
        raw_text = _run(tmp_path)
        return _extract_json(raw_text), raw_text
    finally:
        os.unlink(tmp_path)


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
        # PLM now outputs a flat dict; reconstruct nested domain structure
        nested = _flat_to_domains(parsed)
        for domain in DOMAINS:
            domain_data = nested.get(domain, {})
            observed = domain_data.get("observed", False)
            if not observed:
                domain_ages[domain] = None
                plm_features[domain] = None
                continue
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

    child_present = (
        bool(parsed.get("child_present", False))
        if parsed else _regex_child_present(raw_text)
    )

    return {
        "video_path": video_path,
        "child_present": child_present,
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

    child_present = result.get("child_present", False)
    child_str = "YES" if child_present else "NO"

    print(f"\n{sep}")
    print(f"  Developmental Assessment")
    print(f"  {Path(result['video_path']).name}")
    print(f"  Child present: {child_str}")
    print(sep)

    if not child_present:
        print("\n  No child detected in this video.\n")
        print(f"{sep}\n")
        return

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
            feature_names = _DOMAIN_FEATURES.get(domain, [k for k in features if k != "evidence"])
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

            # Evidence
            evidence = features.get("evidence")
            if evidence:
                # Wrap to 54 chars so it fits within the report width
                import textwrap
                wrapped = textwrap.wrap(str(evidence), width=54)
                print(f"  {'':<12} \033[3mEvidence: {wrapped[0]}\033[0m")
                for w in wrapped[1:]:
                    print(f"  {'':<12}           {w}")

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
    parser.add_argument("--max_gen_len", type=int, default=1024,
                        help="Max tokens to generate (default: 1024).")
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
