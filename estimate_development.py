# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator for children's videos.

Pipeline:
  1. PLM Call 1: Detect whether a child is visible (yes / no).
  2. PLM Call 2: Free-text description of observed behaviour per domain.
  3. Python keyword matching maps each domain description to CDC feature scores.
  4. Inverse-interpolation of CDC milestone curves yields age estimates.
  5. A soft S0-S3 stage distribution is computed and a formatted report is printed.

CDC anchor reference:
  S0 -> < 12 months   (pre-walker, gestures beginning)
  S1 -> 12-18 months  (independent walking, single words)
  S2 -> 18-24 months  (running, 2-word phrases, parallel play)
  S3 -> 24-36 months  (complex motor, sentences, cooperative play)

Usage:
    python estimate_development.py --video data/202503_a/zdgaa.MOV
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


# ------------------------------------------------------------------------------
# PLM prompts
# ------------------------------------------------------------------------------

_PROMPT_CHILD = (
    "Is there a child (infant or toddler under 4 years old) clearly visible "
    "and active in this video? Answer only: yes or no."
)

# Ask for labelled free-text sections - PLM describes, Python scores.
# Kept short to avoid context-window overflow on PLM-3B.
# Note: include passive behaviours (sitting, looking) so the model does not
# default to "not observed" for a stationary child.
_PROMPT_DESCRIBE = """\
Watch this child carefully. For each domain write 1-2 sentences about \
what you actually see — even small or quiet actions count. \
Only write "not observed" if the domain is completely invisible.

Motor: Is the child sitting, standing, walking, running, reaching, grasping?
Autonomy: Does the child reach or explore without help? Feed itself?
Attention: What does the child look at? How long does it stay focused?
Interaction: Does the child make eye contact, look at adults, seek the caregiver?
Language: Any babbling, words, pointing, waving, or other gestures?\
"""


# ------------------------------------------------------------------------------
# Keyword tables  (feature -> [(score, [phrase, ...]), ...])
#
# Score anchors (CDC milestones): 0.35=12mo, 0.60=18mo, 0.80=24mo, 1.00=36mo.
# _score_feature() returns the *highest* score whose phrases appear in the text.
# Phrases use simple substring matching (case-insensitive).
# caregiver_dependency is a DECREASING curve (more dependent = higher score).
# ------------------------------------------------------------------------------

_KEYWORDS: dict = {
    "locomotion": [
        (1.00, ["hopping", "galloping", "jumping", "jumps", "skipping", "runs confidently"]),
        (0.80, ["running", "runs", "climbs stairs", "walks well", "walks steadily",
                "walks independently", "walks without"]),
        (0.60, ["walking", "walks", "toddling", "toddles", "walks around", "takes steps"]),
        (0.35, ["cruising", "pulling to stand", "pulls to stand", "first steps",
                "stands with support", "unsteady steps"]),
        (0.10, ["crawling", "crawls", "creeping", "scooting"]),
        (0.00, ["lying", "stationary", "does not walk", "seated only"]),
    ],
    "coordination": [
        (1.00, ["scissors", "draws circle", "catches ball", "strings beads", "buttons"]),
        (0.80, ["kicks ball", "turns pages", "builds tower", "stacks blocks", "uses fork"]),
        (0.60, ["throws", "scribbles", "uses spoon", "picks up small", "stacks", "pours"]),
        (0.35, ["grasps", "pincer", "reaches for", "picks up", "holds toy", "transfers"]),
    ],
    "stability": [
        (1.00, ["stands on one foot", "hops on one foot", "balances on one", "excellent balance"]),
        (0.80, ["tiptoe", "walks on tiptoe", "steady balance", "good balance", "balances briefly"]),
        (0.60, ["stands alone", "stands independently", "steady on feet", "walks without falling"]),
        (0.35, ["sits independently", "sits alone", "sitting", "sitting up", "pulls to stand",
                "stands briefly", "standing"]),
    ],
    "independence": [
        (1.00, ["dresses independently", "fully independent", "toilet", "brushes teeth"]),
        (0.80, ["removes shoes", "washes hands", "partially dresses", "puts on clothing"]),
        (0.60, ["feeds self", "drinks from cup", "uses spoon alone", "eats independently",
                "self-feeds", "self feeds"]),
        (0.35, ["reaches for toy", "picks up food", "explores nearby", "grabs object"]),
    ],
    "initiative": [
        (1.00, ["complex pretend", "elaborate play", "self-directed", "plans activity",
                "sequential", "organizes play"]),
        (0.80, ["pretend play", "makes choices", "problem solv", "selects toy",
                "leads play", "starts game"]),
        (0.60, ["initiates play", "chooses toy", "opens container", "starts activity",
                "initiates activity"]),
        (0.35, ["initiates reaching", "explores independently", "moves toward", "approaches"]),
    ],
    "duration": [
        (1.00, ["prolonged focus", "sustained engagement", "extended attention",
                "maintains attention", "long period"]),
        (0.80, ["extended play", "focused activity", "stays engaged", "continues playing",
                "concentrates"]),
        (0.60, ["sustained attention", "plays with toy", "attends for several",
                "focused for", "watches attentively"]),
        (0.35, ["briefly attends", "momentary attention", "looks at toy briefly",
                "short attention", "glances", "looks at", "gazes", "watches", "observes"]),
    ],
    "goal_directed": [
        (1.00, ["plans ahead", "sequential actions", "multi-step", "complex problem",
                "organized play"]),
        (0.80, ["completes task", "works to finish", "purposefully arranges", "solves problem"]),
        (0.60, ["purposeful play", "works toward goal", "tries to achieve", "persists"]),
        (0.35, ["reaches for specific", "follows object", "tracks toy", "pursues toy"]),
    ],
    "social_engagement": [
        (1.00, ["cooperative play", "takes turns", "plays with other children",
                "group play", "shares toys"]),
        (0.80, ["plays alongside", "shows affection", "parallel play with interaction",
                "brings toy to", "shows toy to"]),
        (0.60, ["parallel play", "makes eye contact", "shows objects", "imitates",
                "waves at", "responds to"]),
        (0.35, ["responds to name", "smiles at", "turns to voice", "reacts to adult"]),
    ],
    # Decreasing curve: higher score = more caregiver-dependent = younger child
    "caregiver_dependency": [
        (0.80, ["clings to", "cries for caregiver", "separation anxiety",
                "distressed without", "won't leave caregiver", "needs caregiver constantly"]),
        (0.65, ["seeks caregiver", "returns to caregiver", "checks on caregiver",
                "looks to caregiver", "stays near adult", "keeps close to"]),
        (0.45, ["occasionally checks", "glances at caregiver", "aware of caregiver",
                "looks back at"]),
        (0.20, ["plays independently", "ignores caregiver", "fully independent from",
                "comfortable away", "does not seek"]),
    ],
    "verbal": [
        (1.00, ["sentences", "full sentence", "three-word", "conversation", "storytelling",
                "talks in"]),
        (0.80, ["two-word", "2-word", "combining words", "word combinations", "short phrases"]),
        (0.60, ["several words", "multiple words", "vocabulary", "names objects",
                "says words", "many words"]),
        (0.35, ["babbling", "babbles", "single word", "mama", "dada", "first words",
                "one word", "jargon", "vocalizes"]),
        (0.00, ["no words", "no speech", "silent", "no verbal", "does not speak"]),
    ],
    "gesture": [
        (1.00, ["rich gestures", "complex gestures", "gestures with speech", "mime",
                "elaborate gesture"]),
        (0.80, ["gestures to communicate", "uses gestures", "points to show",
                "shows object", "symbolic gesture"]),
        (0.60, ["points", "pointing", "uses pointing"]),
        (0.35, ["waves", "waving", "arms up", "reaching gesture", "claps", "shakes head"]),
        (0.00, ["no gesture", "no pointing", "no waving", "does not gesture"]),
    ],
}


# ------------------------------------------------------------------------------
# CDC milestone anchor curves
# ------------------------------------------------------------------------------

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
        "caregiver_dependency": {0: 1.00, 12: 0.80, 18: 0.65, 24: 0.45, 36: 0.20},
    },
    "language": {
        "verbal":  {0: 0.00, 12: 0.20, 18: 0.40, 24: 0.70, 36: 0.95},
        "gesture": {0: 0.00, 12: 0.50, 18: 0.75, 24: 0.85, 36: 0.90},
    },
}
# fmt: on

STAGE_BOUNDS = {
    "S0": (0, 12),
    "S1": (12, 18),
    "S2": (18, 24),
    "S3": (24, 42),
}

DOMAINS = ["motor", "autonomy", "attention", "interaction", "language"]

_DOMAIN_FEATURES = {
    "motor":       ["locomotion", "coordination", "stability"],
    "autonomy":    ["independence", "initiative"],
    "attention":   ["duration", "goal_directed"],
    "interaction": ["social_engagement", "caregiver_dependency"],
    "language":    ["verbal", "gesture"],
}


# ------------------------------------------------------------------------------
# Keyword scoring helpers
# ------------------------------------------------------------------------------


def _score_feature(text: str, feature: str) -> Optional[float]:
    """Return the highest keyword-matched CDC score for a feature.

    Scans all phrase lists for *feature* and returns the maximum score
    whose phrases appear (case-insensitive substring match) in *text*.
    Returns None if no phrase matches (feature not observable in text).
    """
    entries = _KEYWORDS.get(feature, [])
    best: Optional[float] = None
    text_lower = text.lower()
    for score, phrases in entries:
        for phrase in phrases:
            if phrase.lower() in text_lower:
                if best is None or score > best:
                    best = score
                break  # one phrase per score tier is enough
    return best


def _extract_domain_sections(description: str) -> dict:
    """Split PLM free-text description into per-domain text sections.

    Looks for labelled headers (Motor:, Autonomy:, ...) that the prompt
    requests and splits the output into per-domain strings.
    Falls back to the full description for all domains if no headers found.
    """
    header_pattern = re.compile(
        r"(?:^|\n)\s*(motor|autonomy|attention|interaction|language)\s*:",
        re.IGNORECASE,
    )
    parts = header_pattern.split(description)
    # parts = [pre_text, label1, body1, label2, body2, ...]

    if len(parts) < 3:
        # No headers found; use the full text for every domain
        full = description.strip()
        return {d: full for d in DOMAINS}

    sections = {}
    i = 1
    while i + 1 < len(parts):
        label = parts[i].strip().lower()
        body = parts[i + 1].strip()
        sections[label] = body
        i += 2
    return sections


def _score_domain(domain: str, text: str) -> Optional[dict]:
    """Score all features of a domain from its description text.

    Returns None if the text indicates the domain was not observed.
    Returns a dict mapping feature_name -> score (float or None) plus
    an 'evidence' key with the first 200 characters of the description.
    """
    if re.search(r"\bnot\s+observed\b", text, re.IGNORECASE):
        return None

    feature_names = _DOMAIN_FEATURES[domain]
    result: dict = {}
    any_scored = False

    for feat in feature_names:
        score = _score_feature(text, feat)
        result[feat] = score
        if score is not None:
            any_scored = True

    if not any_scored:
        return None

    result["evidence"] = text[:200].strip()
    return result


# ------------------------------------------------------------------------------
# CDC mapping helpers
# ------------------------------------------------------------------------------


def _score_to_age(score: float, curve: dict) -> float:
    """Inverse-interpolate a feature score to an estimated age in months."""
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
    else:  # decreasing (e.g. caregiver_dependency)
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

    Averages inverse-interpolated ages across all observed features
    that have a CDC anchor curve defined.
    """
    curves = CDC_ANCHORS.get(domain, {})
    ages = []
    for feature, score in features.items():
        if feature in ("observed", "evidence") or score is None:
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
    """Soft S0-S3 stage distribution from a continuous age estimate."""
    from math import erf, sqrt

    def _integral(a, b, mu, sig):
        z = lambda x: (x - mu) / (sig * sqrt(2))
        return 0.5 * (erf(z(b)) - erf(z(a)))

    raw = {s: _integral(lo, hi, age_months, sigma) for s, (lo, hi) in STAGE_BOUNDS.items()}
    total = sum(raw.values()) or 1.0
    return {s: round(v / total, 4) for s, v in raw.items()}


# ------------------------------------------------------------------------------
# Video helpers
# ------------------------------------------------------------------------------


def _transcode_to_h264(video_path: str, tmp_path: str) -> str:
    """Re-encode a video to H.264 MP4 using OpenCV (HEVC compatibility fix)."""
    import cv2
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
    logger.info(f"  Re-encoded {Path(video_path).name} -> {Path(tmp_path).name} "
                f"({out_w}x{out_h}, rotation={rotation} deg)")
    return tmp_path


def _run_plm_text(video_path: str, prompt: str, model, tokenizer, config,
                  num_frames: int = 8, temperature: float = 0.0,
                  max_gen_len: int = 256) -> str:
    """Run PLM with a prompt and return the raw text output."""
    result = generate_description(
        video_path=video_path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        num_frames=num_frames,
        temperature=temperature,
        max_gen_len=max_gen_len,
    )
    return result["description"]


# ------------------------------------------------------------------------------
# Full assessment
# ------------------------------------------------------------------------------


def assess(video_path: str, model, tokenizer, config,
           num_frames: int = 8, temperature: float = 0.0,
           max_gen_len: int = 512, debug: bool = False) -> dict:
    """Run the full two-call developmental assessment pipeline.

    Call 1: child detection (yes/no, tiny budget).
    Call 2: free-text behavioral description per domain (scored in Python).

    Returns:
        Dict with keys: video_path, child_present, plm_features,
        domain_ages, overall_age_months, stage_distribution,
        raw_plm_output (the description text from Call 2).
    """
    import os
    import tempfile

    tmp_path: Optional[str] = None
    plm_path = video_path

    try:
        # ---- Call 1: child detection ----------------------------------------
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
                "video_path": video_path,
                "child_present": False,
                "plm_features": {},
                "domain_ages": {},
                "overall_age_months": None,
                "stage_distribution": None,
                "raw_plm_output": child_text,
            }

        # ---- Call 2: behavioral description ----------------------------------
        description = _run_plm_text(
            plm_path, _PROMPT_DESCRIBE, model, tokenizer, config,
            num_frames=num_frames, temperature=temperature, max_gen_len=max_gen_len,
        )
        logger.info(f"Description output:\n{description}")

        # ---- Python keyword scoring ------------------------------------------
        domain_sections = _extract_domain_sections(description)

        if debug:
            print("\n--- Extracted domain sections ---")
            for d in DOMAINS:
                sec = domain_sections.get(d, "(missing)")
                print(f"  [{d}] {sec!r}")

        plm_features: dict = {}
        domain_ages: dict = {}

        for domain in DOMAINS:
            text = domain_sections.get(domain, "")
            scored = _score_domain(domain, text) if text else None

            if debug:
                print(f"\n--- Keyword scores: {domain} ---")
                if not text:
                    print("  (no section text)")
                elif re.search(r"\bnot\s+observed\b", text, re.IGNORECASE):
                    print(f"  text='{text}' -> marked NOT OBSERVED")
                else:
                    for feat in _DOMAIN_FEATURES[domain]:
                        s = _score_feature(text, feat)
                        print(f"  {feat}: {s}")

            plm_features[domain] = scored
            if scored:
                features_for_age = {k: v for k, v in scored.items() if k != "evidence"}
                domain_ages[domain] = domain_age(domain, features_for_age)
            else:
                domain_ages[domain] = None

        observed_ages = [a for a in domain_ages.values() if a is not None]
        overall_age = sum(observed_ages) / len(observed_ages) if observed_ages else None
        stage_dist = stage_distribution(overall_age) if overall_age is not None else None

        return {
            "video_path": video_path,
            "child_present": True,
            "plm_features": plm_features,
            "domain_ages": domain_ages,
            "overall_age_months": round(overall_age, 1) if overall_age is not None else None,
            "stage_distribution": stage_dist,
            "raw_plm_output": description,
        }

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ------------------------------------------------------------------------------
# Formatted report
# ------------------------------------------------------------------------------

_STAGE_LABELS = {
    "S0": "< 12 months",
    "S1": "12-18 months",
    "S2": "18-24 months",
    "S3": "24-36+ months",
}


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return chr(0x2588) * filled + chr(0x2591) * (width - filled)


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

    print(f"\n{'Domain':<14} {'Feature scores (keyword)':<30} {'Age est.'}")
    print(thin)

    for domain in DOMAINS:
        features = result["plm_features"].get(domain)
        age = result["domain_ages"].get(domain)
        age_str = f"{age:.1f} mo" if age is not None else "n/a"

        if features is None:
            print(f"  {domain:<12} not observed{'':<22} {age_str}")
            continue

        feature_names = _DOMAIN_FEATURES[domain]
        lines = []
        for fname in feature_names:
            val = features.get(fname)
            if val is not None:
                lines.append(f"{fname}: {val:.2f} {_bar(val, 10)}")
            else:
                lines.append(f"{fname}: --")

        print(f"  {domain:<12} {lines[0]:<32} {age_str}")
        for line in lines[1:]:
            print(f"  {'':<12} {line}")

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
        description="Estimate a child's developmental stage from a video using PLM + CDC milestones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video data/202503_a/zdgaa.MOV
  %(prog)s --video clip.mp4 --num_frames 16
  %(prog)s --video clip.mp4 --json_only > result.json
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the child's video.")
    parser.add_argument("--ckpt", type=str, default="facebook/Perception-LM-3B",
                        help="PLM checkpoint or HuggingFace ID.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample per PLM call (default: 8).")
    parser.add_argument("--max_gen_len", type=int, default=512,
                        help="Max tokens for description call (default: 512).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 = greedy (default).")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extracted domain sections and keyword scores.")

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
        debug=args.debug,
    )

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
        print("--- Raw PLM description ---")
        print(result["raw_plm_output"])

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
