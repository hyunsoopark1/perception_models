# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator for children's videos.

Pipeline:
  1. PLM Call 1: Detect whether a child is visible (yes / no).
  2. PLM Calls 2+: For each domain (and each chunk when chunking is enabled),
     call PLM with a focused prompt listing developmental levels in order.
     PLM selects ONE level per feature; no free-form generation.
     Each (domain, chunk) pair is queried num_runs times; majority vote wins.
  3. Aggregate across chunks: pick the most advanced level per feature.
  4. Inverse-interpolation of CDC milestone curves yields age estimates.
  5. A soft S0-S3 stage distribution is computed and a formatted report is printed.

CDC anchor reference:
  S0 -> < 12 months   (pre-walker, gestures beginning)
  S1 -> 12-18 months  (independent walking, single words)
  S2 -> 18-24 months  (running, 2-word phrases, parallel play)
  S3 -> 24-36 months  (complex motor, sentences, cooperative play)

Usage:
    python estimate_development.py --video data/202503_a/zdgaa.MOV
    python estimate_development.py --video clip.mp4 --num_frames 16 --num_runs 3
    python estimate_development.py --video clip.mp4 --chunk_duration 10 --num_runs 3
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

# Per-domain prompts are built at runtime from _FEATURE_LEVELS (see below).

# ------------------------------------------------------------------------------
# Developmental level vocabulary
#
# Each feature has an ORDERED list of (cdc_score, phrase) pairs.
# Scores align EXACTLY with CDC_ANCHORS at the key milestone ages
# (0, 12, 18, 24, 36 months) so that _score_to_age() returns the right age.
# Intermediate scores are interpolated values for sub-milestone ages.
#
# Phrases are behavioral and visually observable — no abstract labels.
# PLM selects one phrase; direct dict lookup replaces all keyword matching.
#
# caregiver_dependency is a DECREASING curve (more dependent = higher score
# = younger).  Its list is ordered youngest→oldest for the prompt.
#
# CDC_ANCHORS reference (scores at key ages):
#   locomotion:          {0: 0.00, 12: 0.35, 18: 0.60, 24: 0.80, 36: 1.00}
#   coordination:        {0: 0.00, 12: 0.30, 18: 0.50, 24: 0.70, 36: 0.95}
#   stability:           {0: 0.00, 12: 0.40, 18: 0.65, 24: 0.85, 36: 1.00}
#   independence:        {0: 0.00, 12: 0.10, 18: 0.30, 24: 0.55, 36: 0.85}
#   initiative:          {0: 0.00, 12: 0.15, 18: 0.35, 24: 0.60, 36: 0.90}
#   duration:            {0: 0.00, 12: 0.20, 18: 0.40, 24: 0.65, 36: 0.90}
#   goal_directed:       {0: 0.00, 12: 0.25, 18: 0.45, 24: 0.70, 36: 0.95}
#   social_engagement:   {0: 0.00, 12: 0.50, 18: 0.65, 24: 0.75, 36: 0.90}
#   caregiver_dependency:{0: 1.00, 12: 0.80, 18: 0.65, 24: 0.45, 36: 0.20}
#   verbal:              {0: 0.00, 12: 0.20, 18: 0.40, 24: 0.70, 36: 0.95}
#   gesture:             {0: 0.00, 12: 0.50, 18: 0.75, 24: 0.85, 36: 0.90}
# ------------------------------------------------------------------------------

_FEATURE_LEVELS: dict = {
    # locomotion: 0.00=0mo  0.20=~8mo  0.35=12mo  0.60=18mo  0.80=24mo  1.00=36mo
    "locomotion": [
        (0.00, "lying or rolling, no self-propelled movement"),
        (0.20, "crawling on hands and knees"),
        (0.35, "taking first steps, unsteady, arms out, frequent falls"),
        (0.60, "walking well and beginning to run"),
        (0.80, "running and going up and down stairs"),
        (1.00, "jumping with both feet off the ground"),
    ],
    # coordination: 0.00=0mo  0.15=~6mo  0.30=12mo  0.50=18mo  0.70=24mo  0.95=36mo
    "coordination": [
        (0.00, "reflexive grasp only, no voluntary reaching"),
        (0.15, "reaches for and holds toys with whole hand"),
        (0.30, "picks up small objects with thumb and index finger"),
        (0.50, "stacks 2 to 4 blocks or uses spoon messily"),
        (0.70, "stacks 6 or more blocks and draws lines"),
        (0.95, "draws a circle and uses a fork"),
    ],
    # stability: 0.00=0mo  0.20=~6mo  0.40=12mo  0.65=18mo  0.85=24mo  1.00=36mo
    "stability": [
        (0.00, "needs full body support, no head or trunk control"),
        (0.20, "sits with support but cannot sit alone"),
        (0.40, "pulls to stand and stands briefly without support"),
        (0.65, "walks steadily without falling"),
        (0.85, "runs without falling and climbs stairs"),
        (1.00, "balances on one foot for 2 or more seconds"),
    ],
    # independence: 0.00=0mo  0.10=12mo  0.30=18mo  0.55=24mo  0.85=36mo
    "independence": [
        (0.00, "fully dependent, caregiver handles all feeding and care"),
        (0.10, "feeds self with finger foods"),
        (0.30, "attempts spoon and open cup, spills often"),
        (0.55, "removes simple clothing and washes hands with prompting"),
        (0.85, "dresses self and uses toilet with minimal help"),
    ],
    # initiative: 0.00=0mo  0.15=12mo  0.35=18mo  0.60=24mo  0.90=36mo
    "initiative": [
        (0.00, "passive, no self-directed activity"),
        (0.15, "explores objects by mouthing, banging, or shaking"),
        (0.35, "retrieves hidden toy and starts simple repetitive play"),
        (0.60, "chooses a toy and begins play independently"),
        (0.90, "creates elaborate pretend play with sequences"),
    ],
    # duration: 0.00=0mo  0.20=12mo  0.40=18mo  0.65=24mo  0.90=36mo
    "duration": [
        (0.00, "disengages in under 10 seconds"),
        (0.20, "attends for 30 to 60 seconds before moving on"),
        (0.40, "plays with a single toy for 1 to 2 minutes"),
        (0.65, "sustains one activity for 3 to 5 minutes"),
        (0.90, "stays with one activity for 5 to 10 or more minutes"),
    ],
    # goal_directed: 0.00=0mo  0.25=12mo  0.45=18mo  0.70=24mo  0.95=36mo
    "goal_directed": [
        (0.00, "random exploration with no visible goal"),
        (0.25, "persists toward a specific toy despite obstacles"),
        (0.45, "completes simple cause and effect tasks"),
        (0.70, "solves simple problems to obtain a toy"),
        (0.95, "plans and carries out multi-step sequences"),
    ],
    # social_engagement: 0.00=0mo  0.35=~10mo  0.50=12mo  0.65=18mo  0.75=24mo  0.90=36mo
    "social_engagement": [
        (0.00, "no visible response to people or voices"),
        (0.35, "responds to own name and smiles at familiar faces"),
        (0.50, "waves bye-bye and offers or shows toys to an adult"),
        (0.65, "plays alongside other children without direct interaction"),
        (0.75, "plays simple interactive games with others"),
        (0.90, "takes turns and cooperates in group play with peers"),
    ],
    # caregiver_dependency: ordered most dependent (youngest) to least (oldest)
    # 1.00=0mo  0.80=12mo  0.65=18mo  0.45=24mo  0.20=36mo
    "caregiver_dependency": [
        (1.00, "distressed immediately when caregiver is out of sight"),
        (0.80, "checks back frequently and returns to caregiver when unsure"),
        (0.65, "seeks caregiver in new situations but accepts brief separation"),
        (0.45, "plays near caregiver but ventures away independently"),
        (0.20, "plays independently for long periods without checking back"),
    ],
    # verbal: 0.00=0mo  0.10=~6mo  0.20=12mo  0.40=18mo  0.70=24mo  0.95=36mo
    "verbal": [
        (0.00, "crying or cooing only, no babbling"),
        (0.10, "babbling with repeated syllables"),
        (0.20, "says 1 to 3 real words such as mama, dada, or no"),
        (0.40, "uses 10 to 50 single words"),
        (0.70, "two-word combinations such as more milk or daddy go"),
        (0.95, "speaks in sentences of 3 or more words and asks questions"),
    ],
    # gesture: 0.00=0mo  0.35=~10mo  0.50=12mo  0.75=18mo  0.85=24mo  0.90=36mo
    "gesture": [
        (0.00, "no intentional gestures"),
        (0.35, "reaches toward objects or people and waves arms"),
        (0.50, "waves bye-bye, claps, or shakes head for no"),
        (0.75, "points to request or show something to an adult"),
        (0.85, "combines pointing or gestures with spoken words"),
        (0.90, "uses varied gestures fluently together with speech"),
    ],
}

# Features whose CDC score DECREASES with age.
# When aggregating across chunks we take MIN (most independent = most advanced).
_DECREASING_FEATURES = {"caregiver_dependency"}

_DOMAIN_DESCRIPTIONS = {
    "motor":       "movement and physical skill",
    "autonomy":    "self-care and independence",
    "attention":   "focus and play engagement",
    "interaction": "social behaviour",
    "language":    "communication",
}


def _feat_label(feat: str) -> str:
    """Convert a feature key to its capitalised display label (used in prompts and parsing)."""
    return feat.replace("_", " ").capitalize()


def _build_domain_prompt(domain: str) -> str:
    """Build a focused PLM prompt for one domain using ordered level vocabulary.

    Options are listed youngest (◄) to most advanced (►).
    PLM must output one option per feature in the exact answer block format.
    """
    features = _DOMAIN_FEATURES[domain]
    desc = _DOMAIN_DESCRIPTIONS[domain]

    level_lines = []
    for feat in features:
        levels = _FEATURE_LEVELS[feat]
        if feat in _DECREASING_FEATURES:
            ordered = levels        # already youngest→oldest
        else:
            ordered = sorted(levels, key=lambda x: x[0])
        option_str = " < ".join(phrase for _, phrase in ordered)
        level_lines.append(f"  {_feat_label(feat)}: {option_str}")

    answer_block = "\n".join(f"{_feat_label(feat)}: <option>" for feat in features)

    return (
        f"Watch this child. For each {desc} skill, select the ONE option that "
        f"best matches what you observe. Options run from youngest (left, ◄) to "
        f"most developed (right, ►). Pick the rightmost option the child "
        f"clearly demonstrates.\n\n"
        + "\n".join(level_lines)
        + f"\n\nReply in this exact format:\n{answer_block}"
    )


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
# Output parser and aggregation helpers
# ------------------------------------------------------------------------------


def _parse_domain_output(domain: str, text: str) -> dict:
    """Extract selected levels from a single PLM domain response.

    Scans for lines of the form  "Feature label: selected phrase"
    and performs an exact lookup against _FEATURE_LEVELS.
    No fuzzy or semantic matching — the PLM is constrained to the vocabulary.

    Returns {feature: (score, phrase)}.  Missing features are absent.
    """
    features = _DOMAIN_FEATURES[domain]
    result: dict = {}

    for feat in features:
        label = _feat_label(feat)
        pattern = re.compile(
            rf"^\s*{re.escape(label)}\s*:\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            continue
        answer = m.group(1).strip().rstrip(".").lower()
        for score, phrase in _FEATURE_LEVELS[feat]:
            if answer == phrase.lower():
                result[feat] = (score, phrase)
                break

    return result


def _majority_vote_features(runs: list) -> dict:
    """Majority vote across multiple PLM runs per feature.

    Args:
        runs: list of {feat: (score, phrase)} dicts, one per PLM run.

    Returns:
        {feat: (score, phrase)} — most common phrase wins.
        On a tie the phrase with the highest CDC score is preferred.
    """
    from collections import Counter

    all_feats: set = {feat for r in runs for feat in r}
    result: dict = {}
    for feat in all_feats:
        candidates = [(s, p) for r in runs for f, (s, p) in r.items() if f == feat]
        if not candidates:
            continue
        phrase_counter = Counter(p for _, p in candidates)
        top_count = phrase_counter.most_common(1)[0][1]
        tied = {p for p, cnt in phrase_counter.items() if cnt == top_count}
        best = max(((s, p) for s, p in candidates if p in tied), key=lambda x: x[0])
        result[feat] = best
    return result


def _assess_domain(seg: str, domain: str, num_runs: int, model, tokenizer, config,
                   num_frames: int, temperature: float, max_gen_len: int,
                   debug: bool = False) -> dict:
    """Run PLM num_runs times for one domain on one video segment.

    Returns:
        {
            "features":    {feat: score},
            "phrases":     {feat: phrase},
            "age":         float | None,
            "raw_outputs": [str, ...],
        }
    """
    prompt = _build_domain_prompt(domain)
    runs: list = []
    raw_outputs: list = []

    for run_idx in range(num_runs):
        txt = _run_plm_text(
            seg, prompt, model, tokenizer, config,
            num_frames=num_frames, temperature=temperature,
            max_gen_len=max_gen_len,
        )
        raw_outputs.append(txt)
        parsed = _parse_domain_output(domain, txt)
        if parsed:
            runs.append(parsed)
        if debug:
            logger.info(f"    [{domain}] run {run_idx + 1}: {txt!r}")
            logger.info(f"    parsed: {parsed}")

    if not runs:
        return {"features": {}, "phrases": {}, "age": None, "raw_outputs": raw_outputs}

    voted = _majority_vote_features(runs)
    features = {feat: score for feat, (score, _) in voted.items()}
    phrases  = {feat: phrase for feat, (_, phrase) in voted.items()}
    return {
        "features":    features,
        "phrases":     phrases,
        "age":         domain_age(domain, features),
        "raw_outputs": raw_outputs,
    }


def _aggregate_domain_scores(per_chunk: list, domain: str) -> dict:
    """Aggregate per-chunk domain results: most developmentally advanced per feature.

    For normal (increasing) features: max score across chunks.
    For caregiver_dependency (decreasing): min score (= most independent = oldest).

    Returns same shape as _assess_domain() but without raw_outputs.
    """
    best_features: dict = {}
    best_phrases:  dict = {}

    for chunk_data in per_chunk:
        for feat, score in chunk_data.get("features", {}).items():
            is_better = (
                feat not in best_features
                or (feat in _DECREASING_FEATURES and score < best_features[feat])
                or (feat not in _DECREASING_FEATURES and score > best_features[feat])
            )
            if is_better:
                best_features[feat] = score
                best_phrases[feat]  = chunk_data["phrases"].get(feat, "")

    age = domain_age(domain, best_features) if best_features else None
    return {"features": best_features, "phrases": best_phrases, "age": age}


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
# Video chunking helper
# ------------------------------------------------------------------------------


def _split_into_chunks(video_path: str, chunk_duration: float) -> list:
    """Split a video into fixed-duration temp files using OpenCV.

    Returns a list of temp file paths (caller must delete them).
    Chunks are already rotation-corrected (same logic as split_videos.py).
    """
    import cv2
    import tempfile
    from split_videos import _apply_rotation, _get_rotation

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = _get_rotation(cap)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    frames_per_chunk = int(fps * chunk_duration)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    chunk_paths = []
    writer = None
    tmp_file = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        chunk_idx = frame_idx // frames_per_chunk
        local_idx = frame_idx % frames_per_chunk

        if local_idx == 0:
            # Close previous chunk
            if writer is not None:
                writer.release()
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_file.close()
            chunk_paths.append(tmp_file.name)
            writer = cv2.VideoWriter(tmp_file.name, fourcc, fps, (out_w, out_h))

        writer.write(_apply_rotation(frame, rotation))
        frame_idx += 1

    if writer is not None:
        writer.release()
    cap.release()

    logger.info(f"Split into {len(chunk_paths)} chunks "
                f"({chunk_duration:.0f}s each, {frame_idx} frames total)")
    return chunk_paths


# ------------------------------------------------------------------------------
# Full assessment
# ------------------------------------------------------------------------------


def assess(video_path: str, model, tokenizer, config,
           num_frames: int = 8, temperature: float = 0.0,
           max_gen_len: int = 128, chunk_duration: Optional[float] = None,
           num_runs: int = 1, debug: bool = False) -> dict:
    """Run the full developmental assessment pipeline.

    Call 1: child detection on the full video (yes/no).
    Calls 2+: for each domain (and each chunk), ask PLM to select ONE
              developmental level from an ordered vocabulary. Repeated
              num_runs times per (domain, chunk) pair; majority vote wins.
    Aggregate across chunks: most advanced level per feature.

    Args:
        num_runs:       PLM calls per (domain, chunk). 1=fast, 3+=reliable.
        chunk_duration: Split video into N-second chunks; None=full video.
    """
    import os
    import tempfile

    tmp_path: Optional[str] = None   # H.264 transcode of original (if needed)
    chunk_paths: list = []           # temp chunk files (deleted in finally)
    plm_path = video_path

    try:
        # ---- HEVC → H.264 transcode if needed (detected on first PLM call) ---
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

        # ---- Determine which video segments to describe ----------------------
        if chunk_duration:
            chunk_paths = _split_into_chunks(plm_path, chunk_duration)
            segments = chunk_paths
        else:
            segments = [plm_path]

        # ---- Assess each segment per domain ---------------------------------
        n_chunks = len(segments)
        total_plm = n_chunks * len(DOMAINS) * num_runs
        logger.info(f"Assessment: {n_chunks} chunk(s) × {len(DOMAINS)} domains "
                    f"× {num_runs} run(s) = {total_plm} PLM calls")

        chunk_details: list = []
        for i, seg in enumerate(segments):
            t_start = i * chunk_duration if chunk_duration else 0
            t_end   = (i + 1) * chunk_duration if chunk_duration else None
            logger.info(f"Segment {i + 1}/{n_chunks}: {seg}")

            domain_scores: dict = {}
            for domain in DOMAINS:
                ds = _assess_domain(
                    seg, domain, num_runs, model, tokenizer, config,
                    num_frames=num_frames, temperature=temperature,
                    max_gen_len=max_gen_len, debug=debug,
                )
                domain_scores[domain] = ds
                if debug:
                    age_d = ds["age"]
                    print(f"  [{domain}] age={'%.1f' % age_d if age_d is not None else 'n/a'}"
                          f"  phrases={ds['phrases']}")

            # Compact text for overlay bottom panel
            raw_text = "\n".join(
                (f"{d}: " + ", ".join(f"{f}={p}"
                                      for f, p in domain_scores[d]["phrases"].items()))
                if domain_scores[d]["phrases"] else f"{d}: none"
                for d in DOMAINS
            )
            chunk_details.append({
                "index":         i + 1,
                "t_start":       t_start,
                "t_end":         t_end,
                "domain_scores": domain_scores,
                "domain_ages":   {d: domain_scores[d]["age"] for d in DOMAINS},
                "raw_text":      raw_text,
            })

        # ---- Aggregate: most advanced level per feature across chunks --------
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


def _print_chunk_timeline(chunk_details: list) -> None:
    """Print raw PLM output and parsed description for every domain in each chunk."""
    if not chunk_details:
        return

    n    = len(chunk_details)
    sep  = "=" * 66
    thin = "-" * 66

    print(f"\n  Per-chunk PLM output ({n} chunk{'s' if n > 1 else ''})")

    for c in chunk_details:
        t_s = int(c["t_start"])
        t_e = int(c["t_end"]) if c["t_end"] is not None else "?"
        time_str = f"{t_s}s – {t_e}s" if c["t_end"] is not None else "full video"

        print(f"\n  {sep}")
        print(f"  Chunk {c['index']}  [{time_str}]")
        print(f"  {sep}")

        domain_scores = c.get("domain_scores", {})
        domain_ages   = c.get("domain_ages",   {})

        for domain in DOMAINS:
            ds      = domain_scores.get(domain, {})
            phrases = ds.get("phrases",  {})
            age_d   = domain_ages.get(domain)
            age_tag = f"  [{age_d:.0f}mo]" if age_d is not None else "  [n/a]"

            print(f"\n  [{domain.upper()}]{age_tag}")
            print(f"  {thin}")

            # Raw PLM text (all runs when num_runs > 1)
            raw_list = ds.get("raw_outputs", [])
            if raw_list:
                for run_i, raw in enumerate(raw_list):
                    if len(raw_list) > 1:
                        print(f"  PLM output (run {run_i + 1}):")
                    else:
                        print("  PLM output:")
                    for line in raw.strip().splitlines():
                        print(f"    {line}")
            else:
                print("  PLM output:  (none)")

            # Parsed level per feature
            print("  Parsed description:")
            if phrases:
                for feat, phrase in phrases.items():
                    score = ds.get("features", {}).get(feat)
                    score_str = f"  (score {score:.2f})" if score is not None else ""
                    print(f"    {_feat_label(feat)}: {phrase}{score_str}")
            else:
                print("    (no levels matched)")

        print()


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

    # Per-chunk breakdown (only when chunking was used)
    chunks = result.get("chunk_details", [])
    if len(chunks) > 1:
        _print_chunk_timeline(chunks)

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
        matched_kw = features.get("matched_keywords", {})
        lines = []
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
# Video overlay renderer
# ------------------------------------------------------------------------------

_DOMAIN_ABBR = {
    "motor": "Motor", "autonomy": "Auto",
    "attention": "Attn", "interaction": "Inter", "language": "Lang",
}

# Colors (BGR)
_COL_HEADER  = (0, 220, 255)   # yellow-ish
_COL_TEXT    = (255, 255, 255) # white
_COL_NONE    = (100, 100, 100) # grey
_COL_MATCHED = (80, 255, 80)   # green  (keyword that drove the final score)


def render_assessment_video(video_path: str, result: dict, output_path: str) -> None:
    """Burn chunk keyword overlays into the video and write to output_path.

    For each frame the overlay shows:
      - Chunk index / time range  (top-left header)
      - Per-domain selected keywords + per-domain age estimate
      - Overall estimated age (bottom of panel)
      - Thin progress bar (bottom edge) with chunk boundary ticks
      - Domain lines are green when the domain drove the aggregated score

    Args:
        video_path: Source video (original — OpenCV reads HEVC fine).
        result:     Dict returned by assess(), must contain chunk_details.
        output_path: Where to write the annotated MP4.
    """
    import cv2
    from split_videos import _apply_rotation, _get_rotation

    # Local helper — needs cv2 in scope
    def _puttext(img, text, pos, scale, color, thickness=1):
        x, y = pos
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    chunk_details = result.get("chunk_details", [])
    if not chunk_details:
        logger.warning("No chunk_details in result; nothing to render.")
        return

    # Build feature -> matched phrase lookup (from aggregated scoring)
    matched_kw: dict = {}
    for domain in DOMAINS:
        feat_data = result.get("plm_features", {}).get(domain) or {}
        for feat, phrase in feat_data.get("matched_keywords", {}).items():
            if phrase:
                matched_kw[feat] = phrase

    domain_ages = result.get("domain_ages", {})
    overall_age = result.get("overall_age_months")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation     = _get_rotation(cap)
    raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    fscale   = max(0.65, min(1.10, out_w / 1440 * 1.10))   # larger for readability
    fscale_s = fscale * 0.80          # smaller font for raw-text panel
    line_h   = int(fscale * 42)
    line_h_s = int(fscale_s * 40)
    pad      = 10
    n_chunks = len(chunk_details)

    # Pre-compute wrapped raw text per chunk (constant across frames)
    # Estimate chars per line from frame width and small font scale
    chars_per_line = max(40, int(out_w / (fscale_s * 12.5)))

    def _wrap(text, width):
        """Wrap text to lines of at most *width* characters."""
        import textwrap
        lines = []
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            lines.extend(textwrap.wrap(raw_line, width) or [raw_line])
        return lines

    chunk_wrapped = {
        c["index"]: _wrap(c.get("raw_text", ""), chars_per_line)
        for c in chunk_details
    }

    def _chunk_for_time(t):
        for c in chunk_details:
            if c["t_end"] is None or t < c["t_end"]:
                return c
        return chunk_details[-1]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _apply_rotation(frame, rotation)
        t     = frame_idx / fps
        chunk = _chunk_for_time(t)

        # --- semi-transparent panel ---
        # Layout: header + (domain-label + feature-scores) * 5 + sep + overall
        n_lines  = 1 + len(DOMAINS) * 2 + 2
        panel_h  = n_lines * line_h_s + pad * 2
        panel_w  = min(out_w, int(out_w * 0.72))
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # --- chunk header ---
        t_s    = int(chunk["t_start"])
        t_e    = int(chunk["t_end"]) if chunk["t_end"] is not None else "?"
        header = f"Chunk {chunk['index']}/{n_chunks}  [{t_s}s – {t_e}s]"
        _puttext(frame, header, (pad, pad + line_h_s), fscale, _COL_HEADER)

        # --- per-domain: label line + feature breakdown line ---
        for row, domain in enumerate(DOMAINS):
            agg_age   = domain_ages.get(domain)
            chunk_age = chunk.get("domain_ages", {}).get(domain)
            chunk_ds  = chunk.get("domain_scores", {}).get(domain, {})
            chunk_phr = chunk_ds.get("phrases", {})

            if n_chunks > 1:
                ca  = f"{chunk_age:.0f}" if chunk_age is not None else "?"
                aa  = f"{agg_age:.0f}"   if agg_age   is not None else "?"
                age_str = f"  [{ca}→{aa}mo]"
            else:
                age_str = f"  [{agg_age:.0f}mo]" if agg_age is not None else ""

            label   = _DOMAIN_ABBR[domain]
            y_label = pad + (row * 2 + 2) * line_h_s
            y_feat  = y_label + line_h_s
            col     = _COL_MATCHED if chunk_phr else _COL_TEXT
            _puttext(frame, f"{label}{age_str}", (pad, y_label), fscale_s, col)

            # "locom=crawling  |  coord=grasping  |  stab=sitting"
            feat_parts = []
            for feat in _DOMAIN_FEATURES[domain]:
                phrase = chunk_phr.get(feat, "")
                score  = chunk_ds.get("features", {}).get(feat)
                if phrase and score is not None:
                    feat_parts.append(f"{feat[:5]}={phrase}")
            feat_line = "  " + "  |  ".join(feat_parts) if feat_parts else "  —"
            max_chars = int((panel_w - pad * 2) / (fscale_s * 0.75 * 12))
            if len(feat_line) > max_chars:
                feat_line = feat_line[:max_chars - 3] + "..."
            _puttext(frame, feat_line, (pad, y_feat), fscale_s * 0.75, _COL_TEXT)

        # --- overall age ---
        y_overall = pad + (len(DOMAINS) * 2 + 2) * line_h_s
        overall_str = (f"Overall: {overall_age:.1f} months"
                       if overall_age is not None else "Overall: insufficient data")
        cv2.line(frame, (pad, y_overall - line_h_s // 2),
                 (panel_w - pad, y_overall - line_h_s // 2), (80, 80, 80), 1)
        _puttext(frame, overall_str, (pad, y_overall), fscale_s, _COL_HEADER)

        # --- bottom panel: raw PLM text for this chunk ---
        raw_lines  = chunk_wrapped.get(chunk["index"], [])
        txt_panel_h = (len(raw_lines) + 1) * line_h_s + pad  # +1 for "PLM:" header
        txt_y0 = out_h - txt_panel_h - 8   # 8px gap above progress bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, txt_y0), (out_w, txt_y0 + txt_panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)

        _puttext(frame, "PLM output:", (pad, txt_y0 + line_h_s),
                 fscale_s, _COL_HEADER)
        for li, line in enumerate(raw_lines):
            y_txt = txt_y0 + (li + 2) * line_h_s
            _puttext(frame, line, (pad, y_txt), fscale_s, _COL_TEXT)

        # --- progress bar (bottom edge) ---
        if total_frames > 0:
            bar_h  = max(4, int(out_h * 0.008))
            filled = int(out_w * frame_idx / total_frames)
            for c in chunk_details:
                if c["t_end"]:
                    tx = int(out_w * c["t_end"] * fps / total_frames)
                    cv2.rectangle(frame, (tx - 1, out_h - bar_h - 4),
                                  (tx + 1, out_h - 4), (180, 180, 180), -1)
            cv2.rectangle(frame, (0, out_h - bar_h), (out_w, out_h),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (0, out_h - bar_h), (filled, out_h),
                          (0, 200, 255), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info(f"Annotated video saved to: {output_path}  ({frame_idx} frames)")


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
    parser.add_argument("--max_gen_len", type=int, default=128,
                        help="Max tokens per domain call (default: 128).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0.0 = greedy (default).")
    parser.add_argument("--chunk_duration", type=float, default=None,
                        help="Split video into chunks of this many seconds. "
                             "Each chunk is assessed per-domain independently; "
                             "the most advanced level across chunks is used. "
                             "Omit to treat the full video as one chunk.")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="PLM calls per (domain, chunk). "
                             "1=fastest; 3=majority vote for reliability (default: 1).")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Render keyword overlays onto the video and save to this path.")
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
        chunk_duration=args.chunk_duration,
        num_runs=args.num_runs,
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

    if args.output_video:
        render_assessment_video(args.video, result, args.output_video)


if __name__ == "__main__":
    main()
