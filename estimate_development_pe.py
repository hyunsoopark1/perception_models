# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Developmental stage estimator using Perception Encoder (PE).

Pipeline:
  1. PE child detection: cosine similarity vs. child / no-child text templates.
  2. For each chunk and domain: cosine similarity between video embedding and
     all _FEATURE_LEVELS phrase embeddings selects the best matching level.
     No text generation, no parsing, no hallucination.
  3. Aggregate across chunks: most advanced level per feature (max score,
     min for caregiver_dependency which decreases with age).
  4. Inverse-interpolation of CDC milestone curves yields age estimates.
  5. Soft S0-S3 stage distribution and formatted report.

PE API:
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    video_emb = model.encode_video(video_tensor, normalize=True)   # (B,N,C,H,W)->(B,D)
    text_emb  = model.encode_text(tokens, normalize=True)          # (B,L)->(B,D)
    sim = (video_emb @ text_emb.T)                                 # cosine sim

Usage:
    python estimate_development_pe.py --video clip.mp4
    python estimate_development_pe.py --video clip.mp4 --chunk_duration 10
    python estimate_development_pe.py --video clip.mp4 --sim_threshold 0.20
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import PIL.Image

from core.vision_encoder import pe, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Developmental level vocabulary (identical to estimate_development.py)
# ------------------------------------------------------------------------------

_FEATURE_LEVELS: dict = {
    "locomotion": [
        (0.00, "lying or rolling, no self-propelled movement"),
        (0.20, "crawling on hands and knees"),
        (0.35, "taking first steps, unsteady, arms out, frequent falls"),
        (0.60, "walking well and beginning to run"),
        (0.80, "running and going up and down stairs"),
        (1.00, "jumping with both feet off the ground"),
    ],
    "coordination": [
        (0.00, "reflexive grasp only, no voluntary reaching"),
        (0.15, "reaches for and holds toys with whole hand"),
        (0.30, "picks up small objects with thumb and index finger"),
        (0.50, "stacks 2 to 4 blocks"),
        (0.70, "stacks 6 or more blocks"),
        (0.95, "draws a circle or cuts with scissors"),
    ],
    "stability": [
        (0.00, "needs full body support, no head or trunk control"),
        (0.20, "sits with support but cannot sit alone"),
        (0.40, "pulls to stand and stands briefly without support"),
        (0.65, "walks steadily without falling"),
        (0.85, "runs without falling"),
        (1.00, "balances on one foot for 2 or more seconds"),
    ],
    "independence": [
        (0.00, "fully dependent, caregiver handles all feeding and care"),
        (0.10, "feeds self with finger foods"),
        (0.30, "attempts to use spoon, spills often"),
        (0.55, "removes simple clothing independently"),
        (0.85, "dresses self and uses toilet with minimal help"),
    ],
    "initiative": [
        (0.00, "passive, no self-directed activity"),
        (0.15, "explores objects by mouthing, banging, or shaking"),
        (0.35, "retrieves a hidden toy and starts simple repetitive play"),
        (0.60, "chooses a toy and begins play independently"),
        (0.90, "creates elaborate pretend play with sequences"),
    ],
    "duration": [
        (0.00, "no focused attention, immediately looks away from objects"),
        (0.20, "briefly attends to an object then quickly shifts attention"),
        (0.40, "engaged with a single object or activity"),
        (0.65, "deeply absorbed in play, not distracted by surroundings"),
        (0.90, "fully immersed in complex activity with intense concentration"),
    ],
    "goal_directed": [
        (0.00, "random exploration with no visible goal"),
        (0.25, "persists toward a specific toy despite obstacles"),
        (0.45, "completes a simple cause and effect task"),
        (0.70, "solves a simple problem to obtain a toy"),
        (0.95, "executes a visible multi-step plan toward a goal"),
    ],
    "social_engagement": [
        (0.00, "no visible response to people or voices"),
        (0.35, "responds to own name and smiles at familiar faces"),
        (0.50, "waves bye-bye or shows toys to an adult"),
        (0.65, "plays alongside other children without direct interaction"),
        (0.75, "plays simple interactive games with others"),
        (0.90, "takes turns and cooperates in group play with peers"),
    ],
    "caregiver_dependency": [
        (1.00, "distressed immediately when caregiver is out of sight"),
        (0.80, "checks back frequently and returns to caregiver when unsure"),
        (0.65, "seeks caregiver in new situations but accepts brief separation"),
        (0.45, "plays near caregiver but ventures away independently"),
        (0.20, "plays without checking back on the caregiver"),
    ],
    "verbal": [
        (0.00, "crying or cooing only, no babbling"),
        (0.10, "babbling with repeated syllables"),
        (0.20, "says 1 to 3 real words"),
        (0.40, "uses 10 to 50 different single words"),
        (0.70, "combines two words together"),
        (0.95, "speaks in sentences of 3 or more words"),
    ],
    "gesture": [
        (0.00, "no intentional gestures"),
        (0.35, "reaches toward objects or people"),
        (0.50, "waves bye-bye or claps hands"),
        (0.75, "points to request or show something to an adult"),
        (0.85, "combines pointing or gestures with spoken words"),
        (0.90, "uses varied gestures fluently together with speech"),
    ],
}

_DECREASING_FEATURES = {"caregiver_dependency"}

_DOMAIN_DESCRIPTIONS = {
    "motor":       "movement and physical skill",
    "autonomy":    "self-care and independence",
    "attention":   "focus and play engagement",
    "interaction": "social behaviour",
    "language":    "communication",
}

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
# CDC mapping helpers (identical to estimate_development.py)
# ------------------------------------------------------------------------------


def _feat_label(feat: str) -> str:
    return feat.replace("_", " ").capitalize()


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
    else:
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
    """Estimate developmental age (months) for one domain."""
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
# Video chunking helper (identical to estimate_development.py)
# ------------------------------------------------------------------------------


def _split_into_chunks(video_path: str, chunk_duration: float) -> list:
    """Split a video into fixed-duration temp files using OpenCV."""
    import cv2
    import tempfile
    from split_videos import _apply_rotation, _get_rotation

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

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
# PE model loading and encoding
# ------------------------------------------------------------------------------


def load_pe_model(name: str = "PE-Core-L14-336", device: str = "cuda") -> tuple:
    """Load PE model, image preprocessor, and text tokenizer.

    Returns:
        (model, preprocess, tokenizer)
    """
    model = pe.CLIP.from_config(name, pretrained=True).to(device).eval()
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    logger.info(f"Loaded PE model: {name}  device={device}")
    return model, preprocess, tokenizer


def _encode_all_levels(model, tokenizer, device: str) -> dict:
    """Pre-encode all _FEATURE_LEVELS phrases into normalized text embeddings.

    Returns:
        {feat: tensor(num_levels, D)} on the given device.
    """
    cache = {}
    with torch.no_grad():
        for feat, levels in _FEATURE_LEVELS.items():
            phrases = [phrase for _, phrase in levels]
            tokens = tokenizer(phrases).to(device)
            embs = model.encode_text(tokens, normalize=True)  # (N, D)
            cache[feat] = embs
    logger.info(f"Encoded {len(cache)} features ({sum(v.shape[0] for v in cache.values())} phrases total)")
    return cache


def _extract_frames(video_path: str, num_frames: int, preprocess) -> tuple:
    """Extract num_frames evenly-spaced frames from a video.

    Returns:
        (pil_images, video_tensor) where:
          pil_images   — list of num_frames PIL.Image (RGB)
          video_tensor — torch.Tensor (1, N, C, H, W) for encode_video()
    """
    import cv2
    from split_videos import _apply_rotation, _get_rotation

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    rotation = _get_rotation(cap)
    indices = [int(i * total / num_frames) for i in range(num_frames)]

    pil_images: list = []
    tensors: list    = []
    last_valid = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if last_valid is not None:
                pil_images.append(pil_images[-1])
                tensors.append(last_valid)
            else:
                pil_images.append(PIL.Image.new("RGB", (224, 224)))
                tensors.append(torch.zeros(3, 224, 224))
            continue
        frame = _apply_rotation(frame, rotation)
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t = preprocess(img)   # (C, H, W)
        last_valid = t
        pil_images.append(img)
        tensors.append(t)

    cap.release()
    video_tensor = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, N, C, H, W)
    return pil_images, video_tensor


def _encode_chunk_pe(video_path: str, model, preprocess, num_frames: int,
                     device: str) -> tuple:
    """Encode a video clip as a single joint embedding over all N frames.

    Returns:
        (pil_images, video_emb) where video_emb is (1, D).
    """
    pil_images, video = _extract_frames(video_path, num_frames, preprocess)
    with torch.no_grad():
        emb = model.encode_video(video.to(device), normalize=True)  # (1, D)
    return pil_images, emb


def _detect_child_pe(video_emb: torch.Tensor, model, tokenizer, device: str) -> bool:
    """Return True if a child is visible in the video via cosine similarity."""
    templates = [
        "a young child or toddler is clearly visible and active",
        "no child present, empty room or adult only",
    ]
    with torch.no_grad():
        tokens = tokenizer(templates).to(device)
        text_embs = model.encode_text(tokens, normalize=True)  # (2, D)
        sims = (video_emb @ text_embs.T).squeeze(0)            # (2,)
    child_sim    = sims[0].item()
    no_child_sim = sims[1].item()
    logger.info(f"Child detection — child_sim={child_sim:.3f}  no_child_sim={no_child_sim:.3f}")
    return child_sim > no_child_sim


# ------------------------------------------------------------------------------
# PE-based domain scoring
# ------------------------------------------------------------------------------


def _score_domain_pe(
    video_emb: torch.Tensor,
    domain: str,
    level_cache: dict,
    device: str,
    sim_threshold: float = 0.15,
) -> dict:
    """Score one domain for one chunk using cosine similarity.

    For each feature the level with the highest cosine similarity to the video
    is selected.  If the maximum similarity falls below sim_threshold the
    feature is treated as not confidently visible and omitted.

    Returns:
        {
            "features":     {feat: cdc_score},
            "phrases":      {feat: phrase},
            "similarities": {feat: best_sim},
            "age":          float | None,
        }
    """
    features_out: dict = {}
    phrases_out:  dict = {}
    sims_out:     dict = {}

    feats = _DOMAIN_FEATURES[domain]
    with torch.no_grad():
        for feat in feats:
            levels    = _FEATURE_LEVELS[feat]
            text_embs = level_cache[feat]              # (N, D) on device
            sims      = (video_emb @ text_embs.T).squeeze(0)  # (N,)
            best_idx  = int(sims.argmax().item())
            best_sim  = float(sims[best_idx].item())

            if best_sim < sim_threshold:
                continue

            score, phrase = levels[best_idx]
            features_out[feat] = score
            phrases_out[feat]  = phrase
            sims_out[feat]     = best_sim

    age = domain_age(domain, features_out) if features_out else None
    return {
        "features":     features_out,
        "phrases":      phrases_out,
        "similarities": sims_out,
        "age":          age,
    }


def _aggregate_domain_scores(per_chunk: list, domain: str) -> dict:
    """Aggregate per-chunk results: most developmentally advanced per feature.

    Normal (increasing) features: max score across chunks.
    caregiver_dependency (decreasing): min score (most independent = oldest).
    """
    best_features: dict = {}
    best_phrases:  dict = {}
    best_sims:     dict = {}

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
                best_sims[feat]     = chunk_data.get("similarities", {}).get(feat, 0.0)

    age = domain_age(domain, best_features) if best_features else None
    return {
        "features":     best_features,
        "phrases":      best_phrases,
        "similarities": best_sims,
        "age":          age,
    }

# ------------------------------------------------------------------------------
# Full assessment
# ------------------------------------------------------------------------------


def save_frame_visualization(
    pil_frames: list,
    chunk_info: dict,
    output_path: str,
    thumb_size: int = 160,
) -> None:
    """Save a matplotlib figure showing the N sampled frames for a chunk.

    Layout:
      Row 0   — N frame thumbnails with frame index labels
      Rows 1+ — one row per domain: domain label | per-frame best phrase + sim score
                cells are colour-coded by similarity (white→green scale)

    Args:
        pil_frames:  List of N PIL.Image objects (RGB).
        chunk_info:  Chunk dict from assess_pe() (has domain_scores, t_start, t_end).
        output_path: Where to write the .png file.
        thumb_size:  Height in pixels for the frame thumbnails.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    n = len(pil_frames)
    n_domains = len(DOMAINS)

    # Figure: n columns, (1 + n_domains) rows
    fig_w = max(12, n * 2.2)
    fig_h = 2.0 + n_domains * 0.9
    fig, axes = plt.subplots(
        1 + n_domains, n,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [thumb_size / 72] + [0.85] * n_domains},
    )
    if n == 1:
        axes = axes.reshape(-1, 1)

    t_s = int(chunk_info["t_start"])
    t_e = chunk_info["t_end"]
    time_str = f"{t_s}s\u2013{int(t_e)}s" if t_e is not None else "full video"
    chunk_age_vals = [v for v in chunk_info["domain_ages"].values() if v is not None]
    chunk_age_str  = f"{sum(chunk_age_vals)/len(chunk_age_vals):.0f}mo" if chunk_age_vals else "?"
    fig.suptitle(
        f"Chunk {chunk_info['index']}  [{time_str}]  est. age \u2248 {chunk_age_str}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # Row 0: frame thumbnails
    total_frames_in_chunk = 1  # placeholder for time offset labelling
    for fi, img in enumerate(pil_frames):
        ax = axes[0, fi]
        ax.imshow(img)
        ax.set_title(f"frame {fi + 1}", fontsize=7, pad=2)
        ax.axis("off")

    # Rows 1+: per-domain similarity table
    domain_scores = chunk_info.get("domain_scores", {})
    for di, domain in enumerate(DOMAINS):
        ds      = domain_scores.get(domain, {})
        phrases = ds.get("phrases", {})
        sims    = ds.get("similarities", {})
        age_d   = chunk_info["domain_ages"].get(domain)
        age_tag = f"{age_d:.0f}mo" if age_d is not None else "n/a"

        for fi in range(n):
            ax = axes[di + 1, fi]
            ax.axis("off")

            # Build text from features in this domain
            lines = []
            bg_sim = 0.0
            for feat in _DOMAIN_FEATURES[domain]:
                phrase = phrases.get(feat, "")
                sim    = sims.get(feat, 0.0)
                if phrase:
                    short = phrase[:28] + "\u2026" if len(phrase) > 28 else phrase
                    lines.append(f"{short}\n({sim:.3f})")
                    bg_sim = max(bg_sim, sim)
                else:
                    lines.append("\u2014")

            # Background colour: white (0) to green (0.4+)
            intensity = min(1.0, bg_sim / 0.4)
            bg_color  = (1 - intensity * 0.55, 1.0, 1 - intensity * 0.55)  # RGB

            ax.set_facecolor(bg_color)
            ax.patch.set_visible(True)

            cell_text = "\n".join(lines)
            ax.text(
                0.5, 0.5, cell_text,
                ha="center", va="center",
                fontsize=6.5, wrap=True,
                transform=ax.transAxes,
            )

            # Domain label only in first column
            if fi == 0:
                ax.set_ylabel(
                    f"{domain}\n[{age_tag}]",
                    fontsize=7, rotation=0, labelpad=50,
                    va="center", ha="right",
                )

    plt.tight_layout(rect=[0.08, 0, 1, 1])
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------
# Full assessment
# ------------------------------------------------------------------------------


def assess_pe(
    video_path: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    num_frames: int = 8,
    chunk_duration: Optional[float] = None,
    sim_threshold: float = 0.15,
    visualize_dir: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """Run the full PE-based developmental assessment pipeline.

    1. Pre-encode all _FEATURE_LEVELS phrases.
    2. Detect child on the full video.
    3. Score each domain per chunk via cosine similarity.
    4. Aggregate: most advanced level per feature.
    5. Compute overall age and stage distribution.

    Returns a dict compatible with print_report() and render_assessment_video().
    """
    import os

    chunk_paths: list = []

    try:
        # Pre-encode all level descriptions once
        level_cache = _encode_all_levels(model, tokenizer, device)

        # Child detection on the full video
        _, full_emb = _encode_chunk_pe(video_path, model, preprocess, num_frames, device)
        child_present = _detect_child_pe(full_emb, model, tokenizer, device)
        logger.info(f"Child present: {child_present}")

        if not child_present:
            return {
                "video_path":         video_path,
                "child_present":      False,
                "pe_features":        {},
                "plm_features":       {},
                "domain_ages":        {},
                "overall_age_months": None,
                "stage_distribution": None,
            }

        # Determine segments
        if chunk_duration:
            chunk_paths = _split_into_chunks(video_path, chunk_duration)
            segments = chunk_paths
        else:
            segments = [video_path]

        n_chunks = len(segments)
        logger.info(f"Assessment: {n_chunks} chunk(s) × {len(DOMAINS)} domains")

        chunk_details: list = []
        for i, seg in enumerate(segments):
            t_start = i * chunk_duration if chunk_duration else 0
            t_end   = (i + 1) * chunk_duration if chunk_duration else None
            logger.info(f"Segment {i + 1}/{n_chunks}: {seg}")

            pil_frames, video_emb = _encode_chunk_pe(seg, model, preprocess, num_frames, device)

            domain_scores: dict = {}
            for domain in DOMAINS:
                ds = _score_domain_pe(video_emb, domain, level_cache, device, sim_threshold)
                domain_scores[domain] = ds
                if debug:
                    age_d = ds["age"]
                    print(f"  [{domain}] age={'%.1f' % age_d if age_d is not None else 'n/a'}"
                          f"  sims={ds['similarities']}")

            # Build raw_text summary for video overlay bottom panel
            raw_text = "\n".join(
                (f"{d}: " + ", ".join(
                    f"{f}={p} ({domain_scores[d]['similarities'].get(f, 0):.3f})"
                    for f, p in domain_scores[d]["phrases"].items()))
                if domain_scores[d]["phrases"] else f"{d}: none"
                for d in DOMAINS
            )

            chunk_info = {
                "index":         i + 1,
                "t_start":       t_start,
                "t_end":         t_end,
                "domain_scores": domain_scores,
                "domain_ages":   {d: domain_scores[d]["age"] for d in DOMAINS},
                "raw_text":      raw_text,
            }
            chunk_details.append(chunk_info)

            if visualize_dir:
                import os
                os.makedirs(visualize_dir, exist_ok=True)
                vis_path = os.path.join(
                    visualize_dir,
                    f"chunk_{i + 1:03d}.png",
                )
                save_frame_visualization(pil_frames, chunk_info, vis_path)
                logger.info(f"Frame visualization saved: {vis_path}")

        # Aggregate across chunks
        pe_features:  dict = {}
        domain_ages:  dict = {}

        for domain in DOMAINS:
            per_chunk = [c["domain_scores"][domain] for c in chunk_details]
            agg = _aggregate_domain_scores(per_chunk, domain)

            if agg["features"]:
                entry = dict(agg["features"])
                entry["evidence"] = ", ".join(
                    f"{f}={p} (sim {agg['similarities'].get(f, 0):.3f})"
                    for f, p in agg["phrases"].items()
                )
                entry["matched_keywords"] = dict(agg["phrases"])
                pe_features[domain] = entry
            else:
                pe_features[domain] = None
            domain_ages[domain] = agg["age"]

            if debug:
                print(f"\n--- Aggregated [{domain}] age={agg['age']} ---")
                for feat, score in agg["features"].items():
                    sim = agg["similarities"].get(feat, 0)
                    print(f"  {feat}: {score:.2f}  sim={sim:.3f}  ({agg['phrases'].get(feat, '')})")

        observed_ages = [a for a in domain_ages.values() if a is not None]
        overall_age   = sum(observed_ages) / len(observed_ages) if observed_ages else None
        stage_dist    = stage_distribution(overall_age) if overall_age is not None else None

        return {
            "video_path":         video_path,
            "child_present":      True,
            "pe_features":        pe_features,
            "plm_features":       pe_features,   # alias for compatibility with print_report
            "domain_ages":        domain_ages,
            "overall_age_months": round(overall_age, 1) if overall_age is not None else None,
            "stage_distribution": stage_dist,
            "chunk_details":      chunk_details,
        }

    finally:
        for p in chunk_paths:
            try:
                os.unlink(p)
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
    """Print PE similarities and selected phrase for every domain in each chunk."""
    if not chunk_details:
        return

    n    = len(chunk_details)
    sep  = "=" * 66
    thin = "-" * 66

    print(f"\n  Per-chunk PE output ({n} chunk{'s' if n > 1 else ''})")

    for c in chunk_details:
        t_s = int(c["t_start"])
        t_e = int(c["t_end"]) if c["t_end"] is not None else "?"
        time_str = f"{t_s}s - {t_e}s" if c["t_end"] is not None else "full video"

        print(f"\n  {sep}")
        print(f"  Chunk {c['index']}  [{time_str}]")
        print(f"  {sep}")

        domain_scores = c.get("domain_scores", {})
        domain_ages   = c.get("domain_ages",   {})

        for domain in DOMAINS:
            ds      = domain_scores.get(domain, {})
            phrases = ds.get("phrases",  {})
            sims    = ds.get("similarities", {})
            age_d   = domain_ages.get(domain)
            age_tag = f"  [{age_d:.0f}mo]" if age_d is not None else "  [n/a]"

            print(f"\n  [{domain.upper()}]{age_tag}")
            print(f"  {thin}")
            print("  PE similarities:")
            if phrases:
                for feat, phrase in phrases.items():
                    sim = sims.get(feat)
                    sim_str = f"  (sim {sim:.3f})" if sim is not None else ""
                    print(f"    {_feat_label(feat)}: {phrase}{sim_str}")
            else:
                print("    (no features above threshold)")

        print()


def print_report(result: dict) -> None:
    sep  = "=" * 62
    thin = "-" * 62

    child_present = result.get("child_present", False)
    child_str = "YES" if child_present else "NO"

    print(f"\n{sep}")
    print(f"  PE-based Developmental Assessment")
    print(f"  {Path(result['video_path']).name}")
    print(f"  Child present: {child_str}")
    print(sep)

    if not child_present:
        print("\n  No child detected in this video.\n")
        print(f"{sep}\n")
        return

    chunks = result.get("chunk_details", [])
    if len(chunks) > 1:
        _print_chunk_timeline(chunks)

    features_key = "pe_features" if "pe_features" in result else "plm_features"
    print(f"\n{'Domain':<14} {'Feature scores (PE)':<30} {'Age est.'}")
    print(thin)

    for domain in DOMAINS:
        features = result[features_key].get(domain)
        age = result["domain_ages"].get(domain)
        age_str = f"{age:.1f} mo" if age is not None else "n/a"

        if features is None:
            print(f"  {domain:<12} not observed{'':<22} {age_str}")
            continue

        feature_names = _DOMAIN_FEATURES[domain]
        matched_kw = features.get("matched_keywords", {})
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
# Video overlay renderer
# ------------------------------------------------------------------------------

_DOMAIN_ABBR = {
    "motor": "Motor", "autonomy": "Auto",
    "attention": "Attn", "interaction": "Inter", "language": "Lang",
}

_COL_HEADER  = (0, 220, 255)   # yellow-ish
_COL_TEXT    = (255, 255, 255) # white
_COL_NONE    = (100, 100, 100) # grey
_COL_MATCHED = (80, 255, 80)   # green


def render_assessment_video(video_path: str, result: dict, output_path: str) -> None:
    """Burn PE similarity overlays into the video and write to output_path."""
    import cv2
    from split_videos import _apply_rotation, _get_rotation

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

    features_key = "pe_features" if "pe_features" in result else "plm_features"
    matched_kw: dict = {}
    for domain in DOMAINS:
        feat_data = result.get(features_key, {}).get(domain) or {}
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

    fscale   = max(0.65, min(1.10, out_w / 1440 * 1.10))
    fscale_s = fscale * 0.80
    line_h   = int(fscale * 42)
    line_h_s = int(fscale_s * 40)
    pad      = 10
    n_chunks = len(chunk_details)

    chars_per_line = max(40, int(out_w / (fscale_s * 12.5)))

    def _wrap(text, width):
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

        n_lines  = 1 + len(DOMAINS) * 2 + 2
        panel_h  = n_lines * line_h_s + pad * 2
        panel_w  = min(out_w, int(out_w * 0.72))
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        t_s    = int(chunk["t_start"])
        t_e    = int(chunk["t_end"]) if chunk["t_end"] is not None else "?"
        header = f"Chunk {chunk['index']}/{n_chunks}  [{t_s}s - {t_e}s]"
        _puttext(frame, header, (pad, pad + line_h_s), fscale, _COL_HEADER)

        for row, domain in enumerate(DOMAINS):
            agg_age   = domain_ages.get(domain)
            chunk_age = chunk.get("domain_ages", {}).get(domain)
            chunk_ds  = chunk.get("domain_scores", {}).get(domain, {})
            chunk_phr = chunk_ds.get("phrases", {})
            chunk_sim = chunk_ds.get("similarities", {})

            if n_chunks > 1:
                ca  = f"{chunk_age:.0f}" if chunk_age is not None else "?"
                aa  = f"{agg_age:.0f}"   if agg_age   is not None else "?"
                age_str = f"  [{ca}->{aa}mo]"
            else:
                age_str = f"  [{agg_age:.0f}mo]" if agg_age is not None else ""

            label   = _DOMAIN_ABBR[domain]
            y_label = pad + (row * 2 + 2) * line_h_s
            y_feat  = y_label + line_h_s
            col     = _COL_MATCHED if chunk_phr else _COL_TEXT
            _puttext(frame, f"{label}{age_str}", (pad, y_label), fscale_s, col)

            feat_parts = []
            for feat in _DOMAIN_FEATURES[domain]:
                phrase = chunk_phr.get(feat, "")
                sim    = chunk_sim.get(feat)
                if phrase and sim is not None:
                    feat_parts.append(f"{feat[:5]}={phrase[:12]}({sim:.2f})")
            feat_line = "  " + "  |  ".join(feat_parts) if feat_parts else "  -"
            max_chars = int((panel_w - pad * 2) / (fscale_s * 0.75 * 12))
            if len(feat_line) > max_chars:
                feat_line = feat_line[:max_chars - 3] + "..."
            _puttext(frame, feat_line, (pad, y_feat), fscale_s * 0.75, _COL_TEXT)

        y_overall = pad + (len(DOMAINS) * 2 + 2) * line_h_s
        overall_str = (f"Overall: {overall_age:.1f} months"
                       if overall_age is not None else "Overall: insufficient data")
        cv2.line(frame, (pad, y_overall - line_h_s // 2),
                 (panel_w - pad, y_overall - line_h_s // 2), (80, 80, 80), 1)
        _puttext(frame, overall_str, (pad, y_overall), fscale_s, _COL_HEADER)

        raw_lines  = chunk_wrapped.get(chunk["index"], [])
        txt_panel_h = (len(raw_lines) + 1) * line_h_s + pad
        txt_y0 = out_h - txt_panel_h - 8
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, txt_y0), (out_w, txt_y0 + txt_panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)

        _puttext(frame, "PE output:", (pad, txt_y0 + line_h_s), fscale_s, _COL_HEADER)
        for li, line in enumerate(raw_lines):
            y_txt = txt_y0 + (li + 2) * line_h_s
            _puttext(frame, line, (pad, y_txt), fscale_s, _COL_TEXT)

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
        description="Estimate a child's developmental stage using Perception Encoder (PE).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4
  %(prog)s --video clip.mp4 --num_frames 16 --chunk_duration 10
  %(prog)s --video clip.mp4 --sim_threshold 0.20 --json_only > result.json
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the child's video.")
    parser.add_argument("--pe_model", type=str, default="PE-Core-L14-336",
                        help="PE model config name (default: PE-Core-L14-336).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu. Auto-detected if omitted.")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample per video chunk (default: 8).")
    parser.add_argument("--chunk_duration", type=float, default=None,
                        help="Split video into chunks of this many seconds. "
                             "Omit to use the full video as one chunk.")
    parser.add_argument("--sim_threshold", type=float, default=0.15,
                        help="Min cosine similarity to accept a feature match (default: 0.15).")
    parser.add_argument("--json_only", action="store_true",
                        help="Print only the JSON result (no formatted report).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full result as JSON to this path.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Render PE similarity overlays onto the video and save here.")
    parser.add_argument("--visualize_dir", type=str, default=None,
                        help="Save per-chunk frame visualization PNGs to this directory. "
                             "Creates one image per chunk showing the N sampled frames "
                             "alongside per-domain similarity scores.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-chunk domain scores and similarities.")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, preprocess, tokenizer = load_pe_model(args.pe_model, device)

    result = assess_pe(
        video_path=args.video,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        num_frames=args.num_frames,
        chunk_duration=args.chunk_duration,
        sim_threshold=args.sim_threshold,
        visualize_dir=args.visualize_dir,
        debug=args.debug,
    )

    if args.json_only:
        # tensors not JSON-serializable; convert to plain dicts
        import copy
        out = copy.deepcopy(result)
        out.pop("pe_features", None)   # aliases plm_features, avoid duplicate
        print(json.dumps(out, indent=2, default=str))
    else:
        print_report(result)

    if args.save:
        import copy
        out = copy.deepcopy(result)
        out.pop("pe_features", None)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"Saved to: {args.save}")

    if args.output_video:
        render_assessment_video(args.video, result, args.output_video)


if __name__ == "__main__":
    main()
