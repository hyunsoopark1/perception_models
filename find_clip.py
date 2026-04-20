# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Given a query text, find the best-matching clip for each calendar month and
copy them to an output folder.

Clips are grouped by the year-month prefix of their source MOV filename
(e.g. "2025-06-15.MOV" → month "2025-06").  Within each month the clip
whose description (or stage / evidence) has the highest semantic similarity
to the query is selected.

Requires the descriptions.json produced by extract_description.py.

Usage:
    python find_clip.py \\
        --descriptions data/clips/descriptions.json \\
        --query "child climbing independently on playground equipment" \\
        --output_dir best_clips/

    # search only within a specific month
    python find_clip.py \\
        --descriptions descriptions.json \\
        --query "first steps, walking unsteadily" \\
        --month 2025-06 \\
        --output_dir best_clips/

    # use stage + evidence text for matching (not just description)
    python find_clip.py \\
        --descriptions descriptions.json \\
        --query "runs and climbs, says two-word phrases" \\
        --match_fields description stage evidence \\
        --output_dir best_clips/
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Sentence encoder (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────────────

_encoder = None
_encoder_name = None


def _load_encoder(model_name: str = "all-mpnet-base-v2"):
    global _encoder, _encoder_name
    if _encoder is None or _encoder_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence encoder ({model_name}) ...")
        _encoder = SentenceTransformer(model_name)
        _encoder_name = model_name
    return _encoder


def _embed(texts: list, model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """Encode a list of strings; returns (N, D) float32 array, L2-normalised."""
    enc = _load_encoder(model_name)
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _embed_query(query: str, model_name: str) -> np.ndarray:
    """Encode the query, adding the BGE instruction prefix when appropriate."""
    enc = _load_encoder(model_name)
    # BGE models benefit from a retrieval instruction on the query side
    if "bge" in model_name.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    return enc.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Text building
# ──────────────────────────────────────────────────────────────────────────────


def _build_match_text(entry: dict, fields: list) -> str:
    """Concatenate the requested fields from one clip entry into a single string."""
    parts = []
    for field in fields:
        if field == "description" and entry.get("description"):
            parts.append(entry["description"])
        elif field == "stage" and entry.get("stage"):
            parts.append(f"Developmental stage: {entry['stage']}")
        elif field == "evidence" and entry.get("evidence"):
            ev = entry["evidence"]
            for domain in ("motor", "autonomy", "attention", "interaction", "language"):
                text = ev.get(domain, "")
                if text:
                    parts.append(f"{domain.capitalize()}: {text}")
    return " ".join(parts).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Period grouping
# ──────────────────────────────────────────────────────────────────────────────


def _period_label(year_month: str, period: str) -> str:
    """Map a year_month string to a period bucket label.

    period="month"   → "2025-06"
    period="quarter" → "2025-Q2"
    """
    if period == "month":
        return year_month
    try:
        year, mon = year_month.split("-")
        q = (int(mon) - 1) // 3 + 1
        return f"{year}-Q{q}"
    except Exception:
        return year_month


# ──────────────────────────────────────────────────────────────────────────────
# Core search
# ──────────────────────────────────────────────────────────────────────────────


def _exact_score(text: str, query: str) -> float:
    """Fraction of query words found in text (case-insensitive, whole-word match).

    Returns 1.0 only when every query word appears in the text.
    Returns 0.0 when no query words match.
    """
    words = re.findall(r"\w+", query.lower())
    if not words:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text_lower))
    return matched / len(words)


def find_best_clips(
    descriptions: dict,
    query: str,
    match_fields: list = None,
    month_filter: Optional[str] = None,
    encoder_model: str = "all-mpnet-base-v2",
    top_k: int = 1,
    threshold: float = 0.0,
    period: str = "quarter",
    exact: bool = False,
) -> dict:
    """Find the best-matching clip for each time period.

    Args:
        descriptions:  Dict loaded from descriptions.json (key = clip_path).
        query:         Query text.
        match_fields:  Which fields to use for matching (default: ["description"]).
        month_filter:  If set (e.g. "2025-06"), only search within that month.
        top_k:         Number of clips to return per period (default 1).
        threshold:     Minimum cosine similarity to include a period (default 0.0).
        period:        Grouping granularity: "month" or "quarter" (default: "quarter").

    Returns:
        {period_label: [{"clip_path": ..., "score": ..., "entry": ...}, ...]}
    """
    if match_fields is None:
        match_fields = ["description"]

    # Group entries by period
    skipped_no_child = 0
    by_period: dict = {}
    for clip_path, entry in descriptions.items():
        if entry.get("child_present") is False:
            skipped_no_child += 1
            continue
        ym = entry.get("year_month", "")
        if not ym:
            m = re.search(r"(\d{4}-\d{2})", clip_path)
            ym = m.group(1) if m else "unknown"
            entry["year_month"] = ym
        if month_filter and ym != month_filter:
            continue
        pk = _period_label(ym, period)
        by_period.setdefault(pk, []).append((clip_path, entry))

    if skipped_no_child:
        logger.info(f"Skipped {skipped_no_child} clip(s) with no child detected.")
    if not by_period:
        logger.warning("No clips matched the filter criteria.")
        return {}

    # Build text corpus and encode
    all_keys  = []   # (period_key, clip_path) pairs
    all_texts = []

    for pk, items in sorted(by_period.items()):
        for clip_path, entry in items:
            text = _build_match_text(entry, match_fields)
            if not text:
                continue
            all_keys.append((pk, clip_path))
            all_texts.append(text)

    if not all_texts:
        logger.warning("No text found in entries for the selected fields.")
        return {}

    if exact:
        logger.info(f"Exact word matching for {len(all_texts)} clips ...")
        sims = [_exact_score(t, query) for t in all_texts]
    else:
        logger.info(f"Encoding {len(all_texts)} clip texts and query ...")
        corpus_embs = _embed(all_texts, encoder_model)
        query_emb   = _embed_query(query, encoder_model)
        sims = list(corpus_embs @ query_emb)

    # Build per-period ranked results
    period_scores: dict = {}
    for (pk, clip_path), sim in zip(all_keys, sims):
        period_scores.setdefault(pk, []).append({
            "clip_path": clip_path,
            "score":     float(sim),
            "entry":     descriptions[clip_path],
        })

    # Sort and keep top_k per period; drop periods below threshold
    results = {}
    for pk, items in period_scores.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        best_score = items[0]["score"]
        if best_score < threshold:
            logger.info(f"  Period {pk}: best score {best_score:.3f} < threshold {threshold:.3f} — skipped.")
            continue
        results[pk] = items[:top_k]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# PLM-based verification
# ──────────────────────────────────────────────────────────────────────────────


def _plm_verify(clip_path: str, query: str, model, tokenizer, config,
                num_frames: int = 8) -> bool:
    """Ask PLM 8B whether the query activity is visible in the clip.

    Returns True if the model answers 'yes'.
    """
    from generate_video_description import generate_description
    prompt = (
        f"Does this video show: {query}?\n"
        "Answer with exactly one word: yes or no."
    )
    result = generate_description(
        video_path=clip_path, model=model, tokenizer=tokenizer,
        config=config, prompt=prompt, num_frames=num_frames,
        temperature=0.0, max_gen_len=8,
    )
    return result["description"].strip().lower().startswith("yes")


def plm_rerank(candidates: dict, query: str, model, tokenizer, config,
               num_frames: int = 8) -> dict:
    """Re-rank/filter candidates using PLM yes/no verification per clip.

    Args:
        candidates: Output of find_best_clips — {period: [{clip_path, score, entry}]}.
        query:      The original search query.
        model/tokenizer/config: Loaded PLM 8B model.

    Returns:
        Filtered dict keeping only periods where PLM answered 'yes' for
        the top candidate. Score updated to 1.0 (yes) or 0.0 (no).
    """
    results = {}
    for pk, items in candidates.items():
        verified = []
        for item in items:
            clip_path = item["clip_path"]
            if not os.path.exists(clip_path):
                logger.warning(f"  PLM verify: file not found — {clip_path}")
                continue
            logger.info(f"  PLM verify [{pk}]: {Path(clip_path).name} ...")
            yes = _plm_verify(clip_path, query, model, tokenizer, config, num_frames)
            logger.info(f"    → {'yes ✓' if yes else 'no ✗'}  (embed score was {item['score']:.3f})")
            if yes:
                verified.append({**item, "score": 1.0, "plm_verified": True})
        if verified:
            results[pk] = verified
        else:
            logger.info(f"  Period {pk}: no clips verified by PLM — skipped.")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Compilation video
# ──────────────────────────────────────────────────────────────────────────────


def _letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize frame into target_w×target_h canvas, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _wrapped_lines(text: str, font, scale: float, thickness: int, max_w: int) -> list:
    """Split text into lines that fit within max_w pixels."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw > max_w and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_overlay(frame: np.ndarray, date_str: str, stage_str: str,
                  description: str, evidence: dict, overlay_h: int,
                  score: float = 0.0) -> np.ndarray:
    """Draw a semi-transparent text panel at the bottom of frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_x = 18
    max_text_w = w - margin_x * 2

    # Semi-transparent dark band
    roi = frame[h - overlay_h:h, :]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.6, roi, 0.4, 0, roi)
    frame[h - overlay_h:h, :] = roi

    y = h - overlay_h + 26

    # Line 1: date + stage + score (highlighted)
    header = f"{date_str}   |   {stage_str}   |   score: {score:.3f}"
    cv2.putText(frame, header, (margin_x, y), font, 0.65,
                (255, 210, 60), 2, cv2.LINE_AA)
    y += 30

    # Description
    if description:
        for line in _wrapped_lines(description, font, 0.48, 1, max_text_w):
            cv2.putText(frame, line, (margin_x, y), font, 0.48,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += 21
            if y > h - 8:
                return frame
        y += 4

    # Evidence / reasoning (one line per domain)
    domain_colors = {
        "motor":       (160, 230, 160),
        "autonomy":    (160, 200, 255),
        "attention":   (255, 200, 160),
        "interaction": (230, 160, 230),
        "language":    (160, 240, 240),
    }
    for domain in ("motor", "autonomy", "attention", "interaction", "language"):
        text = evidence.get(domain, "").strip()
        if not text:
            continue
        label = f"{domain.capitalize()}: "
        color = domain_colors[domain]
        (lw, _), _ = cv2.getTextSize(label, font, 0.42, 1)
        cv2.putText(frame, label, (margin_x, y), font, 0.42, color, 1, cv2.LINE_AA)
        for line in _wrapped_lines(text, font, 0.42, 1, max_text_w - lw):
            cv2.putText(frame, line, (margin_x + lw, y), font, 0.42,
                        (190, 210, 190), 1, cv2.LINE_AA)
            y += 19
            lw = 0  # indent only first line
            if y > h - 6:
                return frame
    return frame


def create_compilation(results: dict, output_path: str,
                       target_w: int = 1280, target_h: int = 720,
                       fps_out: float = 30.0) -> None:
    """Concatenate selected clips into a compilation video with text overlay.

    Args:
        results:     Output of find_best_clips — {period: [{clip_path, score, entry}]}.
        output_path: Path for the output mp4 file.
        target_w/h:  Output resolution (default 1280×720).
        fps_out:     Output frame rate (default 30).
    """
    overlay_h = 240  # px reserved for text panel

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_out, (target_w, target_h))

    # Sort periods chronologically
    ordered = []
    for pk in sorted(results.keys()):
        for item in results[pk]:
            ordered.append((pk, item["clip_path"], item["entry"], item["score"]))

    logger.info(f"Creating compilation: {len(ordered)} clip(s) → {output_path}")

    for period_key, clip_path, entry, score in ordered:
        # Extract full date from filename
        dm = re.search(r"(\d{4}-\d{2}-\d{2})", clip_path)
        date_str = dm.group(1) if dm else entry.get("year_month", "")
        stage_str = entry.get("stage", "")
        description = entry.get("description", "")
        evidence = entry.get("evidence", {})
        if isinstance(evidence, dict):
            evidence.pop("_raw", None)

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open: {clip_path}")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_out
        # Sample frames to match output fps
        frame_step = max(1, round(src_fps / fps_out))
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % frame_step == 0:
                frame = _letterbox(frame, target_w, target_h)
                frame = _draw_overlay(frame, date_str, stage_str,
                                      description, evidence, overlay_h, score)
                writer.write(frame)
            fi += 1

        cap.release()
        logger.info(f"  [{period_key}] {Path(clip_path).name} — {date_str} | {stage_str}")

    writer.release()
    logger.info(f"Compilation saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Find the best-matching clip per month given a query text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --descriptions data/clips/descriptions.json \\
           --query "child walking independently" \\
           --compile compilation.mp4

  %(prog)s --descriptions descriptions.json \\
           --query "first steps unsteady walking" \\
           --period month --compile out.mp4

  %(prog)s --descriptions descriptions.json \\
           --query "runs, says two-word phrases" \\
           --match_fields description stage evidence \\
           --compile out.mp4
        """,
    )
    parser.add_argument("--descriptions",  type=str, required=True,
                        help="descriptions.json from extract_description.py.")
    parser.add_argument("--query",         type=str, required=True,
                        help="Query text to match against.")
    parser.add_argument("--month",         type=str, default=None,
                        help="Restrict search to one month, e.g. 2025-06.")
    parser.add_argument("--match_fields",  type=str, nargs="+",
                        default=["description"],
                        choices=["description", "stage", "evidence"],
                        help="Fields to use for similarity matching (default: description).")
    parser.add_argument("--period",         type=str, default="quarter",
                        choices=["month", "quarter"],
                        help="Grouping granularity: 'month' or 'quarter' (default: quarter).")
    parser.add_argument("--top_k",         type=int, default=1,
                        help="Clips to select per period (default: 1).")
    parser.add_argument("--encoder",       type=str,
                        default="all-mpnet-base-v2",
                        help="Sentence transformer model (default: all-mpnet-base-v2). "
                             "Try BAAI/bge-large-en-v1.5 for best quality.")
    parser.add_argument("--threshold",     type=float, default=0.0,
                        help="Minimum score to include a period (default: 0.0).")
    parser.add_argument("--exact",          action="store_true",
                        help="Match query words exactly (whole-word, case-insensitive) instead "
                             "of semantic similarity. Score = fraction of query words found in text.")
    parser.add_argument("--compile",        type=str, default=None,
                        nargs="?", const="", metavar="OUTPUT_VIDEO",
                        help="Create a compilation video. Optionally specify filename "
                             "(e.g. --compile out.mp4). If no name given, auto-generates "
                             "from the query (e.g. compilation_child_dancing.mp4).")
    parser.add_argument("--plm_verify",     action="store_true",
                        help="Use PLM 8B to verify each candidate clip with a yes/no question. "
                             "More accurate than embedding similarity. Requires --ckpt.")
    parser.add_argument("--ckpt",           type=str, default="facebook/Perception-LM-8B",
                        help="PLM checkpoint for --plm_verify (default: facebook/Perception-LM-8B).")
    parser.add_argument("--plm_top_k",      type=int, default=3,
                        help="Number of candidates per period to send to PLM for verification "
                             "(default: 3). Higher = more accurate, slower.")
    args = parser.parse_args()

    # Load descriptions
    with open(args.descriptions) as f:
        descriptions = json.load(f)
    logger.info(f"Loaded {len(descriptions)} entries from {args.descriptions}")

    # Run initial embedding/exact search
    # When PLM verify is on, fetch more candidates per period to give PLM options
    initial_top_k = args.plm_top_k if args.plm_verify else args.top_k
    results = find_best_clips(
        descriptions=descriptions,
        query=args.query,
        match_fields=args.match_fields,
        month_filter=args.month,
        encoder_model=args.encoder,
        top_k=initial_top_k,
        threshold=args.threshold,
        period=args.period,
        exact=args.exact,
    )

    if not results:
        logger.info("No results.")
        return

    # PLM verification pass
    if args.plm_verify:
        from apps.plm.generate import load_consolidated_model_and_tokenizer
        logger.info(f"Loading PLM model for verification: {args.ckpt} ...")
        model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)
        logger.info("PLM loaded. Verifying candidates ...")
        results = plm_rerank(results, args.query, model, tokenizer, config)
        # Trim to requested top_k after verification
        results = {pk: items[:args.top_k] for pk, items in results.items()}
        if not results:
            logger.info("No clips verified by PLM.")
            return

    # Display results
    print(f"\nQuery: {args.query!r}")
    print(f"Period: {args.period}  |  Match fields: {args.match_fields}")
    print()
    for pk in sorted(results):
        print(f"  {pk}:")
        for rank, item in enumerate(results[pk], 1):
            entry = item["entry"]
            print(f"    [{rank}] score={item['score']:.3f}  {Path(item['clip_path']).name}")
            if entry.get("stage"):
                print(f"         stage: {entry['stage']}")
            desc = entry.get("description", "")
            if desc:
                print(f"         desc:  {desc[:100]}{'...' if len(desc) > 100 else ''}")
        print()

    # Create compilation video
    if args.compile is not None:
        # Auto-generate filename from query if --compile given without a path
        compile_path = args.compile
        if not compile_path:
            slug = re.sub(r"[^\w]+", "_", args.query.strip().lower()).strip("_")[:40]
            compile_path = f"compilation_{slug}.mp4"
            logger.info(f"Output video: {compile_path}")
        create_compilation(results, compile_path)


if __name__ == "__main__":
    main()
