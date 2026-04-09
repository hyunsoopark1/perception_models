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
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Sentence encoder (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────────────

_encoder = None


def _load_encoder(model_name: str = "all-MiniLM-L6-v2"):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence encoder ({model_name}) ...")
        _encoder = SentenceTransformer(model_name)
    return _encoder


def _embed(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode a list of strings; returns (N, D) float32 array, L2-normalised."""
    enc = _load_encoder(model_name)
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


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
# Core search
# ──────────────────────────────────────────────────────────────────────────────


def find_best_clips(
    descriptions: dict,
    query: str,
    match_fields: list = None,
    month_filter: Optional[str] = None,
    encoder_model: str = "all-MiniLM-L6-v2",
    top_k: int = 1,
    threshold: float = 0.0,
) -> dict:
    """Find the best-matching clip for each calendar month.

    Args:
        descriptions:  Dict loaded from descriptions.json (key = clip_path).
        query:         Query text.
        match_fields:  Which fields to use for matching
                       (default: ["description"]).
        month_filter:  If set (e.g. "2025-06"), only search within that month.
        top_k:         Number of clips to return per month (default 1).
        threshold:     Minimum cosine similarity to include a clip (default 0.0).
                       Months where the best score is below this are excluded.

    Returns:
        {year_month: [{"clip_path": ..., "score": ..., "entry": ...}, ...]}
    """
    if match_fields is None:
        match_fields = ["description"]

    # Group entries by year_month
    by_month: dict = {}
    for clip_path, entry in descriptions.items():
        ym = entry.get("year_month", "")
        if not ym:
            # Try to extract from clip_path
            m = re.search(r"(\d{4}-\d{2})", clip_path)
            ym = m.group(1) if m else "unknown"
            entry["year_month"] = ym
        if month_filter and ym != month_filter:
            continue
        by_month.setdefault(ym, []).append((clip_path, entry))

    if not by_month:
        logger.warning("No clips matched the filter criteria.")
        return {}

    # Build text corpus and encode
    all_keys   = []   # (year_month, clip_path) pairs
    all_texts  = []

    for ym, items in sorted(by_month.items()):
        for clip_path, entry in items:
            text = _build_match_text(entry, match_fields)
            if not text:
                continue
            all_keys.append((ym, clip_path))
            all_texts.append(text)

    if not all_texts:
        logger.warning("No text found in entries for the selected fields.")
        return {}

    logger.info(f"Encoding {len(all_texts)} clip texts and query ...")
    corpus_embs = _embed(all_texts, encoder_model)
    query_emb   = _embed([query], encoder_model)[0]

    sims = corpus_embs @ query_emb   # cosine similarity

    # Build per-month ranked results
    month_scores: dict = {}
    for (ym, clip_path), sim in zip(all_keys, sims):
        month_scores.setdefault(ym, []).append({
            "clip_path": clip_path,
            "score":     float(sim),
            "entry":     descriptions[clip_path],
        })

    # Sort and keep top_k per month; drop months below threshold
    results = {}
    for ym, items in month_scores.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        best_score = items[0]["score"]
        if best_score < threshold:
            logger.info(f"  Month {ym}: best score {best_score:.3f} < threshold {threshold:.3f} — skipped.")
            continue
        results[ym] = items[:top_k]

    return results


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
           --query "child climbing independently" \\
           --output_dir best_clips/

  %(prog)s --descriptions descriptions.json \\
           --query "first steps unsteady walking" \\
           --month 2025-06 \\
           --top_k 3

  %(prog)s --descriptions descriptions.json \\
           --query "runs, says two-word phrases" \\
           --match_fields description stage evidence
        """,
    )
    parser.add_argument("--descriptions",  type=str, required=True,
                        help="descriptions.json from extract_description.py.")
    parser.add_argument("--query",         type=str, required=True,
                        help="Query text to match against.")
    parser.add_argument("--output_dir",    type=str, default=None,
                        help="Copy best clips here. If omitted, just prints results.")
    parser.add_argument("--month",         type=str, default=None,
                        help="Restrict search to one month, e.g. 2025-06.")
    parser.add_argument("--match_fields",  type=str, nargs="+",
                        default=["description"],
                        choices=["description", "stage", "evidence"],
                        help="Fields to use for similarity matching (default: description).")
    parser.add_argument("--top_k",         type=int, default=1,
                        help="Clips to select per month (default: 1).")
    parser.add_argument("--encoder",       type=str,
                        default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2).")
    parser.add_argument("--threshold",     type=float, default=0.0,
                        help="Minimum cosine similarity to include a month (default: 0.0). "
                             "Months where the best-matching clip scores below this are excluded.")
    parser.add_argument("--copy",          action="store_true",
                        help="Alias for --output_dir: copy files even if no dir set "
                             "(uses ./best_clips/).")
    args = parser.parse_args()

    # Load descriptions
    with open(args.descriptions) as f:
        descriptions = json.load(f)
    logger.info(f"Loaded {len(descriptions)} entries from {args.descriptions}")

    # Run search
    results = find_best_clips(
        descriptions=descriptions,
        query=args.query,
        match_fields=args.match_fields,
        month_filter=args.month,
        encoder_model=args.encoder,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    if not results:
        logger.info("No results.")
        return

    # Display results
    print(f"\nQuery: {args.query!r}")
    print(f"Match fields: {args.match_fields}")
    print()
    for ym in sorted(results):
        print(f"  Month {ym}:")
        for rank, item in enumerate(results[ym], 1):
            entry = item["entry"]
            print(f"    [{rank}] score={item['score']:.3f}  {Path(item['clip_path']).name}")
            if entry.get("stage"):
                print(f"         stage: {entry['stage']}")
            desc = entry.get("description", "")
            if desc:
                print(f"         desc:  {desc[:100]}{'...' if len(desc) > 100 else ''}")
        print()

    # Copy best clips if requested
    output_dir = args.output_dir or ("best_clips" if args.copy else None)
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        copied = 0
        for ym in sorted(results):
            for rank, item in enumerate(results[ym], 1):
                src = Path(item["clip_path"])
                if not src.exists():
                    logger.warning(f"Source not found: {src}")
                    continue
                # Rename to include month and rank for clarity
                suffix = f"_rank{rank:02d}" if args.top_k > 1 else ""
                dst = out / f"{ym}{suffix}_{src.name}"
                shutil.copy2(str(src), str(dst))
                logger.info(f"  Copied: {src.name} → {dst.name}")
                copied += 1
        logger.info(f"Copied {copied} clip(s) to {out}")

    # Save search results as JSON
    save_path = (Path(output_dir) / "search_results.json") if output_dir else None
    if save_path:
        serialisable = {
            ym: [{"clip_path": it["clip_path"], "score": it["score"],
                  "stage": it["entry"].get("stage", ""),
                  "description": it["entry"].get("description", "")}
                 for it in items]
            for ym, items in results.items()
        }
        with open(save_path, "w") as f:
            json.dump({"query": args.query, "results": serialisable}, f, indent=2)
        logger.info(f"Search results saved to: {save_path}")


if __name__ == "__main__":
    main()
