# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Query-based video compilation using PerceptionLM (PLM) descriptions.

Finds clips whose PLM-generated descriptions best match a free-text query,
then assembles them into a compilation video with text overlay.

Requires a descriptions.json produced by preprocess_videos.py.

Scoring uses TF-IDF cosine similarity between the query and each clip
description — no additional model loading required.

Usage:
    python query_video_compilation.py \\
        --descriptions ./output/descriptions.json \\
        --query "a child playing with a dog" \\
        --output child_dog.mp4

    # Return the 5 most relevant clips, skip text overlay
    python query_video_compilation.py \\
        --descriptions ./output/descriptions.json \\
        --query "sunset over the ocean" \\
        --output sunset.mp4 \\
        --top_k 5 \\
        --no_overlay

    # Show ranked scores without creating a video
    python query_video_compilation.py \\
        --descriptions ./output/descriptions.json \\
        --query "people dancing" \\
        --dry_run
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from compile_story_video import create_compilation_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Query scoring
# ──────────────────────────────────────────────────────────────────────────────


def score_clips_by_query(
    clips_with_descriptions: List[dict],
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> List[dict]:
    """Rank clips by TF-IDF cosine similarity to a text query.

    Fits a TF-IDF vocabulary over all clip descriptions plus the query,
    then measures cosine similarity between the query vector and each
    clip's description vector.

    Args:
        clips_with_descriptions: List of dicts with ``clip_path`` and
            ``description`` keys (from preprocess_videos.py output).
        query: Free-text search query, e.g. "a child playing with a dog".
        top_k: Maximum number of clips to return.
        min_score: Minimum similarity score threshold (0.0–1.0). Clips
            scoring below this are excluded even if top_k is not reached.

    Returns:
        List of up to ``top_k`` clip dicts sorted by descending relevance,
        each with an added ``_query_score`` key (float, 0.0–1.0).
    """
    descriptions = [item.get("description", "") for item in clips_with_descriptions]

    # Append the query as the last document so the vocabulary includes query terms
    all_texts = descriptions + [query]

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[-1]          # last row
    desc_matrix = tfidf_matrix[:-1]       # all clip rows

    similarities = cosine_similarity(query_vec, desc_matrix)[0]

    ranked = []
    for item, score in zip(clips_with_descriptions, similarities):
        entry = dict(item)
        entry["_query_score"] = float(score)
        ranked.append(entry)

    ranked.sort(key=lambda x: x["_query_score"], reverse=True)

    # Filter by minimum score then cap at top_k
    ranked = [r for r in ranked if r["_query_score"] >= min_score]
    return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile clips that best match a text query using PLM descriptions. "
            "Run preprocess_videos.py first to generate the descriptions JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --descriptions ./output/descriptions.json \\
           --query "a child playing with a dog" --output child_dog.mp4

  %(prog)s --descriptions ./output/descriptions.json \\
           --query "sunset over the ocean" --output sunset.mp4 --top_k 5

  %(prog)s --descriptions ./output/descriptions.json \\
           --query "people dancing" --dry_run
        """,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--descriptions",
        type=str,
        required=True,
        help="Path to descriptions.json produced by preprocess_videos.py.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help='Text query describing the content you want, e.g. "a child playing with a dog".',
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default="query_compilation.mp4",
        help="Output video path (default: query_compilation.mp4).",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable text overlay on the output video.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print ranked clips and scores without creating a video.",
    )

    # ── Retrieval ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-matching clips to include (default: 10).",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help=(
            "Minimum TF-IDF cosine similarity score to include a clip (0.0–1.0). "
            "Useful to exclude completely unrelated clips (default: 0.0)."
        ),
    )

    args = parser.parse_args()

    # ── Load descriptions ──────────────────────────────────────────────────────
    if not os.path.exists(args.descriptions):
        logger.error(f"Descriptions file not found: {args.descriptions}")
        raise SystemExit(1)

    with open(args.descriptions) as f:
        clips_with_desc = json.load(f)

    logger.info(
        f"Loaded {len(clips_with_desc)} clip descriptions from {args.descriptions}"
    )

    # ── Score by query ─────────────────────────────────────────────────────────
    logger.info(f'Scoring clips against query: "{args.query}"')
    ranked_clips = score_clips_by_query(
        clips_with_descriptions=clips_with_desc,
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not ranked_clips:
        logger.error(
            f"No clips matched the query with min_score={args.min_score}. "
            "Try lowering --min_score or rephrasing the query."
        )
        raise SystemExit(1)

    # ── Print ranked results ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f'Query: "{args.query}"')
    print(f"Matched clips: {len(ranked_clips)} / {len(clips_with_desc)}")
    print(f"{'=' * 60}")
    for i, item in enumerate(ranked_clips, 1):
        score = item["_query_score"]
        desc = item.get("description", "")
        print(f"  {i:2d}. [score={score:.3f}] {Path(item['clip_path']).name}")
        print(f"       {desc[:80]}{'...' if len(desc) > 80 else ''}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        logger.info("Dry run — no video created.")
        return

    # ── Assemble video ─────────────────────────────────────────────────────────
    logger.info(f"Assembling {len(ranked_clips)} clips into: {args.output}")
    create_compilation_video(
        ordered_clips=ranked_clips,
        output_path=args.output,
        overlay_text=not args.no_overlay,
    )

    logger.info(f"Done. Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
