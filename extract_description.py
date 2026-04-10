# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Extract PLM-generated descriptions and developmental assessments for video clips.

Three tasks (combinable via flags):
  --describe   : free-form scene description
  --stage      : predict one of 7 developmental stages (0-3m … 25-36m)
  --evidence   : domain-level evidence (motor / autonomy / attention /
                 interaction / language)

Results are saved to a JSON file and updated incrementally so the script
can be safely interrupted and resumed.

Usage:
    # all three tasks on a clips folder
    python extract_description.py --clips_dir data/clips/ --all

    # description only (fast pass)
    python extract_description.py --clips_dir data/clips/ --describe

    # stage + evidence on existing JSON (skips already-processed clips)
    python extract_description.py --clips_dir data/clips/ --stage --evidence \\
        --output descriptions.json

    # explicit clip list from generate_clip.py output
    python extract_description.py --clips_json data/clips/clips.json --all
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from apps.plm.generate import load_consolidated_model_and_tokenizer
from generate_video_description import generate_description

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_CHILD_PRESENT = (
    "Is there a child (baby, toddler, or young child) visible in this video?\n"
    "Reply with exactly one word: yes or no."
)

PROMPT_DESCRIBE = "Describe what is happening in this video in detail."

PROMPT_STAGE = (
    "What is the developmental stage of the child in this video?\n"
    "Choose exactly ONE of these stages based on what you directly observe:\n\n"
    "  0-3 months  : lying/rolling, limited movement, crying or cooing\n"
    "  4-6 months  : reaching, rolling, sitting with support, babbling\n"
    "  7-9 months  : crawling, pulling up, exploring objects, babbling\n"
    "  10-12 months: pulling to stand, first steps, first words, pointing\n"
    "  13-18 months: walking, climbing, 10-50 words, parallel play\n"
    "  19-24 months: running, two-word phrases, symbolic play, dressing attempts\n"
    "  25-36 months: jumping, short sentences, cooperative play, dresses self\n\n"
    "Reply with the stage label only (e.g., \"13-18 months\")."
)

PROMPT_EVIDENCE = (
    "Describe the specific evidence you see in this video for each developmental domain.\n"
    "Be concrete — refer to actions and behaviors visible in the clip.\n\n"
    "Motor (movement, balance, hand and finger skills):\n"
    "Autonomy (self-care, independence, self-initiated activities):\n"
    "Attention (focus, sustained engagement, goal-directed behavior):\n"
    "Interaction (social behavior, response to people, caregiver relationship):\n"
    "Language (spoken words, gestures, communication attempts):\n\n"
    "Reply in this exact format:\n"
    "Motor: <evidence>\n"
    "Autonomy: <evidence>\n"
    "Attention: <evidence>\n"
    "Interaction: <evidence>\n"
    "Language: <evidence>"
)

STAGE_LABELS = [
    "0-3 months", "4-6 months", "7-9 months", "10-12 months",
    "13-18 months", "19-24 months", "25-36 months",
]

EVIDENCE_DOMAINS = ["Motor", "Autonomy", "Attention", "Interaction", "Language"]


# ──────────────────────────────────────────────────────────────────────────────
# PLM helpers
# ──────────────────────────────────────────────────────────────────────────────


def _run(video_path: str, prompt: str, model, tokenizer, config,
         num_frames: int = 8, temperature: float = 0.0,
         max_gen_len: int = 256) -> str:
    result = generate_description(
        video_path=video_path, model=model, tokenizer=tokenizer,
        config=config, prompt=prompt, num_frames=num_frames,
        temperature=temperature, max_gen_len=max_gen_len,
    )
    return result["description"]


def get_child_present(video_path, model, tokenizer, config,
                      num_frames=8, temperature=0.0) -> bool:
    """Return True if the model detects a child in the clip."""
    raw = _run(video_path, PROMPT_CHILD_PRESENT, model, tokenizer, config,
               num_frames=num_frames, temperature=temperature, max_gen_len=8)
    return raw.strip().lower().startswith("yes")


def get_description(video_path, model, tokenizer, config,
                    num_frames=8, temperature=0.0) -> str:
    return _run(video_path, PROMPT_DESCRIBE, model, tokenizer, config,
                num_frames=num_frames, temperature=temperature, max_gen_len=256)


def get_stage(video_path, model, tokenizer, config,
              num_frames=8, temperature=0.0) -> str:
    """Return the predicted stage label (one of STAGE_LABELS)."""
    raw = _run(video_path, PROMPT_STAGE, model, tokenizer, config,
               num_frames=num_frames, temperature=temperature, max_gen_len=32)
    # Try to match to a known label
    raw_lower = raw.strip().lower()
    for label in STAGE_LABELS:
        if label.lower() in raw_lower:
            return label
    # Fallback: return raw stripped text
    return raw.strip()


def get_evidence(video_path, model, tokenizer, config,
                 num_frames=8, temperature=0.0) -> dict:
    """Return per-domain evidence dict."""
    raw = _run(video_path, PROMPT_EVIDENCE, model, tokenizer, config,
               num_frames=num_frames, temperature=temperature, max_gen_len=1024)
    evidence = {}
    for domain in EVIDENCE_DOMAINS:
        pattern = re.compile(
            rf"^\s*{re.escape(domain)}\s*:\s*(.+?)(?=\n\s*(?:{'|'.join(EVIDENCE_DOMAINS)})\s*:|$)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        m = pattern.search(raw)
        evidence[domain.lower()] = m.group(1).strip() if m else ""
    evidence["_raw"] = raw
    return evidence


# ──────────────────────────────────────────────────────────────────────────────
# Clip collection helpers
# ──────────────────────────────────────────────────────────────────────────────


def _collect_clips(clips_dir: Optional[str], clips_json: Optional[str]) -> list:
    """Return list of clip dicts with at least {"clip_path": str}."""
    if clips_json:
        with open(clips_json) as f:
            data = json.load(f)
        # accept both list-of-dicts and the {metadata, clips} format
        if isinstance(data, list):
            return data
        return list(data.get("clips", {}).values())

    if clips_dir:
        clips = []
        for p in sorted(Path(clips_dir).iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                # Try to extract year_month and index from filename
                m = re.match(r"(\d{4}-\d{2})", p.stem)
                idx_m = re.search(r"_clip(\d+)", p.stem)
                clips.append({
                    "clip_path":   str(p),
                    "source_file": p.name,
                    "year_month":  m.group(1) if m else "",
                    "clip_index":  int(idx_m.group(1)) if idx_m else 0,
                })
        return clips
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────


def process_clips(
    clips: list,
    model, tokenizer, config,
    output_path: str,
    do_describe: bool = True,
    do_stage: bool = True,
    do_evidence: bool = True,
    num_frames: int = 8,
    temperature: float = 0.0,
    overwrite: bool = False,
) -> dict:
    """Process a list of clip dicts and save results to output_path JSON.

    The JSON is updated after each clip so progress is not lost on interruption.
    Returns the full results dict.
    """
    # Load existing results (for resumption)
    if os.path.exists(output_path) and not overwrite:
        with open(output_path) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing entries from {output_path}")
    else:
        results = {}

    total = len(clips)
    for i, clip in enumerate(clips):
        clip_path = clip["clip_path"]
        if not os.path.exists(clip_path):
            logger.warning(f"[{i+1}/{total}] Not found: {clip_path} — skipping")
            continue

        key = clip_path
        entry = results.get(key, {})
        entry.update({k: v for k, v in clip.items() if k not in entry})

        # Determine what needs to be run
        need_child    = overwrite or entry.get("child_present") is None
        need_describe = do_describe and (overwrite or not entry.get("description"))
        need_stage    = do_stage    and (overwrite or not entry.get("stage"))
        need_evidence = do_evidence and (overwrite or not entry.get("evidence"))

        if not (need_child or need_describe or need_stage or need_evidence):
            logger.info(f"[{i+1}/{total}] Skip (already processed): {Path(clip_path).name}")
            results[key] = entry
            continue

        t0 = time.time()
        logger.info(f"[{i+1}/{total}] Processing: {Path(clip_path).name}")

        # Child presence check — gates all other tasks
        if need_child:
            logger.info("  → child present? ...")
            entry["child_present"] = get_child_present(
                clip_path, model, tokenizer, config, num_frames, temperature
            )
            logger.info(f"     {'yes' if entry['child_present'] else 'no'}")

        if not entry.get("child_present"):
            logger.info("  No child detected — skipping description/stage/evidence.")
            results[key] = entry
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

        if need_describe:
            logger.info("  → description ...")
            entry["description"] = get_description(
                clip_path, model, tokenizer, config, num_frames, temperature
            )
            desc_preview = entry['description']
            logger.info(f"     {desc_preview!r}")

        if need_stage:
            logger.info("  → stage ...")
            entry["stage"] = get_stage(
                clip_path, model, tokenizer, config, num_frames, temperature
            )
            logger.info(f"     {entry['stage']!r}")

        if need_evidence:
            logger.info("  → evidence ...")
            entry["evidence"] = get_evidence(
                clip_path, model, tokenizer, config, num_frames, temperature
            )
            for dom in ["motor", "autonomy", "attention", "interaction", "language"]:
                val = entry["evidence"].get(dom, "")
                status = repr(val[:80]) if val else "(empty)"
                logger.info(f"     {dom}: {status}")

        elapsed = time.time() - t0
        remaining = (total - i - 1) * elapsed
        logger.info(f"  clip done in {elapsed:.1f}s | est. remaining: {remaining/60:.0f} min")

        results[key] = entry

        # Save after each clip (resumable)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Results saved to: {output_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract PLM descriptions / stage / evidence for video clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --clips_dir data/clips/ --all
  %(prog)s --clips_dir data/clips/ --describe
  %(prog)s --clips_json data/clips/clips.json --stage --evidence
  %(prog)s --clips_dir data/clips/ --all --output results.json --overwrite
        """,
    )

    # Input
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips_dir",  type=str,
                     help="Folder of clip files.")
    src.add_argument("--clips_json", type=str,
                     help="clips.json produced by generate_clip.py.")

    # Task selection
    parser.add_argument("--describe",  action="store_true",
                        help="Generate free-form scene description.")
    parser.add_argument("--stage",     action="store_true",
                        help="Predict developmental stage (0-3m … 25-36m).")
    parser.add_argument("--evidence",  action="store_true",
                        help="Generate per-domain developmental evidence.")
    parser.add_argument("--all",       action="store_true",
                        help="Run all three tasks.")

    # Model
    parser.add_argument("--ckpt",       type=str, default="facebook/Perception-LM-8B",
                        help="PLM checkpoint (default: facebook/Perception-LM-8B).")
    parser.add_argument("--num_frames", type=int,   default=8)
    parser.add_argument("--temperature",type=float, default=0.0)

    # Output
    parser.add_argument("--output",    type=str, default=None,
                        help="Output JSON path (default: <clips_dir>/descriptions.json).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process clips that already have results.")

    args = parser.parse_args()

    do_describe = args.describe or args.all
    do_stage    = args.stage    or args.all
    do_evidence = args.evidence or args.all

    if not (do_describe or do_stage or do_evidence):
        parser.error("Specify at least one of --describe, --stage, --evidence, or --all.")

    clips = _collect_clips(args.clips_dir, args.clips_json)
    if not clips:
        logger.error("No clips found.")
        return

    logger.info(f"Found {len(clips)} clip(s).")

    if args.output:
        output_path = args.output
    elif args.clips_dir:
        output_path = str(Path(args.clips_dir) / "descriptions.json")
    else:
        output_path = str(Path(args.clips_json).parent / "descriptions.json")

    logger.info(f"Loading model: {args.ckpt}  (this can take 2-4 min for 8B) ...")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)
    logger.info("Model loaded and ready.")

    tasks = [t for t, f in [("describe", do_describe), ("stage", do_stage), ("evidence", do_evidence)] if f]
    logger.info(f"Tasks: {tasks}  |  clips: {len(clips)}  |  output: {output_path}")

    process_clips(
        clips=clips,
        model=model, tokenizer=tokenizer, config=config,
        output_path=output_path,
        do_describe=do_describe,
        do_stage=do_stage,
        do_evidence=do_evidence,
        num_frames=args.num_frames,
        temperature=args.temperature,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
