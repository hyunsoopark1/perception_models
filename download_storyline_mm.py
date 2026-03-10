#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Download a multi-modal storyline dataset from a JSON file + S3.

JSON format
-----------
[
  {
    "dependent_id": 8615,
    "comment": "9월 1일 스토리라인입니다.",
    "s3_keys": [
      "s3://bucket/datalake/8615/2025-09-01/image_selection/xxx.jpg",
      "s3://bucket/datalake/8615/2025-09-01/video_generation/xxx.mp4",
      ...
    ]
  },
  ...
]

Output layout  ~/data/storyline_mm/
  0008615/
    text.txt
    xxx.jpg
    yyy.jpg
    ...

MP4 files are skipped. All image files are placed flat under the
zero-padded 7-digit folder (no date sub-directory).

S3 paths must be full s3:// URIs.  The sub-path below
datalake/<dependent_id>/ is preserved as the local directory structure.

Usage
-----
  python download_storyline_mm.py data.json
  python download_storyline_mm.py data.json --out ~/data/storyline_mm --workers 8
  python download_storyline_mm.py data.json --profile my-aws-profile
  python download_storyline_mm.py data.json --start 5   # resume from entry index 5
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3:// URI."""
    uri = uri.strip()
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_s3_file(s3_client, uri: str, dest: Path) -> None:
    bucket, key = parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(dest))


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(idx: int, entry: dict, out_root: Path, s3_client) -> list[str]:
    """
    Create the folder for this entry, write text.txt, download all s3_keys.

    All image files are placed flat under the 7-digit folder (mp4 files skipped):
      out_root/
        0000001/
          text.txt
          xxx.jpg
          yyy.jpg

    Returns a list of warning strings (empty = success).
    """
    dependent_id = str(entry["dependent_id"])
    comment = entry.get("comment", "")
    s3_keys = entry.get("s3_keys", [])

    folder = out_root / f"{idx:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    warnings = []

    # --- comment → text.txt ---
    (folder / "text.txt").write_text(comment, encoding="utf-8")

    # --- download each S3 key ---
    # Strip the leading "datalake/<dependent_id>/" from the S3 object key
    # so the rest becomes the relative path under the local folder.
    strip_prefix = f"datalake/{dependent_id}/"

    for uri in s3_keys:
        uri = uri.strip()
        if not uri:
            continue

        # Skip video files
        if uri.lower().endswith(".mp4"):
            continue

        try:
            _, key = parse_s3_uri(uri)
        except ValueError as e:
            warnings.append(f"Entry {idx} (dependent_id={dependent_id}): {e}")
            continue

        # Place all image files flat under the 7-digit folder (filename only)
        dest = folder / Path(key).name
        if dest.exists():
            continue  # resume support

        try:
            download_s3_file(s3_client, uri, dest)
        except ClientError as e:
            warnings.append(f"Entry {idx} (dependent_id={dependent_id}): S3 error for {uri!r}: {e}")
        except Exception as e:
            warnings.append(f"Entry {idx} (dependent_id={dependent_id}): Unexpected error for {uri!r}: {e}")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download multi-modal storyline dataset from JSON")
    p.add_argument("input_json", help="Path to JSON file")
    p.add_argument("--out", default="~/data/storyline_mm",
                   help="Output root directory (default: ~/data/storyline_mm)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download threads (default: 4)")
    p.add_argument("--profile", default=None,
                   help="AWS profile name (default: uses env / instance role)")
    p.add_argument("--start", type=int, default=0,
                   help="0-based entry index to start from (for resuming, default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading JSON: {args.input_json}")
    with open(args.input_json, encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        print("ERROR: JSON root must be a list of entries.", file=sys.stderr)
        sys.exit(1)

    total = len(entries)
    entries_to_process = entries[args.start:]
    print(f"  {total} entries found, processing {len(entries_to_process)} (start={args.start})")
    print(f"Output directory   : {out_root}")
    print(f"Download threads   : {args.workers}\n")

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    all_warnings = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_entry, args.start + i, entry, out_root, s3): args.start + i
            for i, entry in enumerate(entries_to_process)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                warns = future.result()
                all_warnings.extend(warns)
            except Exception as e:
                all_warnings.append(f"Entry {idx}: Fatal error: {e}")

            completed += 1
            if completed % 50 == 0 or completed == len(entries_to_process):
                print(f"  Progress: {completed}/{len(entries_to_process)}", flush=True)

    print(f"\nDone. {completed} entries processed.")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  {w}")
    else:
        print("No errors.")


if __name__ == "__main__":
    main()
