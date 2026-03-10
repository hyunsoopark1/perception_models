#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Download a multi-modal storyline dataset from a spreadsheet + S3.

Spreadsheet columns
-------------------
  A : text
  B : s3 image path  (required)
  C : s3 image path  (optional)
  D : s3 image path  (optional)

Output layout  ~/data/storyline_mm/
  0000001/
    text.txt
    image_0.<ext>
    image_1.<ext>   (if column C is present)
    image_2.<ext>   (if column D is present)
  0000002/
    ...

S3 paths may be:
  s3://bucket/key/path/image.jpg   (full URI)
  bucket/key/path/image.jpg        (bucket + key, no scheme)

Usage
-----
  python download_storyline_mm.py data.csv
  python download_storyline_mm.py data.xlsx --out ~/data/storyline_mm --workers 8
  python download_storyline_mm.py data.csv --profile my-aws-profile
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3:// URI or 'bucket/key' string."""
    uri = uri.strip()
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    else:
        parts = uri.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse S3 path: {uri!r}")
        bucket, key = parts[0], parts[1]
    return bucket, key


def download_s3_file(s3_client, uri: str, dest: Path) -> None:
    bucket, key = parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(dest))


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(row_idx: int, row: pd.Series, out_root: Path, s3_client) -> list[str]:
    """
    Create the folder for this row, write text.txt, download images.
    Returns a list of warning strings (empty = success).
    """
    folder = out_root / f"{row_idx:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    warnings = []

    # --- text ---
    text = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
    (folder / "text.txt").write_text(text, encoding="utf-8")

    # --- images (columns B, C, D → indices 1, 2, 3) ---
    for img_idx, col_pos in enumerate([1, 2, 3]):
        if col_pos >= len(row):
            break
        uri = row.iloc[col_pos]
        if pd.isna(uri) or str(uri).strip() == "":
            continue

        uri = str(uri).strip()
        ext = Path(urlparse(uri).path).suffix or ".jpg"
        dest = folder / f"image_{img_idx}{ext}"

        if dest.exists():
            continue  # already downloaded (resume support)

        try:
            download_s3_file(s3_client, uri, dest)
        except ClientError as e:
            warnings.append(f"Row {row_idx}: S3 error for {uri!r}: {e}")
        except Exception as e:
            warnings.append(f"Row {row_idx}: Unexpected error for {uri!r}: {e}")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_spreadsheet(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p, header=None)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, header=None)
    else:
        # Try CSV as fallback
        df = pd.read_csv(p, header=None)

    # Keep only the first 4 columns (A–D)
    df = df.iloc[:, :4]

    # Drop header row if it looks like one (non-numeric first cell of col B)
    first_b = str(df.iloc[0, 1]) if len(df) > 0 else ""
    if not first_b.startswith("s3://") and "/" not in first_b:
        print(f"Skipping header row: {df.iloc[0].tolist()}")
        df = df.iloc[1:].reset_index(drop=True)

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Download multi-modal storyline dataset")
    p.add_argument("spreadsheet", help="Path to CSV or Excel file")
    p.add_argument("--out", default="~/data/storyline_mm",
                   help="Output root directory (default: ~/data/storyline_mm)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download threads (default: 4)")
    p.add_argument("--profile", default=None,
                   help="AWS profile name (default: uses env / instance role)")
    p.add_argument("--start", type=int, default=1,
                   help="1-based row number to start from (for resuming)")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading spreadsheet: {args.spreadsheet}")
    df = load_spreadsheet(args.spreadsheet)
    total = len(df)
    print(f"  {total} rows found.")
    print(f"Output directory   : {out_root}")
    print(f"Download threads   : {args.workers}\n")

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    all_warnings = []
    completed = 0

    # Row numbers are 1-based (matching spreadsheet line numbers)
    rows = list(df.iterrows())  # (df_index, series)
    rows_to_process = [
        (df_idx + 1, series)           # 1-based row number
        for df_idx, series in rows
        if df_idx + 1 >= args.start
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_row, row_num, series, out_root, s3): row_num
            for row_num, series in rows_to_process
        }
        for future in as_completed(futures):
            row_num = futures[future]
            try:
                warns = future.result()
                all_warnings.extend(warns)
            except Exception as e:
                all_warnings.append(f"Row {row_num}: Fatal error: {e}")

            completed += 1
            if completed % 50 == 0 or completed == len(rows_to_process):
                print(f"  Progress: {completed}/{len(rows_to_process)}", flush=True)

    print(f"\nDone. {completed} rows processed.")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  {w}")
    else:
        print("No errors.")


if __name__ == "__main__":
    main()
