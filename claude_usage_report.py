#!/usr/bin/env python3
"""
Calculate monthly Claude Code usage for multiple users and export to CSV.

Usage:
    # Single user (current user):
    python claude_usage_report.py

    # Multiple users (specify their home dirs):
    python claude_usage_report.py --users alice:/home/alice bob:/home/bob

    # Custom output file:
    python claude_usage_report.py --output usage_report.csv

    # Filter by year/month:
    python claude_usage_report.py --month 2026-04
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Approximate Claude pricing (per 1M tokens) as of early 2026.
# Adjust these if your model/pricing differs.
PRICING = {
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-6":  {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5":   {"input":  0.80, "output":  4.00, "cache_write":  1.00, "cache_read": 0.08},
    # fallback
    "default":            {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
}


def get_price(model: str, kind: str) -> float:
    """Return price per token (not per million) for a given model and token type."""
    rates = PRICING.get(model) or PRICING.get(
        next((k for k in PRICING if model.startswith(k)), "default"), PRICING["default"]
    )
    return rates.get(kind, 0.0) / 1_000_000


def parse_usage_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Extract assistant messages with usage stats from a JSONL session file."""
    records = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("type") != "assistant":
                    continue
                msg = obj.get("message", {})
                usage = msg.get("usage")
                if not usage:
                    continue

                timestamp = obj.get("timestamp", "")
                model = msg.get("model", "unknown")

                records.append({
                    "timestamp": timestamp,
                    "model": model,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                })
    except (OSError, PermissionError) as e:
        print(f"  Warning: cannot read {jsonl_path}: {e}", file=sys.stderr)
    return records


def collect_user_usage(claude_dir: Path) -> list[dict]:
    """Walk ~/.claude/projects/**/*.jsonl and return all usage records."""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return []

    all_records = []
    for jsonl_file in projects_dir.rglob("*.jsonl"):
        all_records.extend(parse_usage_from_jsonl(jsonl_file))
    return all_records


def aggregate_monthly(records: list[dict], username: str) -> dict[tuple, dict]:
    """
    Aggregate records by (username, year-month, model).
    Returns dict keyed by (username, month_str, model).
    """
    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "api_calls": 0,
    })

    for rec in records:
        ts = rec.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            month_str = dt.strftime("%Y-%m")
        except ValueError:
            month_str = "unknown"

        model = rec.get("model", "unknown")
        key = (username, month_str, model)
        b = buckets[key]

        input_t  = rec["input_tokens"]
        output_t = rec["output_tokens"]
        cache_w  = rec["cache_creation_input_tokens"]
        cache_r  = rec["cache_read_input_tokens"]

        b["input_tokens"]          += input_t
        b["output_tokens"]         += output_t
        b["cache_creation_tokens"] += cache_w
        b["cache_read_tokens"]     += cache_r
        b["total_tokens"]          += input_t + output_t + cache_w + cache_r
        b["api_calls"]             += 1
        b["estimated_cost_usd"]    += (
            input_t  * get_price(model, "input")  +
            output_t * get_price(model, "output") +
            cache_w  * get_price(model, "cache_write") +
            cache_r  * get_price(model, "cache_read")
        )

    return buckets


def write_csv(rows: list[dict], output_path: str) -> None:
    fieldnames = [
        "username",
        "month",
        "model",
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "total_tokens",
        "estimated_cost_usd",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_user_specs(specs: list[str]) -> list[tuple[str, Path]]:
    """
    Parse user specs like 'alice:/home/alice' or just 'alice' (uses /home/alice).
    Returns list of (username, claude_dir).
    """
    users = []
    for spec in specs:
        if ":" in spec:
            name, home = spec.split(":", 1)
        else:
            name = spec
            home = f"/home/{spec}"
        claude_dir = Path(home) / ".claude"
        users.append((name, claude_dir))
    return users


def main():
    parser = argparse.ArgumentParser(description="Claude Code monthly usage report")
    parser.add_argument(
        "--users", nargs="+", metavar="USER[:HOME]",
        help="Users to report on (e.g. alice alice:/home/alice). "
             "Defaults to current user.",
    )
    parser.add_argument(
        "--output", default="claude_usage.csv",
        help="Output CSV file path (default: claude_usage.csv)",
    )
    parser.add_argument(
        "--month", metavar="YYYY-MM",
        help="Filter to a specific month (e.g. 2026-04)",
    )
    args = parser.parse_args()

    # Resolve user list
    if args.users:
        user_list = parse_user_specs(args.users)
    else:
        current_home = Path.home()
        current_user = os.environ.get("USER") or current_home.name or "current_user"
        user_list = [(current_user, current_home / ".claude")]

    all_rows = []

    for username, claude_dir in user_list:
        print(f"Processing {username} ({claude_dir}) ...")
        if not claude_dir.exists():
            print(f"  Skipping: {claude_dir} not found.")
            continue

        records = collect_user_usage(claude_dir)
        print(f"  Found {len(records)} usage records.")

        monthly = aggregate_monthly(records, username)

        for (uname, month, model), stats in monthly.items():
            if args.month and month != args.month:
                continue
            all_rows.append({
                "username": uname,
                "month": month,
                "model": model,
                "api_calls": stats["api_calls"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "cache_creation_tokens": stats["cache_creation_tokens"],
                "cache_read_tokens": stats["cache_read_tokens"],
                "total_tokens": stats["total_tokens"],
                "estimated_cost_usd": round(stats["estimated_cost_usd"], 6),
            })

    # Sort by username -> month -> model
    all_rows.sort(key=lambda r: (r["username"], r["month"], r["model"]))

    write_csv(all_rows, args.output)
    print(f"\nReport written to: {args.output}")
    print(f"Total rows: {len(all_rows)}")

    # Print a quick summary table
    if all_rows:
        print("\nSummary:")
        print(f"{'User':<20} {'Month':<10} {'Model':<25} {'Calls':>7} {'Total Tokens':>14} {'Cost USD':>10}")
        print("-" * 90)
        for r in all_rows:
            print(
                f"{r['username']:<20} {r['month']:<10} {r['model']:<25} "
                f"{r['api_calls']:>7} {r['total_tokens']:>14,} {r['estimated_cost_usd']:>10.4f}"
            )


if __name__ == "__main__":
    main()
