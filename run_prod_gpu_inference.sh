#!/usr/bin/env bash
#
# Loops over the dates in a CSV and runs:
#     python trigger_prod_gpu_inference.py --date {yyyy-mm-dd}
#
# - The CSV is expected to have a header row and a "Date" column as the
#   first field, formatted as M/D/YYYY (e.g. 8/13/2025).
# - Dates are normalized to yyyy-mm-dd and deduplicated, so each unique
#   day triggers exactly one inference run.
#
# Usage:
#     ./run_prod_gpu_inference.sh [path/to/Sequence__Sheet3.csv]

set -euo pipefail

CSV_FILE="${1:-Sequence__Sheet3.csv}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# Extract the first column (Date), skip the header, normalize M/D/YYYY -> yyyy-mm-dd,
# drop blanks, and deduplicate while preserving first-seen order.
mapfile -t DATES < <(
    tail -n +2 "$CSV_FILE" \
        | cut -d',' -f1 \
        | sed '/^[[:space:]]*$/d' \
        | awk -F'/' '{ printf "%04d-%02d-%02d\n", $3, $1, $2 }' \
        | awk '!seen[$0]++'
)

echo "Found ${#DATES[@]} unique date(s) to process."

for date in "${DATES[@]}"; do
    echo "=== Running inference for ${date} ==="
    python trigger_prod_gpu_inference.py --date "${date}"
done

echo "Done."
