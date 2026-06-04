#!/usr/bin/env bash
#
# Reads the sequence CSV and, for each unique date, runs:
#     bash create_annotation.sh {yyyy-mm-dd} "{sync_ids}"
#
# sync_id encoding:
#   - sync_id 0  == 09:00
#   - sync_id 1  == 09:10   (10-minute intervals)
#   - sync_id n  == 09:00 + n*10 minutes
#
# For a row, sync_id = floor( minutes_since_09:00 / 10 ), and the row's
# ids span floor(start) .. floor(end) inclusive.
#   e.g. start 09:25, end 09:54  ->  "2 3 4 5"
#
# Rows that share the same date have their sync_ids concatenated (in CSV
# order) into a single call.
#
# Usage:
#     ./run_create_annotation.sh [path/to/Sequence__Sheet3.csv]

set -euo pipefail

CSV_FILE="${1:-Sequence__Sheet3.csv}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# Emit one "yyyy-mm-dd<TAB>sync ids" line per unique date.
while IFS=$'\t' read -r date ids; do
    echo "=== create_annotation.sh ${date} \"${ids}\" ==="
    bash create_annotation.sh "${date}" "${ids}"
done < <(
    awk -F',' '
        NR == 1 { next }                       # skip header
        {
            date = $1; start = $2; end = $3
            gsub(/\r/, "", end)                # strip trailing CR
            gsub(/[[:space:]]/, "", date)
            gsub(/[[:space:]]/, "", start)
            gsub(/[[:space:]]/, "", end)
            if (date == "") next

            split(date, d, "/")                # M/D/YYYY -> yyyy-mm-dd
            iso = sprintf("%04d-%02d-%02d", d[3], d[1], d[2])

            split(start, s, ":")
            split(end,   e, ":")
            sid = int(((s[1] - 9) * 60 + s[2]) / 10)
            eid = int(((e[1] - 9) * 60 + e[2]) / 10)

            ids = ""
            for (i = sid; i <= eid; i++)
                ids = ids (ids == "" ? "" : " ") i

            if (!(iso in acc)) { order[++n] = iso; acc[iso] = ids }
            else                 acc[iso] = acc[iso] " " ids
        }
        END {
            for (j = 1; j <= n; j++)
                print order[j] "\t" acc[order[j]]
        }
    ' "$CSV_FILE"
)

echo "Done."
