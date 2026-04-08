#!/usr/bin/env bash
#
# organize_mov_files.sh
#
# Find all .MOV files under a source directory (recursively) whose duration
# exceeds 5 seconds and copy them into a destination directory, renaming each
# file as YYYY-MM-DD-NUM.MOV. NUM is an incremental counter (starting at 1)
# used to disambiguate multiple videos that share the same creation date.
#
# Usage:
#   ./organize_mov_files.sh <source_dir> <destination_dir>
#
# Requirements:
#   - ffprobe (ships with ffmpeg) is used to read the duration and embedded
#     creation_time of each video.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <source_dir> <destination_dir>" >&2
    exit 1
fi

SRC_DIR="$1"
DEST_DIR="$2"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: source directory '$SRC_DIR' does not exist." >&2
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "Error: ffprobe is required (install ffmpeg) but was not found in PATH." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

# Print the duration of a video in seconds (floating point) using ffprobe.
get_duration() {
    ffprobe -v error \
        -show_entries format=duration \
        -of default=nokey=1:noprint_wrappers=1 \
        "$1" 2>/dev/null
}

# Print the file's modification date as YYYY-MM-DD, in a portable way.
get_mtime_date() {
    if stat --version >/dev/null 2>&1; then
        # GNU stat (Linux)
        stat -c %y "$1" | cut -d' ' -f1
    else
        # BSD stat (macOS / BSD)
        stat -f %Sm -t %Y-%m-%d "$1"
    fi
}

# Print the creation date as YYYY-MM-DD. Prefer the embedded creation_time
# metadata; fall back to the file's modification time if it is missing.
get_creation_date() {
    local file="$1"
    local creation_time
    creation_time=$(ffprobe -v error \
        -show_entries format_tags=creation_time \
        -of default=nokey=1:noprint_wrappers=1 \
        "$file" 2>/dev/null || true)
    if [[ -n "$creation_time" ]]; then
        # creation_time format: 2023-05-12T10:34:56.000000Z
        echo "${creation_time:0:10}"
    else
        get_mtime_date "$file"
    fi
}

copied=0
skipped=0

# Use -print0 / read -d '' to safely handle filenames with spaces or newlines.
while IFS= read -r -d '' file; do
    duration=$(get_duration "$file")
    if [[ -z "$duration" ]]; then
        echo "Skip (no duration): $file" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # Floating-point comparison via awk: keep only files longer than 5s.
    if ! awk -v d="$duration" 'BEGIN { exit !(d > 5) }'; then
        echo "Skip (<= 5s, ${duration}s): $file"
        skipped=$((skipped + 1))
        continue
    fi

    date_str=$(get_creation_date "$file")
    if [[ -z "$date_str" ]]; then
        echo "Skip (no date): $file" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # Pick the next free incremental number for this date.
    num=1
    while [[ -e "$DEST_DIR/${date_str}-${num}.MOV" ]]; do
        num=$((num + 1))
    done

    new_path="$DEST_DIR/${date_str}-${num}.MOV"
    echo "Copy: $file -> $new_path"
    cp -p -- "$file" "$new_path"
    copied=$((copied + 1))
done < <(find "$SRC_DIR" -type f -name "*.MOV" -print0)

echo "Done. Copied: $copied, Skipped: $skipped."
