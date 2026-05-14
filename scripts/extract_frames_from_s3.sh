#!/usr/bin/env bash
#
# Download 1.MP4 files from S3 and extract a frame every 2 minutes.
#
# S3 layout (under the given root):
#   <yyyy-mm-dd>/processed/sync<NNN>/1.MP4
#
# Output: <out_dir>/<yyyy-mm-dd>_sync<NNN>_<minute>.jpg
#
# Usage: ./extract_frames_from_s3.sh s3://my-bucket/root-folder ./frames
#
# Requires: aws cli, ffmpeg

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <s3-root-uri> <local-output-dir>" >&2
    exit 1
fi

S3_ROOT="${1%/}"
OUT_DIR="$2"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT_DIR"

FRAME_MINUTES=(0 2 4 6 8)
S3_BUCKET_URI=$(echo "$S3_ROOT" | sed -E 's#^(s3://[^/]+).*#\1#')

aws s3 ls --recursive "${S3_ROOT}/" \
    | awk '{print $4}' \
    | grep -E '/processed/sync[0-9]+/1\.MP4$' \
    | while read -r key; do

    date=$(echo "$key" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -n1)
    sync=$(echo "$key" | grep -oE 'sync[0-9]+' | head -n1)

    if [[ -z "$date" || -z "$sync" ]]; then
        echo "warn: could not parse date/sync from $key" >&2
        continue
    fi

    local_mp4="${TMP_DIR}/${date}_${sync}.MP4"

    echo "Processing ${date} ${sync}"
    aws s3 cp "${S3_BUCKET_URI}/${key}" "$local_mp4" --only-show-errors

    for minute in "${FRAME_MINUTES[@]}"; do
        out_file="${OUT_DIR}/${date}_${sync}_${minute}.jpg"
        if [[ -f "$out_file" ]]; then
            continue
        fi
        ts=$(printf "00:%02d:00" "$minute")
        ffmpeg -hide_banner -loglevel error -ss "$ts" -i "$local_mp4" \
            -frames:v 1 -q:v 2 -y "$out_file"
    done

    rm -f "$local_mp4"
done

echo "Done. Frames in: $OUT_DIR"
