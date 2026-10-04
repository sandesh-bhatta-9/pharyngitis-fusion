#!/usr/bin/env bash
# Download PGUPharyngitis (Shojaei et al., Sci Data 2025; CC BY 4.0) from Figshare into data/raw/
# and check the file against the MD5 checksum published by Figshare.
# Run from the project root:  bash scripts/download_data.sh
set -eo pipefail
URL="https://ndownloader.figshare.com/files/51543050"
MD5="998a7d12bc6e3192e61b8d2a3293d868"
OUT="data/raw/PGUPharyngitis.zip"

mkdir -p data/raw
if [[ ! -f "$OUT" ]]; then
  echo "Downloading PGUPharyngitis.zip (191 MB) from Figshare..."
  curl -L --fail -o "$OUT" "$URL"
fi

if command -v md5 >/dev/null; then got=$(md5 -q "$OUT"); else got=$(md5sum "$OUT" | cut -d' ' -f1); fi
if [[ "$got" != "$MD5" ]]; then
  echo "Checksum mismatch: expected $MD5, got $got. Delete $OUT and try again." >&2
  exit 1
fi
echo "Checksum OK."

unzip -q -o "$OUT" -d data/raw
echo "Done: data/raw/excel.xlsx and data/raw/data_image_pharyngitis_nature/"
