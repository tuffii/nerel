#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/NEREL"
if [[ -d "$dest/.git" ]]; then
  echo "Dataset already exists: $dest"
  git -C "$dest" rev-parse HEAD
  exit 0
fi
git clone https://github.com/nerel-ds/NEREL.git "$dest"
git -C "$dest" checkout 8f07d3e410f322d5dbf6b68b13fbf4a4aa077fe9
