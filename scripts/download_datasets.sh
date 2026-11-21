#!/usr/bin/env bash
set -euo pipefail

# Wrapper script calling Python dataset downloader (works across Linux/macOS/Docker/WSL)
python scripts/download_datasets.py "$@"
