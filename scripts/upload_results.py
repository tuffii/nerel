"""Upload checkpoints and saved metrics to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Upload nerel3 checkpoints and reports to Hugging Face.")
    parser.add_argument("repo", help="Dataset repo id, for example username/nerel3-results")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=ROOT / "checkpoints",
        path_in_repo="checkpoints",
        ignore_patterns=[".training.json", "**/__pycache__/**"],
    )
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=ROOT / "results",
        path_in_repo="results",
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
