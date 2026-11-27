"""Build test reports for every saved checkpoint, in both evaluation modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rich_eval
DATASET = str(ROOT / "data" / "NEREL" / "NEREL-v1.1")


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    efficiency = payload.get("efficiency") or {}
    return bool(payload.get("documents")) and "parameters" in efficiency


def main():
    checkpoints = [
        path for path in sorted((ROOT / "checkpoints").iterdir())
        if path.is_dir() and (path / "model.safetensors").exists()
    ]
    for checkpoint in checkpoints:
        for mode in ("flat", "nested"):
            target = rich_eval.report_path(checkpoint, "test", DATASET, mode)
            if complete(target):
                print("SKIP", target.name, flush=True)
                continue
            print("EVAL", checkpoint.name, mode, flush=True)
            saved = rich_eval.save_report(checkpoint, "test", DATASET, 1000, mode)
            payload = json.loads(saved.read_text(encoding="utf-8"))
            strict = payload["strict"]
            for key in ("by_char_length", "by_token_length", "by_doc_length"):
                false_positives = sum(row["fp"] for row in payload[key])
                if false_positives != strict["fp"]:
                    raise SystemExit(f"{saved.name} {key} fp {false_positives} != {strict['fp']}")
            print(
                "SAVED", saved.name,
                f"F1={strict['f1']:.4f}",
                f"sec={payload['efficiency']['inference_seconds']:.2f}",
                flush=True,
            )
    print("STUDY EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
