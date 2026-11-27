"""Summarise repeated seeds and the Tiny2 learning-rate sweep for the metrics page."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rich_eval import load_saved_report, paired_bootstrap

ROOT = Path(__file__).resolve().parent


def _config(path: Path) -> dict:
    file = path / "training_config.json"
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def _family(config: dict) -> str:
    model = str(config.get("model", ""))
    if "tiny" in model:
        return "tiny2"
    if "base" in model:
        return "base"
    return "other"


def _history_best(path: Path) -> dict | None:
    file = path / "training_history.csv"
    if not file.exists():
        return None
    import csv
    rows = list(csv.DictReader(file.open(encoding="utf-8")))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["dev_f1"]))
    last = rows[-1]
    return {
        "epochs": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_dev_f1": float(best["dev_f1"]),
        "last_dev_f1": float(last["dev_f1"]),
        "last_epoch": int(float(last["epoch"])),
    }


def iter_runs():
    checkpoints = ROOT / "checkpoints"
    if not checkpoints.exists():
        return
    for path in sorted(checkpoints.iterdir()):
        if not path.is_dir() or not (path / "config.json").exists():
            continue
        config = _config(path)
        if not config:
            continue
        yield path, config


def seed_rows(dataset: Path, split: str, evaluation_mode: str, family: str = "base") -> list[dict]:
    grouped: dict[int, dict] = {}
    for path, config in iter_runs():
        if _family(config) != family:
            continue
        if family == "base" and not (config.get("learning_rate") == 2e-5 and config.get("batch_size") == 8):
            continue
        seed = int(config.get("seed", -1))
        trained = config.get("annotation_mode")
        report = load_saved_report(path, split, dataset, evaluation_mode)
        bucket = grouped.setdefault(seed, {"seed": seed, "flat": None, "nested": None})
        bucket[trained] = {
            "name": path.name,
            "report": report,
            "history": _history_best(path),
        }
    rows = []
    for seed, pair in sorted(grouped.items()):
        flat = pair.get("flat") or {}
        nested = pair.get("nested") or {}
        flat_report = flat.get("report")
        nested_report = nested.get("report")
        row = {
            "seed": seed,
            "flat_f1": None if not flat_report else flat_report["strict"]["f1"],
            "nested_f1": None if not nested_report else nested_report["strict"]["f1"],
            "flat_history": flat.get("history"),
            "nested_history": nested.get("history"),
        }
        if flat_report and nested_report and flat_report.get("documents") and nested_report.get("documents"):
            row["paired"] = paired_bootstrap(nested_report["documents"], flat_report["documents"])
        rows.append(row)
    return rows


def mean_std(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.array(values, dtype=float)
    return {"n": int(array.size), "mean": float(array.mean()), "std": float(array.std(ddof=1) if array.size > 1 else 0.0)}


def tiny_sweep(dataset: Path, split: str, evaluation_mode: str) -> list[dict]:
    rows = []
    for path, config in iter_runs():
        if _family(config) != "tiny2":
            continue
        if config.get("batch_size") != 16:
            continue
        report = load_saved_report(path, split, dataset, evaluation_mode)
        history = _history_best(path)
        rows.append({
            "name": (path / "display_name.txt").read_text(encoding="utf-8").strip() if (path / "display_name.txt").exists() else path.name,
            "lr": config.get("learning_rate"),
            "mode": config.get("annotation_mode"),
            "seed": config.get("seed"),
            "history": history,
            "test_f1": None if not report else report["strict"]["f1"],
            "test_macro": None if not report else report["strict"]["macro_f1"],
        })
    rows.sort(key=lambda row: (row["mode"] or "", row["lr"] or 0))
    return rows


def efficiency_pair(dataset: Path, split: str) -> list[dict]:
    """One flat-eval measurement per architecture, preferring the seed-42 Base and the tuned Tiny2."""
    found = []
    for path, config in iter_runs():
        report = load_saved_report(path, split, dataset, "flat")
        if not report or "efficiency" not in report:
            continue
        found.append({
            "name": report.get("display_name") or path.name,
            "family": _family(config),
            "parameters": report["efficiency"]["parameters"],
            "inference_seconds": report["efficiency"]["inference_seconds"],
            "peak_memory_bytes": report["efficiency"]["peak_memory_bytes"],
            "documents": report["efficiency"]["documents"],
            "f1": report["strict"]["f1"],
        })
    return found
