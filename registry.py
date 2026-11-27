"""Discover fine-tuned models and keep a single training job at a time."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
LOCK = CHECKPOINTS / ".training.json"
UNTRAINED = "ruBERT-base без дообучения"
BASE_MODEL = "DeepPavlov/rubert-base-cased"
PRESETS = {
    "ruBERT-tiny2": "cointegrated/rubert-tiny2",
    "ruBERT-base": "DeepPavlov/rubert-base-cased",
}


def available_datasets(data_root: Path | None = None) -> list[tuple[str, Path]]:
    """Directories under data/ that contain train, dev and test."""
    data_root = data_root or DATA
    if not data_root.exists():
        return []
    found = []
    for current, dirnames, _filenames in os.walk(data_root, followlinks=True):
        path = Path(current)
        if not all((path / split).is_dir() for split in ("train", "dev", "test")):
            continue
        if not any((path / "train").glob("*.ann")):
            continue
        dirnames.clear()
        found.append((str(path.relative_to(data_root)), path))
    found.sort(key=lambda item: (not str(item[0]).endswith("NEREL-v1.1"), item[0]))
    return found


def ready_checkpoints() -> list[Path]:
    if not CHECKPOINTS.exists():
        return []
    found = []
    for path in sorted(CHECKPOINTS.iterdir()):
        if path.is_dir() and (path / "config.json").exists() and (path / "model.safetensors").exists():
            found.append(path)
    return found


def display_name(path: Path) -> str:
    name_file = path / "display_name.txt"
    if name_file.exists():
        text = name_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return path.name


def set_display_name(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "display_name.txt").write_text(name.strip() + "\n", encoding="utf-8")


def named_models() -> list[tuple[str, Path]]:
    """Unique labels for the UI. A repeated name keeps the folder in parentheses."""
    used: dict[str, int] = {}
    items = []
    for path in ready_checkpoints():
        label = display_name(path)
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label} ({path.name})"
        items.append((label, path))
    return items


def label_source() -> Path | None:
    for path in ready_checkpoints():
        if "weighted" in path.name or "base" in path.name:
            return path
    found = ready_checkpoints()
    return found[0] if found else None


def slug(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "-", name.strip())
    cleaned = cleaned.strip("-._")
    return (cleaned or "run")[:80]


def output_dir_for(name: str) -> Path:
    base = slug(name)
    candidate = CHECKPOINTS / base
    index = 2
    while candidate.exists():
        candidate = CHECKPOINTS / f"{base}-{index}"
        index += 1
    return candidate


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    # A finished child stays visible until the parent calls wait. Streamlit
    # does not reap the training process, so a zombie must count as stopped.
    state = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[-1].split()[0]
    if state == "Z":
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        return False
    return True


def training_status() -> dict | None:
    if not LOCK.exists():
        return None
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    data["running"] = _pid_alive(int(data.get("pid", 0)))
    return data


def start_training(
    name: str,
    model: str,
    epochs: int,
    lr: float,
    batch_size: int,
    loss: str,
    seed: int,
    dataset: str,
    mode: str = "nested",
) -> Path:
    current = training_status()
    if current and current["running"]:
        raise RuntimeError("Пока идёт обучение, запустить ещё одно невозможно: GPU занята.")
    output = output_dir_for(name)
    output.mkdir(parents=True)
    set_display_name(output, name)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{output.name}.log"
    command = [
        sys.executable,
        str(ROOT / "train_baseline.py"),
        "--model", model,
        "--output", str(output),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--max-length", "256",
        "--stride", "64",
        "--seed", str(seed),
        "--loss", loss,
        "--skip-test",
        "--root", dataset,
        "--mode", mode,
    ]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    LOCK.write_text(json.dumps({
        "pid": process.pid,
        "name": name,
        "output": str(output),
        "log": str(log_path),
        "epochs": epochs,
        "model": model,
        "dataset": dataset,
        "mode": mode,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def trained_mode(path: Path) -> str:
    """How the checkpoint was supervised. Missing configs in this project are nested."""
    config_path = path / "training_config.json"
    if not config_path.exists():
        return "nested"
    mode = json.loads(config_path.read_text(encoding="utf-8")).get("annotation_mode", "nested")
    return mode if mode in ("flat", "nested") else "nested"


def read_history(path: Path):
    history = path / "training_history.csv"
    if not history.exists():
        return None
    import pandas as pd
    return pd.read_csv(history)
