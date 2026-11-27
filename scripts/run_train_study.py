"""Train the Tiny2 learning-rate sweep and extra Base seeds.

Each run uses the same entry point as the training page. Test is not read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOCK = ROOT / "checkpoints" / ".training.json"
DATASET = str(ROOT / "data" / "NEREL" / "NEREL-v1.1")


def jobs():
    tiny = []
    for mode in ("flat", "nested"):
        for lr in (3e-5, 5e-5, 1e-4):
            label = f"{lr:.0e}".replace("+0", "").replace("+", "")
            mode_ru = "плоский" if mode == "flat" else "вложенный"
            tiny.append({
                "name": f"ruBERT-tiny2 {mode_ru}, lr {label}, batch 16",
                "folder": f"tiny-{mode}-lr{label}-b16",
                "model": "cointegrated/rubert-tiny2",
                "lr": lr,
                "batch": 16,
                "epochs": 18,
                "mode": mode,
                "seed": 42,
            })
    base = []
    for seed in (43, 44, 45, 46):
        for mode in ("flat", "nested"):
            mode_ru = "плоский" if mode == "flat" else "вложенный"
            base.append({
                "name": f"ruBERT-base {mode_ru}, seed {seed}",
                "folder": f"base-{mode}-seed{seed}",
                "model": "DeepPavlov/rubert-base-cased",
                "lr": 2e-5,
                "batch": 8,
                "epochs": 12,
                "mode": mode,
                "seed": seed,
            })
    return tiny + base


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    for job in jobs():
        output = ROOT / "checkpoints" / job["folder"]
        if (output / "model.safetensors").exists() and (output / "training_history.csv").exists():
            history = (output / "training_history.csv").read_text(encoding="utf-8").strip().splitlines()
            if len(history) - 1 >= job["epochs"]:
                print("SKIP", job["folder"], flush=True)
                continue
        output.mkdir(parents=True, exist_ok=True)
        (output / "display_name.txt").write_text(job["name"] + "\n", encoding="utf-8")
        log_path = ROOT / "logs" / f"{job['folder']}.log"
        command = [
            PYTHON, str(ROOT / "train_baseline.py"),
            "--model", job["model"],
            "--root", DATASET,
            "--output", str(output),
            "--epochs", str(job["epochs"]),
            "--lr", str(job["lr"]),
            "--batch-size", str(job["batch"]),
            "--max-length", "256",
            "--stride", "64",
            "--seed", str(job["seed"]),
            "--loss", "ce",
            "--mode", job["mode"],
            "--skip-test",
        ]
        LOCK.write_text(json.dumps({
            "pid": 0,
            "name": job["name"],
            "output": str(output),
            "log": str(log_path),
            "epochs": job["epochs"],
            "model": job["model"],
            "dataset": DATASET,
            "mode": job["mode"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("START", job["folder"], flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        LOCK.write_text(json.dumps({
            "pid": process.pid,
            "name": job["name"],
            "output": str(output),
            "log": str(log_path),
            "epochs": job["epochs"],
            "model": job["model"],
            "dataset": DATASET,
            "mode": job["mode"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        code = process.wait()
        print("DONE", job["folder"], "exit", code, flush=True)
        if code != 0:
            print("FAILED", job["folder"], flush=True)
    if LOCK.exists():
        LOCK.unlink()
    print("STUDY TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
