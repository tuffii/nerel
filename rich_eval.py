"""Strict and relaxed entity metrics, confusion, and document bootstrap.

The strict score is unchanged: a hit needs the same document, start, end, and type.
Relaxed scores use one-to-one matching and answer a different question.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from data_prep import load_split
from levels import LEVEL_NAMES, NUM_LEVELS
from modeling import MultiLevelBertForTokenClassification
from registry import CHECKPOINTS, DATA, display_name
from train_baseline import predict_document, prepare_split, write_json

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "results" / "reports"


def f1_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def overlap_len(gold, pred) -> int:
    return max(0, min(gold[1], pred[1]) - max(gold[0], pred[0]))


def greedy_match(gold, pred, accepts) -> list[tuple[int, int]]:
    pairs = []
    for gold_index, gold_entity in enumerate(gold):
        for pred_index, pred_entity in enumerate(pred):
            if accepts(gold_entity, pred_entity):
                pairs.append((overlap_len(gold_entity, pred_entity), -gold_index, -pred_index, gold_index, pred_index))
    pairs.sort(reverse=True)
    used_gold = set()
    used_pred = set()
    matches = []
    for _score, _gold_key, _pred_key, gold_index, pred_index in pairs:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        matches.append((gold_index, pred_index))
    return matches


def same_span_and_type(gold, pred) -> bool:
    return gold[0] == pred[0] and gold[1] == pred[1] and gold[2] == pred[2]


def overlap_and_type(gold, pred) -> bool:
    return gold[2] == pred[2] and overlap_len(gold, pred) > 0


def same_start_and_type(gold, pred) -> bool:
    return gold[0] == pred[0] and gold[2] == pred[2]


def same_end_and_type(gold, pred) -> bool:
    return gold[1] == pred[1] and gold[2] == pred[2]


def any_overlap(gold, pred) -> bool:
    return overlap_len(gold, pred) > 0


def match_counts(gold, pred, accepts) -> tuple[int, int, int]:
    matched = len(greedy_match(gold, pred, accepts))
    return matched, len(pred) - matched, len(gold) - matched


def char_bucket(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 15:
        return "6-15"
    if length <= 40:
        return "16-40"
    return "41+"


def token_bucket(count: int) -> str:
    if count <= 1:
        return "1 токен"
    if count <= 3:
        return "2-3 токена"
    return "4+ токена"


def doc_bucket(length: int) -> str:
    if length < 1000:
        return "<1k"
    if length < 3000:
        return "1k-3k"
    if length < 8000:
        return "3k-8k"
    return ">=8k"


def percentile_interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "low": float(np.percentile(array, 2.5)),
        "high": float(np.percentile(array, 97.5)),
    }


def macro_f1(per_type: dict[str, dict]) -> float:
    scores = [row["f1"] for row in per_type.values() if row["tp"] + row["fp"] + row["fn"]]
    return float(sum(scores) / len(scores)) if scores else 0.0


def weighted_f1(per_type: dict[str, dict]) -> float:
    support = sum(row["tp"] + row["fn"] for row in per_type.values())
    if not support:
        return 0.0
    return float(sum(row["f1"] * (row["tp"] + row["fn"]) for row in per_type.values()) / support)


def _add_type(store, entity_type, tp, fp, fn):
    row = store[entity_type]
    row["tp"] += tp
    row["fp"] += fp
    row["fn"] += fn


def bootstrap(documents: list[dict], types: list[str], draws: int = 1000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    strict_f1 = []
    macro = []
    relaxed_f1 = []
    for _ in range(draws):
        sample = [documents[rng.randrange(len(documents))] for _ in documents]
        tp = sum(item["strict_tp"] for item in sample)
        fp = sum(item["strict_fp"] for item in sample)
        fn = sum(item["strict_fn"] for item in sample)
        strict_f1.append(f1_counts(tp, fp, fn)["f1"])
        relaxed_f1.append(f1_counts(
            sum(item["relaxed_tp"] for item in sample),
            sum(item["relaxed_fp"] for item in sample),
            sum(item["relaxed_fn"] for item in sample),
        )["f1"])
        per_type = {entity_type: Counter() for entity_type in types}
        for item in sample:
            for entity_type, counts in item["per_type"].items():
                _add_type(per_type, entity_type, counts["tp"], counts["fp"], counts["fn"])
        scored = {}
        for entity_type, counts in per_type.items():
            scored[entity_type] = f1_counts(counts["tp"], counts["fp"], counts["fn"])
        macro.append(macro_f1(scored))
    return {
        "draws": draws,
        "strict_f1": percentile_interval(strict_f1),
        "strict_macro_f1": percentile_interval(macro),
        "relaxed_f1": percentile_interval(relaxed_f1),
    }


def paired_bootstrap(left: list[dict], right: list[dict], draws: int = 1000, seed: int = 42) -> dict:
    """Document-paired interval for F1(left) - F1(right)."""
    right_by = {row["id"]: row for row in right}
    ids = [row["id"] for row in left if row["id"] in right_by]
    left_by = {row["id"]: row for row in left}
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        left_f1 = f1_counts(
            sum(left_by[item]["strict_tp"] for item in sample),
            sum(left_by[item]["strict_fp"] for item in sample),
            sum(left_by[item]["strict_fn"] for item in sample),
        )["f1"]
        right_f1 = f1_counts(
            sum(right_by[item]["strict_tp"] for item in sample),
            sum(right_by[item]["strict_fp"] for item in sample),
            sum(right_by[item]["strict_fn"] for item in sample),
        )["f1"]
        diffs.append(left_f1 - right_f1)
    observed_left = f1_counts(
        sum(left_by[item]["strict_tp"] for item in ids),
        sum(left_by[item]["strict_fp"] for item in ids),
        sum(left_by[item]["strict_fn"] for item in ids),
    )["f1"]
    observed_right = f1_counts(
        sum(right_by[item]["strict_tp"] for item in ids),
        sum(right_by[item]["strict_fp"] for item in ids),
        sum(right_by[item]["strict_fn"] for item in ids),
    )["f1"]
    return {
        "documents": len(ids),
        "draws": draws,
        "left_f1": observed_left,
        "right_f1": observed_right,
        "observed_diff": observed_left - observed_right,
        **percentile_interval(diffs),
    }


def bucket_table(counter: dict[str, Counter]) -> list[dict]:
    rows = []
    for name, counts in counter.items():
        scores = f1_counts(counts["tp"], counts["fp"], counts["fn"])
        scores["bucket"] = name
        scores["support"] = counts["tp"] + counts["fn"]
        rows.append(scores)
    return rows


def build_report(checkpoint: Path, split: str, root: str, draws: int, mode: str = "nested") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = MultiLevelBertForTokenClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    raw = load_split(root, split)
    docs, prep = prepare_split(raw, tokenizer, split.upper(), mode)
    types = sorted({entity_type for doc in docs for _s, _e, entity_type in doc["gold_entities"]})

    strict_tp = strict_fp = strict_fn = 0
    relaxed_tp = relaxed_fp = relaxed_fn = 0
    start_tp = start_fp = start_fn = 0
    end_tp = end_fp = end_fn = 0
    per_type = {entity_type: Counter() for entity_type in types}
    confusion = Counter()
    kinds = Counter()
    char_counts = defaultdict(Counter)
    token_counts = defaultdict(Counter)
    doc_counts = defaultdict(Counter)
    documents = []
    continuous = 0
    discontinuous = 0
    recovered_continuous = 0
    dropped_annotations = 0
    fp_total = 0
    fp_on_dropped = 0
    level_gold = [set() for _ in range(NUM_LEVELS)]
    level_pred = [set() for _ in range(NUM_LEVELS)]
    torch.cuda.reset_peak_memory_stats()
    inference_seconds = 0.0

    for doc in docs:
        gold = doc["gold_entities"]
        started = time.perf_counter()
        predicted, by_level = predict_document(
            model, tokenizer, doc["text"], id2label, device, 256, 64,
            doc.get("ignored_by_level"), return_levels=True, mode=mode,
        )
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        active_levels = (0,) if mode == "flat" else range(NUM_LEVELS)
        dropped = doc.get("dropped_within_level", [])
        dropped_annotations += len(dropped)
        for level in active_levels:
            for start, end, entity_type in doc["gold_by_level"][level]:
                level_gold[level].add((doc["id"], start, end, entity_type))
            for start, end, entity_type in by_level[level]:
                level_pred[level].add((doc["id"], start, end, entity_type))
        continuous += len(doc["ents"])
        discontinuous += doc.get("discontinuous_count", 0)
        recovered_continuous += len(greedy_match(doc["ents"], predicted, overlap_and_type))

        doc_strict = match_counts(gold, predicted, same_span_and_type)
        doc_relaxed = match_counts(gold, predicted, overlap_and_type)
        doc_start = match_counts(gold, predicted, same_start_and_type)
        doc_end = match_counts(gold, predicted, same_end_and_type)
        strict_tp += doc_strict[0]
        strict_fp += doc_strict[1]
        strict_fn += doc_strict[2]
        relaxed_tp += doc_relaxed[0]
        relaxed_fp += doc_relaxed[1]
        relaxed_fn += doc_relaxed[2]
        start_tp += doc_start[0]
        start_fp += doc_start[1]
        start_fn += doc_start[2]
        end_tp += doc_end[0]
        end_fp += doc_end[1]
        end_fn += doc_end[2]

        doc_types = {}
        for entity_type in types:
            gold_type = [entity for entity in gold if entity[2] == entity_type]
            pred_type = [entity for entity in predicted if entity[2] == entity_type]
            tp, fp, fn = match_counts(gold_type, pred_type, same_span_and_type)
            _add_type(per_type, entity_type, tp, fp, fn)
            doc_types[entity_type] = {"tp": tp, "fp": fp, "fn": fn}

        token_len = {
            (record["entity"][0], record["entity"][1], record["entity"][2]): len(record["offsets"])
            for record in doc["entity_records"]
        }
        content_offsets = [
            (int(start), int(end))
            for start, end in tokenizer(
                doc["text"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            if int(start) != int(end)
        ]

        def predicted_token_count(start: int, end: int) -> int:
            return sum(1 for left, right in content_offsets if left < end and right > start)
        strict_matches = set(greedy_match(gold, predicted, same_span_and_type))
        matched_gold = {gold_index for gold_index, _pred_index in strict_matches}
        matched_pred = {pred_index for _gold_index, pred_index in strict_matches}
        for pred_index, entity in enumerate(predicted):
            if pred_index in matched_pred:
                continue
            fp_total += 1
            if any(overlap_len(entity, removed_entity) > 0 for removed_entity in dropped):
                fp_on_dropped += 1
        for index, entity in enumerate(gold):
            bucket_name = char_bucket(entity[1] - entity[0])
            char_counts[bucket_name]["fn" if index not in matched_gold else "tp"] += 1
            token_counts[token_bucket(token_len.get(tuple(entity), 1))]["fn" if index not in matched_gold else "tp"] += 1
            doc_counts[doc_bucket(len(doc["text"]))]["fn" if index not in matched_gold else "tp"] += 1
        document_bucket = doc_bucket(len(doc["text"]))
        for index, entity in enumerate(predicted):
            if index in matched_pred:
                continue
            char_counts[char_bucket(entity[1] - entity[0])]["fp"] += 1
            token_counts[token_bucket(predicted_token_count(entity[0], entity[1]))]["fp"] += 1
            doc_counts[document_bucket]["fp"] += 1

        overlap_matches = greedy_match(gold, predicted, any_overlap)
        used_pred = set()
        for gold_index, pred_index in overlap_matches:
            used_pred.add(pred_index)
            gold_type = gold[gold_index][2]
            pred_type = predicted[pred_index][2]
            confusion[(gold_type, pred_type)] += 1
            if gold[gold_index][:2] == predicted[pred_index][:2] and gold_type == pred_type:
                kinds["tp"] += 1
            elif gold_type == pred_type:
                kinds["boundary"] += 1
            elif gold[gold_index][:2] == predicted[pred_index][:2]:
                kinds["wrong_type"] += 1
            else:
                kinds["boundary_and_type"] += 1
        matched_gold_overlap = {gold_index for gold_index, _pred_index in overlap_matches}
        for gold_index, entity in enumerate(gold):
            if gold_index not in matched_gold_overlap:
                kinds["missed"] += 1
                confusion[(entity[2], "не найдено")] += 1
        for pred_index, entity in enumerate(predicted):
            if pred_index in used_pred:
                continue
            kinds["spurious"] += 1
            confusion[("лишнее", entity[2])] += 1

        documents.append({
            "id": doc["id"],
            "strict_tp": doc_strict[0],
            "strict_fp": doc_strict[1],
            "strict_fn": doc_strict[2],
            "relaxed_tp": doc_relaxed[0],
            "relaxed_fp": doc_relaxed[1],
            "relaxed_fn": doc_relaxed[2],
            "per_type": doc_types,
        })

    per_type_scores = {
        entity_type: f1_counts(counts["tp"], counts["fp"], counts["fn"])
        for entity_type, counts in per_type.items()
    }
    for entity_type, row in per_type_scores.items():
        row["support"] = row["tp"] + row["fn"]
    strict = f1_counts(strict_tp, strict_fp, strict_fn)
    strict["macro_f1"] = macro_f1(per_type_scores)
    strict["weighted_f1"] = weighted_f1(per_type_scores)
    strict["gold_entities"] = strict_tp + strict_fn
    strict["predicted_entities"] = strict_tp + strict_fp

    return {
        "checkpoint": str(checkpoint),
        "display_name": display_name(checkpoint),
        "split": split,
        "annotation_mode": mode,
        "preprocessing": prep,
        "strict": strict,
        "relaxed_overlap": f1_counts(relaxed_tp, relaxed_fp, relaxed_fn),
        "same_start": f1_counts(start_tp, start_fp, start_fn),
        "same_end": f1_counts(end_tp, end_fp, end_fn),
        "bootstrap": bootstrap(documents, types, draws=draws),
        "per_type": per_type_scores,
        "confusion": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in sorted(confusion.items())
        ],
        "error_kinds": dict(kinds),
        "nested": {
            "continuous": continuous,
            "within_level_removed": prep.get("within_level_removed", 0),
            "dropped_annotations": dropped_annotations,
            "discontinuous": discontinuous,
            "flat_gold": strict["gold_entities"],
            "relaxed_hits_on_continuous": recovered_continuous,
            "relaxed_recall_on_continuous": recovered_continuous / continuous if continuous else 0.0,
            "strict_false_positives": fp_total,
            "false_positives_overlapping_dropped_annotation": fp_on_dropped,
        },
        "per_level": {
            LEVEL_NAMES[level]: f1_counts(
                len(level_gold[level] & level_pred[level]),
                len(level_pred[level] - level_gold[level]),
                len(level_gold[level] - level_pred[level]),
            )
            for level in ((0,) if mode == "flat" else range(NUM_LEVELS))
        },
        "by_char_length": bucket_table(char_counts),
        "by_token_length": bucket_table(token_counts),
        "by_doc_length": bucket_table(doc_counts),
        "documents": [
            {
                "id": row["id"],
                "strict_tp": row["strict_tp"],
                "strict_fp": row["strict_fp"],
                "strict_fn": row["strict_fn"],
            }
            for row in documents
        ],
        "efficiency": {
            "parameters": parameter_count,
            "inference_seconds": inference_seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "documents": len(documents),
        },
    }


def dataset_slug(root: str | Path) -> str:
    """Name of the dataset directory, the same on every machine.

    Resolving the path is unsafe: a symlink to the corpus makes the resolved
    path leave ``data/``, and a normal clone stays inside it. The saved report
    names use only the directory, for example ``NEREL-v1.1``.
    """
    return Path(root).name.replace("\\", "__").replace("/", "__")


def report_path(checkpoint: Path, split: str, root: str | Path | None = None, mode: str = "nested") -> Path:
    if root is None:
        return REPORTS / f"{checkpoint.name}_{split}.json"
    return REPORTS / f"{checkpoint.name}__{dataset_slug(root)}__{mode}__{split}.json"


def load_saved_report(checkpoint: Path, split: str, root: str | Path, mode: str = "nested") -> dict | None:
    specific = report_path(checkpoint, split, root, mode)
    candidates = [specific]
    if mode == "nested":
        candidates.append(REPORTS / f"{checkpoint.name}__{dataset_slug(root)}__{split}.json")
        if dataset_slug(root).endswith("NEREL-v1.1"):
            candidates.append(REPORTS / f"{checkpoint.name}_{split}.json")
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_mode = payload.get("annotation_mode", "nested")
        if saved_mode != mode:
            continue
        saved = payload.get("dataset")
        if saved and Path(saved).resolve() != Path(root).resolve() and path != specific:
            continue
        return payload
    return None


def save_report(checkpoint: Path, split: str, root: str, draws: int, mode: str = "nested") -> Path:
    payload = build_report(checkpoint, split, root, draws, mode)
    payload["dataset"] = str(Path(root).resolve())
    path = report_path(checkpoint, split, root, mode)
    write_json(path, payload)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build a detailed flat-NER report.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--root", default="data/NEREL/NEREL-v1.1")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("nested", "flat"), default="nested")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    targets = []
    if args.all:
        targets = [
            path for path in sorted(CHECKPOINTS.iterdir())
            if path.is_dir() and (path / "model.safetensors").exists()
        ]
    elif args.checkpoint:
        targets = [Path(args.checkpoint)]
    else:
        raise SystemExit("Pass --checkpoint or --all")
    for target in targets:
        path = save_report(target, args.split, args.root, args.draws, args.mode)
        print("Saved", path)


if __name__ == "__main__":
    main()
