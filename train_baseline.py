"""Train a multi-level BIO tagger so nested NEREL mentions stay in the data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup

from data_prep import load_split
from levels import LEVEL_NAMES, MODES, NUM_LEVELS, layers_for_mode
from modeling import MultiLevelBertForTokenClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune multi-level ruBERT NER on NEREL.")
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--root", default="data/NEREL/NEREL-v1.1")
    parser.add_argument("--output", default="checkpoints/rubert-tiny2-levels")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=("ce", "weighted"), default="ce")
    parser.add_argument("--mode", choices=MODES, default="nested")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def token_alignment(text: str, entities, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"] if pair[0] != pair[1]]
    records = []
    aligned = []
    ignored = set()
    unaligned = 0
    for start, end, entity_type in entities:
        indices = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
        if not indices:
            unaligned += 1
            continue
        entity_offsets = offsets[indices[0]:indices[-1] + 1]
        if entity_offsets[0][0] != start or entity_offsets[-1][1] != end:
            unaligned += 1
            ignored.update(entity_offsets)
            continue
        records.append({"entity": (start, end, entity_type), "offsets": tuple(entity_offsets)})
        aligned.append((start, end, entity_type))
    return records, aligned, ignored, unaligned


def prepare_split(docs, tokenizer, split_name: str, mode: str = "nested"):
    if mode not in MODES:
        raise ValueError(f"Unknown annotation mode: {mode}")
    prepared = []
    totals = Counter()
    totals["annotation_mode"] = mode
    for doc in docs:
        levels, level_stats = layers_for_mode(doc["ents"], mode)
        records = []
        aligned_levels = []
        ignored_by_level = []
        for level, entities in enumerate(levels):
            level_records, aligned, ignored, unaligned = token_alignment(doc["text"], entities, tokenizer)
            for record in level_records:
                record["level"] = level
            records.extend(level_records)
            aligned_levels.append(aligned)
            ignored_by_level.append(ignored)
            totals[f"level_{level}_aligned"] += len(aligned)
            totals["token_unaligned"] += unaligned
        item = dict(doc)
        item["entity_records"] = records
        item["gold_by_level"] = aligned_levels
        item["gold_entities"] = [entity for level in aligned_levels for entity in level]
        item["dropped_within_level"] = [
            entity for entity in doc["ents"]
            if entity not in {kept for level in levels for kept in level}
        ]
        item["ignored_by_level"] = ignored_by_level
        item["ignored_offsets"] = set().union(*ignored_by_level)
        item["level_stats"] = level_stats
        item["annotation_mode"] = mode
        prepared.append(item)
        totals["continuous_input"] += len(doc["ents"])
        totals["selected_entities"] += level_stats["selected_entities"]
        totals["within_level_removed"] += level_stats["within_level_removed"]
        totals["token_aligned"] += len(item["gold_entities"])
        totals["discontinuous"] += doc.get("discontinuous_count", 0)
    print(f"\n[{split_name}] PREPROCESSING")
    for key, value in totals.items():
        print(f"  {key}: {value}")
    return prepared, dict(totals)


def normalize_windows(values):
    if not values:
        return values
    first = values[0]
    if isinstance(first, (int, float)):
        return [values]
    if isinstance(first, (list, tuple)) and len(first) == 2 and all(isinstance(value, (int, float)) for value in first):
        return [values]
    return values


def _label_row(attention, offsets, label2id):
    return [
        label2id["O"] if attention[index] and pair[0] != pair[1] else -100
        for index, pair in enumerate(offsets)
    ]


def encode_train_docs(docs, tokenizer, label2id, max_length: int, stride: int):
    all_input_ids, all_attention, all_labels = [], [], []
    covered = missed = 0
    for doc in docs:
        encoded = tokenizer(
            doc["text"], truncation=True, max_length=max_length, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )
        windows = zip(
            normalize_windows(encoded["input_ids"]),
            normalize_windows(encoded["attention_mask"]),
            normalize_windows(encoded["offset_mapping"]),
        )
        seen = [False] * len(doc["entity_records"])
        for input_ids, attention, window_offsets in windows:
            offsets = [tuple(map(int, pair)) for pair in window_offsets]
            position_by_offset = {
                pair: index for index, pair in enumerate(offsets)
                if pair[0] != pair[1] and attention[index]
            }
            level_rows = []
            for level in range(NUM_LEVELS):
                labels = _label_row(attention, offsets, label2id)
                for offset in doc["ignored_by_level"][level]:
                    position = position_by_offset.get(offset)
                    if position is not None:
                        labels[position] = -100
                level_rows.append(labels)
            for entity_index, record in enumerate(doc["entity_records"]):
                positions = [position_by_offset.get(offset) for offset in record["offsets"]]
                present = [position for position in positions if position is not None]
                if not present:
                    continue
                row = level_rows[record["level"]]
                if len(present) == len(record["offsets"]):
                    seen[entity_index] = True
                    entity_type = record["entity"][2]
                    for token_index, position in enumerate(present):
                        prefix = "B-" if token_index == 0 else "I-"
                        row[position] = label2id[prefix + entity_type]
                else:
                    for position in present:
                        row[position] = -100
            if any(label != -100 for row in level_rows for label in row):
                all_input_ids.append(input_ids)
                all_attention.append(attention)
                all_labels.append(level_rows)
        covered += sum(seen)
        missed += len(seen) - sum(seen)
    if not all_input_ids:
        raise RuntimeError("No training windows were generated.")
    print("\n[TRAIN] windows:", len(all_input_ids))
    print("[TRAIN] entities fully present in >=1 window:", covered)
    print("[TRAIN] entities not fully present in any window:", missed)
    return TensorDataset(
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_attention, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
    )


def decode_bio(token_predictions, id2label):
    entities = []
    active = None

    def close_active():
        nonlocal active
        if active is not None:
            entities.append((active["start"], active["end"], active["type"]))
            active = None

    for start, end, label_id in token_predictions:
        label = id2label[int(label_id)]
        if label == "O":
            close_active()
            continue
        prefix, separator, entity_type = label.partition("-")
        if not separator or prefix not in ("B", "I"):
            close_active()
            continue
        if prefix == "B" or active is None or active["type"] != entity_type:
            close_active()
            active = {"start": start, "end": end, "type": entity_type}
        else:
            active["end"] = max(active["end"], end)
    close_active()
    return entities


def merge_window_predictions(offset_windows, attention_rows, logit_rows, ignored_offsets, outside_id):
    sums = {}
    counts = Counter()
    ignored_offsets = ignored_offsets or set()
    for window_index, window in enumerate(offset_windows):
        for token_index, (start, end) in enumerate(window):
            start, end = int(start), int(end)
            if start == end or not attention_rows[window_index][token_index]:
                continue
            key = (start, end)
            logit = logit_rows[window_index][token_index]
            logit = logit.float() if torch.is_tensor(logit) else torch.tensor(logit, dtype=torch.float32)
            sums[key] = logit.clone() if key not in sums else sums[key] + logit
            counts[key] += 1
    predictions = []
    for key, total in sorted(sums.items()):
        start, end = key
        label_id = outside_id if key in ignored_offsets else int((total / counts[key]).argmax())
        predictions.append((start, end, label_id))
    return predictions


def predict_document(model, tokenizer, text, id2label, device, max_length, stride, ignored_by_level=None, return_levels=False, mode="nested"):
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length", return_tensors="pt",
    )
    offsets = encoded["offset_mapping"].tolist()
    attention = encoded["attention_mask"].tolist()
    model_inputs = {key: value.to(device) for key, value in encoded.items() if key in tokenizer.model_input_names}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(**model_inputs)["logits"].float().cpu()
    outside_id = next(index for index, label in id2label.items() if label == "O")
    num_levels = logits.shape[1]
    ignored_by_level = ignored_by_level or [set() for _ in range(num_levels)]
    by_level = []
    combined = []
    seen = set()
    active = (0,) if mode == "flat" else range(num_levels)
    for level in range(num_levels):
        level_logits = logits[:, level]
        decoded = decode_bio(merge_window_predictions(
            offsets, attention, level_logits, ignored_by_level[level], outside_id
        ), id2label)
        by_level.append(decoded)
        if level not in active:
            continue
        for entity in decoded:
            if entity not in seen:
                seen.add(entity)
                combined.append(entity)
    if return_levels:
        return combined, by_level
    return combined


def f1_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": score, "tp": tp, "fp": fp, "fn": fn}


def calculate_metrics(gold_set, pred_set, entity_types):
    overall = f1_counts(
        len(gold_set & pred_set),
        len(pred_set - gold_set),
        len(gold_set - pred_set),
    )
    per_class = {}
    for entity_type in entity_types:
        class_gold = {item for item in gold_set if item[3] == entity_type}
        class_pred = {item for item in pred_set if item[3] == entity_type}
        per_class[entity_type] = f1_counts(
            len(class_gold & class_pred),
            len(class_pred - class_gold),
            len(class_gold - class_pred),
        )
    overall["gold_entities"] = len(gold_set)
    overall["predicted_entities"] = len(pred_set)
    overall["macro_f1"] = (
        sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
    )
    overall["per_class"] = per_class
    return overall


def evaluate(model, docs, tokenizer, id2label, entity_types, device, max_length, stride, name):
    model.eval()
    gold_set, pred_set = set(), set()
    level_gold = [set() for _ in range(NUM_LEVELS)]
    level_pred = [set() for _ in range(NUM_LEVELS)]
    started = time.perf_counter()
    mode = docs[0].get("annotation_mode", "nested") if docs else "nested"
    active_levels = (0,) if mode == "flat" else range(NUM_LEVELS)
    for doc in docs:
        doc_id = doc["id"]
        predicted, by_level = predict_document(
            model, tokenizer, doc["text"], id2label, device, max_length, stride,
            doc["ignored_by_level"], return_levels=True, mode=mode,
        )
        for start, end, entity_type in doc["gold_entities"]:
            gold_set.add((doc_id, start, end, entity_type))
        for start, end, entity_type in predicted:
            pred_set.add((doc_id, start, end, entity_type))
        for level in active_levels:
            for start, end, entity_type in doc["gold_by_level"][level]:
                level_gold[level].add((doc_id, start, end, entity_type))
            for start, end, entity_type in by_level[level]:
                level_pred[level].add((doc_id, start, end, entity_type))
    torch.cuda.synchronize()
    metrics = calculate_metrics(gold_set, pred_set, entity_types)
    metrics["evaluation_seconds"] = time.perf_counter() - started
    metrics["per_level"] = {
        LEVEL_NAMES[level]: f1_counts(
            len(level_gold[level] & level_pred[level]),
            len(level_pred[level] - level_gold[level]),
            len(level_gold[level] - level_pred[level]),
        )
        for level in active_levels
    }
    print(
        f"{name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} macro-F1={metrics['macro_f1']:.4f}"
    )
    for level_name, row in metrics["per_level"].items():
        print(f"  {level_name}: F1={row['f1']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}")
    return metrics


def class_weights_from(dataset, num_labels, device):
    counts = torch.zeros(num_labels, dtype=torch.float64)
    labels = dataset.tensors[2]
    observed = labels[labels != -100]
    counts += torch.bincount(observed.reshape(-1), minlength=num_labels).double()
    present = counts > 0
    weights = torch.ones(num_labels, dtype=torch.float32)
    raw = torch.zeros_like(weights)
    raw[present] = (counts[present].sum() / (present.sum() * counts[present])).float()
    raw = raw / raw[present].mean()
    weights[present] = raw[present].clamp(0.2, 10.0)
    return weights.to(device)


def build_model(model_name, id2label, label2id):
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = len(id2label)
    config.id2label = {str(index): label for index, label in id2label.items()}
    config.label2id = label2id
    config.num_levels = NUM_LEVELS
    model = MultiLevelBertForTokenClassification(config)
    encoder = BertModel.from_pretrained(model_name)
    model.bert.load_state_dict(encoder.state_dict(), strict=False)
    return model


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    set_seed(args.seed)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Levels:", NUM_LEVELS, LEVEL_NAMES)
    print("Annotation mode:", args.mode)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    root = Path(args.root)
    splits = {}
    prepared = {}
    for split in ("train", "dev", "test"):
        prepared[split], splits[split] = prepare_split(
            load_split(root, split), tokenizer, split.upper(), args.mode
        )
    entity_types = sorted({entity_type for doc in prepared["train"] for _s, _e, entity_type in doc["gold_entities"]})
    labels = ["O"] + [f"{prefix}-{entity_type}" for entity_type in entity_types for prefix in ("B", "I")]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    train_dataset = encode_train_docs(prepared["train"], tokenizer, label2id, args.max_length, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(args.model, id2label, label2id).to(device)
    weights = class_weights_from(train_dataset, len(labels), device) if args.loss == "weighted" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_config.json", {
        "model": args.model,
        "dataset": str(root),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
        "loss": args.loss,
        "num_levels": NUM_LEVELS,
        "level_names": list(LEVEL_NAMES),
        "annotation_mode": args.mode,
        "task": (
            "flat BIO on the outer layer; nested mentions are not supervised"
            if args.mode == "flat"
            else "multi-level BIO NER; discontinuous mentions are excluded"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "hip": torch.version.hip,
    })
    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_macro = -1.0
    best_macro_epoch = 0
    optimizer_steps = skipped = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = steps = 0
        for input_ids, attention_mask, labels_batch in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            if weights is None:
                loss = nn_cross_entropy(logits, labels_batch)
            else:
                loss = nn_cross_entropy(logits, labels_batch, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                optimizer_steps += 1
            else:
                skipped += 1
            loss_sum += float(loss.item())
            steps += 1
        torch.cuda.synchronize()
        dev_metrics = evaluate(
            model, prepared["dev"], tokenizer, id2label, entity_types,
            device, args.max_length, args.stride, f"Epoch {epoch} DEV",
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(steps, 1),
            "dev_precision": dev_metrics["precision"],
            "dev_recall": dev_metrics["recall"],
            "dev_f1": dev_metrics["f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs}: loss={row['train_loss']:.5f} dev_F1={row['dev_f1']:.4f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            write_json(output / "best_dev_metrics.json", dev_metrics)
            print(f"Saved best checkpoint at epoch {epoch} (dev F1={best_f1:.4f})")
        if dev_metrics["macro_f1"] > best_macro:
            best_macro = dev_metrics["macro_f1"]
            best_macro_epoch = epoch
            write_json(output / "best_macro_dev.json", {
                "epoch": epoch,
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_micro_f1": dev_metrics["f1"],
            })
    print(f"\nBest epoch: {best_epoch}; best DEV F1: {best_f1:.4f}")
    print(f"Best macro epoch: {best_macro_epoch}; best DEV macro F1: {best_macro:.4f}")
    print(f"Optimizer updates: {optimizer_steps}; skipped due to AMP overflow: {skipped}")
    if args.skip_test:
        print("\nTEST EVALUATION SKIPPED")
        print("Checkpoint:", output.resolve())
        return
    model = MultiLevelBertForTokenClassification.from_pretrained(output).to(device)
    test_metrics = evaluate(
        model, prepared["test"], tokenizer, id2label, entity_types,
        device, args.max_length, args.stride, "FINAL TEST",
    )
    write_json(output / "test_metrics.json", test_metrics)
    print("Checkpoint:", output.resolve())


def nn_cross_entropy(logits, labels, weights=None):
    losses = []
    for level in range(logits.size(1)):
        losses.append(torch.nn.functional.cross_entropy(
            logits[:, level].float().reshape(-1, logits.size(-1)),
            labels[:, level].reshape(-1),
            weight=weights,
            ignore_index=-100,
        ))
    return torch.stack(losses).mean()


if __name__ == "__main__":
    main()
