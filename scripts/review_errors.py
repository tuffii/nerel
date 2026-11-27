"""Dump EVENT and selected confusion examples for the metrics page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from data_prep import load_split
from modeling import MultiLevelBertForTokenClassification
from train_baseline import predict_document, prepare_split

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "ruBERT-base-вложенный"
DATASET = ROOT / "data" / "NEREL" / "NEREL-v1.1"
OUT = ROOT / "results" / "error_review.json"


def window(text: str, start: int, end: int) -> str:
    left = max(0, start - 70)
    right = min(len(text), end + 70)
    return text[left:right].replace("\n", " ")


def main():
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    model = MultiLevelBertForTokenClassification.from_pretrained(CHECKPOINT).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    docs, _prep = prepare_split(load_split(DATASET, "test"), tokenizer, "TEST", "nested")
    examples = []
    for doc in docs:
        predicted, _levels = predict_document(
            model, tokenizer, doc["text"], id2label, device, 256, 64,
            doc["ignored_by_level"], return_levels=True, mode="nested",
        )
        text = doc["text"]
        for start, end, entity_type in doc["gold_entities"]:
            if entity_type != "EVENT":
                continue
            exact = [item for item in predicted if item[0] == start and item[1] == end]
            overlap = [
                item for item in predicted
                if item[0] < end and item[1] > start and not (item[0] == start and item[1] == end)
            ]
            if any(item[2] == "EVENT" for item in exact):
                continue
            if exact:
                kind = "те же границы, другой тип"
                pred = exact[0]
            elif overlap:
                kind = "пересечение"
                pred = overlap[0]
            else:
                kind = "не найдено"
                pred = None
            examples.append({
                "группа": "EVENT",
                "документ": doc["id"],
                "эталон": text[start:end],
                "модель": "" if pred is None else f"{pred[2]}: {text[pred[0]:pred[1]]}",
                "что случилось": kind,
                "контекст": window(text, start, end),
            })
        for start, end, entity_type in predicted:
            overlaps = [
                gold for gold in doc["gold_entities"]
                if gold[0] < end and gold[1] > start
            ]
            for gold_start, gold_end, gold_type in overlaps:
                pair = None
                if gold_type == "COUNTRY" and entity_type == "NATIONALITY":
                    pair = "COUNTRY → NATIONALITY"
                elif gold_type == "FACILITY" and entity_type == "ORGANIZATION":
                    pair = "FACILITY → ORGANIZATION"
                if pair is None:
                    continue
                examples.append({
                    "группа": pair,
                    "документ": doc["id"],
                    "эталон": text[gold_start:gold_end],
                    "модель": f"{entity_type}: {text[start:end]}",
                    "что случилось": "тот же span" if (gold_start, gold_end) == (start, end) else "пересечение",
                    "контекст": window(text, min(gold_start, start), max(gold_end, end)),
                })
    event = [row for row in examples if row["группа"] == "EVENT"][:25]
    country = [row for row in examples if row["группа"] == "COUNTRY → NATIONALITY"][:15]
    facility = [row for row in examples if row["группа"] == "FACILITY → ORGANIZATION"][:15]
    chosen = event + country + facility
    payload = {
        "model": str(CHECKPOINT),
        "summary": "",
        "examples": chosen,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("examples", len(chosen), "event", len(event), "country", len(country), "facility", len(facility))


if __name__ == "__main__":
    main()
