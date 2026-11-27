from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

Entity = tuple[int, int, str]


def parse_ann(path: Path, stats: Counter | None = None) -> list[Entity]:
    """Read continuous text-bound entities from a BRAT .ann file.

    Discontinuous entities (multiple offset segments separated by ';') are
    counted and skipped because the baseline uses a single-span BIO format.
    """
    entities: list[Entity] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("T"):
            continue

        parts = line.split("\t", 2)
        if len(parts) < 3:
            if stats is not None:
                stats["malformed"] += 1
            continue

        fields = parts[1].split()
        if len(fields) < 3:
            if stats is not None:
                stats["malformed"] += 1
            continue

        entity_type = fields[0]
        span_text = " ".join(fields[1:])

        if ";" in span_text:
            if stats is not None:
                stats["discontinuous"] += 1
            continue

        values = span_text.split()
        if len(values) != 2:
            if stats is not None:
                stats["malformed"] += 1
            continue

        try:
            start, end = map(int, values)
        except ValueError:
            if stats is not None:
                stats["malformed"] += 1
            continue

        if start < 0 or end <= start:
            if stats is not None:
                stats["malformed"] += 1
            continue

        entities.append((start, end, entity_type))

    return entities


def _prefer_schedule(
    left: list[int],
    right: list[int],
    intervals: list[Entity],
) -> list[int]:
    """Choose the better interval schedule deterministically.

    Priority: (1) retain the largest number of entities, (2) retain the
    greatest total character coverage, (3) choose lexicographically smaller
    spans for deterministic tie-breaking.
    """
    left_score = (
        len(left),
        sum(intervals[i][1] - intervals[i][0] for i in left),
    )
    right_score = (
        len(right),
        sum(intervals[i][1] - intervals[i][0] for i in right),
    )

    if left_score != right_score:
        return left if left_score > right_score else right

    def key(indices: list[int]):
        return tuple(sorted(intervals[i] for i in indices))

    return left if key(left) <= key(right) else right


def outer_entities(
    entities: Iterable[Entity],
    *,
    return_stats: bool = False,
) -> list[Entity] | tuple[list[Entity], dict[str, int]]:
    """Resolve spans that share one BIO layer.

    Nested mentions are assigned to different layers before this function runs.
    Inside one layer the same rules remain: drop duplicate and conflicting
    spans, then keep a non-overlapping subset.
    """
    stats = Counter()
    grouped: dict[tuple[int, int], set[str]] = defaultdict(set)
    original = list(entities)

    for start, end, entity_type in original:
        grouped[(start, end)].add(entity_type)

    unique: list[Entity] = []
    for (start, end), types in grouped.items():
        if len(types) > 1:
            stats["exact_conflict_groups"] += 1
            stats["exact_conflict_entities"] += len(types)
            continue
        entity_type = next(iter(types))
        unique.append((start, end, entity_type))

    # Count duplicate rows with identical boundaries and identical type.
    stats["duplicate_rows"] = len(original) - sum(len(v) for v in grouped.values())

    # Remove every entity strictly contained in another entity.
    maximal: list[Entity] = []
    for entity in unique:
        start, end, _ = entity
        contained = any(
            other_start <= start
            and end <= other_end
            and (other_end - other_start) > (end - start)
            for other_start, other_end, _ in unique
        )
        if not contained:
            maximal.append(entity)

    stats["nested_removed"] = len(unique) - len(maximal)

    # Select a maximum-cardinality set of non-overlapping intervals.
    ordered = sorted(maximal, key=lambda x: (x[1], x[0], x[2]))
    ends = [entity[1] for entity in ordered]
    dp: list[list[int]] = [[]]

    for i, (start, _end, _entity_type) in enumerate(ordered):
        previous = bisect_right(ends, start, 0, i) - 1
        include = dp[previous + 1] + [i]
        exclude = dp[i]
        dp.append(_prefer_schedule(include, exclude, ordered))

    selected = [ordered[i] for i in dp[-1]]
    selected.sort(key=lambda x: (x[0], x[1], x[2]))

    stats["partial_or_conflict_removed"] = len(maximal) - len(selected)
    stats["input_entities"] = len(original)
    stats["selected_entities"] = len(selected)

    if return_stats:
        return selected, dict(stats)
    return selected


def load_split(root: str | Path, split: str) -> list[dict]:
    """Return documents with keys: id, text, ents, discontinuous_count."""
    directory = Path(root) / split
    ann_files = sorted(directory.glob("*.ann"))
    if not ann_files:
        raise FileNotFoundError(f"No .ann files found in {directory}")

    docs: list[dict] = []
    for ann_path in ann_files:
        txt_path = ann_path.with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing paired text file: {txt_path}")

        parse_stats = Counter()
        entities = parse_ann(ann_path, parse_stats)
        docs.append({
            "id": ann_path.stem,
            "text": txt_path.read_text(encoding="utf-8"),
            "ents": entities,
            "discontinuous_count": parse_stats["discontinuous"],
            "malformed_count": parse_stats["malformed"],
        })

    return docs


def check_offsets(docs: list[dict]) -> int:
    """Count invalid or empty spans; text-to-annotation equality is checked by audit_data.py."""
    bad = 0
    for doc in docs:
        text = doc["text"]
        for start, end, _entity_type in doc["ents"]:
            if start < 0 or end > len(text) or start >= end or not text[start:end].strip():
                bad += 1
    return bad


if __name__ == "__main__":
    root = Path("data/NEREL/NEREL-v1.1")
    for split in ("train", "dev", "test"):
        docs = load_split(root, split)
        raw_count = sum(len(doc["ents"]) for doc in docs)
        flat_count = 0
        flatten_stats = Counter()
        for doc in docs:
            flat, stats = outer_entities(doc["ents"], return_stats=True)
            flat_count += len(flat)
            flatten_stats.update(stats)

        avg_len = sum(len(doc["text"]) for doc in docs) / len(docs)
        discontinuous = sum(doc["discontinuous_count"] for doc in docs)
        print(
            f"{split:5s} docs={len(docs):4d} "
            f"continuous={raw_count:6d} flat={flat_count:6d} "
            f"discontinuous={discontinuous:3d} avg_chars={avg_len:.0f} "
            f"bad_offsets={check_offsets(docs)}"
        )
        print("       flattening:", dict(flatten_stats))
