"""Assign nested NEREL mentions to BIO layers.

Level 0 holds entities that contain others. Level 1 holds mentions inside
those, and level 2 holds anything nested more deeply. Discontinuous mentions
stay out: a BIO layer still needs one continuous span. Within one level,
overlapping spans are resolved by the same deterministic rule as the flat
project, because one layer cannot store two labels on one token.
"""

from __future__ import annotations

from collections import Counter

from data_prep import Entity, outer_entities

NUM_LEVELS = 3
LEVEL_NAMES = ("lvl_0", "lvl_1", "lvl_2")
MODES = ("nested", "flat")
MODE_LABELS = {"nested": "вложенный", "flat": "плоский"}


def nesting_depth(entity: Entity, entities: list[Entity]) -> int:
    start, end, _entity_type = entity
    containers = 0
    for other_start, other_end, _other_type in entities:
        if other_start <= start and end <= other_end and (other_end - other_start) > (end - start):
            containers += 1
    return containers


def assign_levels(entities: list[Entity]) -> tuple[list[list[Entity]], dict[str, int]]:
    buckets: list[list[Entity]] = [[] for _ in range(NUM_LEVELS)]
    for entity in entities:
        level = min(nesting_depth(entity, entities), NUM_LEVELS - 1)
        buckets[level].append(entity)

    kept: list[list[Entity]] = []
    stats: Counter = Counter()
    stats["input_entities"] = len(entities)
    for level, bucket in enumerate(buckets):
        selected, level_stats = outer_entities(bucket, return_stats=True)
        kept.append(selected)
        stats[f"level_{level}_input"] = len(bucket)
        stats[f"level_{level}_kept"] = len(selected)
        stats["within_level_removed"] += len(bucket) - len(selected)
        stats["selected_entities"] += len(selected)
        for key, value in level_stats.items():
            stats[f"level_{level}_{key}"] += value
    return kept, dict(stats)


def layers_for_mode(entities: list[Entity], mode: str) -> tuple[list[list[Entity]], dict[str, int]]:
    """Build BIO layers for one document.

    ``nested`` keeps a mention on the layer equal to how many mentions contain it.
    ``flat`` keeps only the outer non-overlapping spans on layer 0. The other
    layers stay empty, so the same three-head model can be trained either way.
    """
    if mode == "nested":
        return assign_levels(entities)
    if mode != "flat":
        raise ValueError(f"Unknown annotation mode: {mode}")
    selected, raw = outer_entities(entities, return_stats=True)
    stats = {
        "input_entities": len(entities),
        "selected_entities": len(selected),
        "within_level_removed": len(entities) - len(selected),
        "nested_removed": raw.get("nested_removed", 0),
        "partial_removed": raw.get("partial_or_conflict_removed", 0),
        "level_0_input": len(entities),
        "level_0_kept": len(selected),
        "level_1_kept": 0,
        "level_2_kept": 0,
    }
    return [selected, [], []], stats
