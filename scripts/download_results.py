"""Download checkpoints and saved metrics from the Hugging Face dataset repo.

Running the script asks in the console which models to take. Reports for the
chosen models are downloaded with them.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "tuffi/nerel-results"


def repo_index(repo: str) -> tuple[dict[str, int], list[str]]:
    sizes: dict[str, int] = {}
    reports: list[str] = []
    for item in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True):
        path = item.path
        if path.startswith("results/"):
            reports.append(path)
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "checkpoints":
            sizes[parts[1]] = sizes.get(parts[1], 0) + int(getattr(item, "size", 0) or 0)
    return sizes, reports


def presets(names: list[str]) -> list[tuple[str, str, list[str]]]:
    by_name = set(names)

    def keep(predicate) -> list[str]:
        return [name for name in names if predicate(name)]

    groups = [
        ("a", "Base, seed 42, оба режима", keep(lambda name: name.startswith("ruBERT-base"))),
        ("b", "Подобранный Tiny2, шаг 1e-4, оба режима", keep(lambda name: "lr1e-04" in name)),
        ("c", "Все Base, пять seed", keep(lambda name: "base" in name.lower() and "tiny" not in name.lower())),
        ("d", "Все Tiny2", keep(lambda name: "tiny" in name.lower())),
        ("e", "Все модели", list(names)),
    ]
    return [(key, title, found) for key, title, found in groups if found and set(found) <= by_name]


def report_patterns(names: list[str], reports: list[str]) -> list[str]:
    patterns = ["results/error_review.json"]
    for name in names:
        prefix = f"results/reports/{name}__"
        patterns.extend(path for path in reports if path.startswith(prefix))
    return patterns


def megabytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} ГБ"
    return f"{size / 1024**2:.0f} МБ"


def parse_choice(answer: str, names: list[str], groups: list[tuple[str, str, list[str]]]) -> list[str]:
    tokens = [token.strip().lower() for token in answer.replace(",", " ").split() if token.strip()]
    if not tokens:
        return []
    by_key = {key: chosen for key, _title, chosen in groups}
    selected: list[str] = []
    for token in tokens:
        if token in by_key:
            batch = by_key[token]
        elif token.isdigit() and 1 <= int(token) <= len(names):
            batch = [names[int(token) - 1]]
        else:
            raise ValueError(token)
        for name in batch:
            if name not in selected:
                selected.append(name)
    return selected


def ask(names: list[str], sizes: dict[str, int]) -> list[str]:
    groups = presets(names)
    print("Что скачать?\n")
    print("Наборы:")
    for key, title, chosen in groups:
        weight = sum(sizes[name] for name in chosen)
        print(f"  {key}. {title} ({megabytes(weight)}, моделей: {len(chosen)})")
    print("\nМодели:")
    for index, name in enumerate(names, start=1):
        print(f"  {index:2}. {name:42} {megabytes(sizes[name]):>8}")
    print("\nБуква набора, номера моделей или и то и другое. Пример: a  12  15")
    print("Пустая строка отменяет загрузку.")
    while True:
        try:
            answer = input("\nВыбор: ")
        except EOFError:
            print()
            return []
        try:
            return parse_choice(answer, names, groups)
        except ValueError as error:
            print(f"Нет такого варианта: {error}. Повтори выбор.")


def main():
    print(f"Датасет: {REPO}")
    sizes, reports = repo_index(REPO)
    if not sizes:
        raise SystemExit(f"В {REPO} нет чекпоинтов.")
    names = sorted(sizes)
    chosen = ask(names, sizes)
    if not chosen:
        print("Ничего не скачано.")
        return
    weight = sum(sizes[name] for name in chosen)
    print("\nБудут скачаны:")
    for name in chosen:
        print(f"  {name} ({megabytes(sizes[name])})")
    print(f"Вместе около {megabytes(weight)}.")
    try:
        confirm = input("Начать загрузку? [y/N]: ").strip().lower()
    except EOFError:
        print()
        return
    if confirm not in {"y", "yes", "д", "да"}:
        print("Загрузка отменена.")
        return
    patterns = [f"checkpoints/{name}/**" for name in chosen]
    patterns.extend(report_patterns(chosen, reports))
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=ROOT,
        allow_patterns=patterns,
    )
    print("Checkpoints:", ROOT / "checkpoints")
    print("Reports:", ROOT / "results")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗагрузка прервана.")
        raise SystemExit(130)
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
