"""Local sandbox: train a flat NER model, inspect metrics, compare predictions.

Text stays on this machine. Models are discovered in checkpoints/.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

import rich_eval
import study
from levels import MODE_LABELS
from modeling import MultiLevelBertForTokenClassification
from registry import (
    BASE_MODEL,
    PRESETS,
    UNTRAINED,
    available_datasets,
    label_source,
    named_models,
    read_history,
    start_training,
    trained_mode,
    training_status,
)
from train_baseline import predict_document

MAX_LENGTH = 256
STRIDE = 64
EXAMPLES = {
    "Короткий текст": (
        "Президент России Владимир Путин провёл встречу в Москве 12 марта 2020 года."
    ),
    "Несколько типов": (
        "Компания Яндекс открыла офис в Санкт-Петербурге. "
        "Инженер-программист Анна Смирнова получила премию в размере 150 тысяч рублей."
    ),
    "Длинный текст": (
        "В 1991 году Борис Ельцин стал президентом России. "
        "Московский государственный университет находится на Воробьёвых горах. "
        "Закон «Об образовании» обсуждали в Государственной думе 15 января. "
        "Режиссёр Андрей Тарковский снял фильм «Солярис». "
        "Около 12 процентов жителей района говорят на татарском языке. "
        "Организация «Врачи без границ» работала в районе Цхинвала во время конфликта. "
        "Профессор Иванов опубликовал статью в журнале «Наука и жизнь» и получил орден. "
        "Город Казань расположен в Республике Татарстан. "
        "Суд назначил штраф 50 тысяч рублей по статье уголовного кодекса. "
        "Фестиваль «Золотая маска» прошёл в театре на Большой Дмитровке вечером 3 апреля."
    ),
}
PALETTE = [
    "#f4c7c3", "#fce8b2", "#c8e6c9", "#bbdefb", "#e1bee7",
    "#ffe0b2", "#b2dfdb", "#d7ccc8", "#c5cae9", "#f8bbd0",
    "#dcedc8", "#b3e5fc", "#fff9c4", "#d1c4e9", "#ffccbc",
]


def colors(entity_types):
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(sorted(entity_types))}


@st.cache_resource(show_spinner="Загрузка модели")
def load_trained(path_str: str):
    if not torch.cuda.is_available():
        raise RuntimeError("Нужна GPU cuda:0.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    path = Path(path_str)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = MultiLevelBertForTokenClassification.from_pretrained(path).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return model, tokenizer, id2label, device


@st.cache_resource(show_spinner="Загрузка модели без дообучения")
def load_untrained(label_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("Нужна GPU cuda:0.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    source = Path(label_path)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    trained = MultiLevelBertForTokenClassification.from_pretrained(source)
    id2label = {int(key): value for key, value in trained.config.id2label.items()}
    label2id = {label: index for index, label in id2label.items()}
    del trained
    from train_baseline import build_model
    model = build_model(BASE_MODEL, id2label, label2id).to(device)
    model.eval()
    return model, tokenizer, id2label, device


def recognize(choice: str, catalog: dict[str, Path], text: str, mode: str):
    if choice == UNTRAINED:
        source = label_source()
        if source is None:
            raise RuntimeError("Нет обученного checkpoint, из которого можно взять набор меток.")
        model, tokenizer, id2label, device = load_untrained(str(source))
    else:
        model, tokenizer, id2label, device = load_trained(str(catalog[choice]))
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    entities = predict_document(
        model, tokenizer, text, id2label, device, MAX_LENGTH, STRIDE, mode=mode
    )
    finished.record()
    torch.cuda.synchronize()
    elapsed = started.elapsed_time(finished) / 1000
    rows = [
        {"type": entity_type, "text": text[start:end], "start": start, "end": end}
        for start, end, entity_type in entities
    ]
    return rows, elapsed


def highlight(text, rows):
    palette = colors({row["type"] for row in rows})
    ordered = sorted(rows, key=lambda item: (item["start"], -(item["end"] - item["start"]), item["type"]))

    def walk(start, end, pool):
        pieces = []
        cursor = start
        index = 0
        while index < len(pool):
            row = pool[index]
            if row["start"] >= end:
                break
            if row["start"] < cursor or row["end"] > end:
                index += 1
                continue
            pieces.append(html.escape(text[cursor:row["start"]]))
            children = [
                child for child in pool[index + 1:]
                if child["start"] >= row["start"] and child["end"] <= row["end"]
            ]
            label = html.escape(row["type"])
            pieces.append(
                f'<mark style="background:{palette[row["type"]]};padding:0 3px;border-radius:3px" '
                f'title="{label}">{walk(row["start"], row["end"], children)} <sub>{label}</sub></mark>'
            )
            cursor = row["end"]
            index += 1
            while index < len(pool) and pool[index]["end"] <= cursor:
                index += 1
        pieces.append(html.escape(text[cursor:end]))
        return "".join(pieces)

    return walk(0, len(text), ordered)


def load_report(path: Path, split: str, dataset: Path, mode: str):
    return rich_eval.load_saved_report(path, split, dataset, mode)


MODE_CHOICES = ("вложенный", "плоский")
MODE_CODE = {"вложенный": "nested", "плоский": "flat"}
SPLIT_TITLE = {
    "dev": "проверка, dev",
    "test": "итоговый тест",
}


def top_confusion(report, limit=8):
    rows = [
        {"было": row["gold"], "модель сказала": row["pred"], "раз": row["count"]}
        for row in report["confusion"]
        if row["gold"] not in {"лишнее"}
        and row["pred"] not in {"не найдено"}
        and row["gold"] != row["pred"]
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("раз", ascending=False).head(limit)


def missed_table(report, limit=8):
    rows = [
        {"тип": row["gold"], "раз": row["count"]}
        for row in report["confusion"]
        if row["pred"] == "не найдено"
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("раз", ascending=False).head(limit)


def invented_table(report, limit=8):
    rows = [
        {"тип, который поставила модель": row["pred"], "раз": row["count"]}
        for row in report["confusion"]
        if row["gold"] == "лишнее"
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("раз", ascending=False).head(limit)


def page_playground(catalog):
    st.subheader("Сравнение на тексте")
    st.caption(
        "Все сохранённые модели остаются в списке. "
        "Новый прогон добавляется рядом и не заменяет прежний."
    )
    example = st.selectbox("Пример", list(EXAMPLES))
    text = st.text_area("Текст", EXAMPLES[example], height=160)
    mode_label = st.radio(
        "Что показывать",
        MODE_CHOICES,
        horizontal=True,
        help="Плоский режим показывает только lvl_0. Вложенный показывает все три.",
    )
    mode = MODE_CODE[mode_label]
    options = [UNTRAINED] + list(catalog)
    primary = [name for name in options if "seed" not in name and "lr" not in name]
    selected = st.multiselect("Модели", options, default=primary[:5])
    if st.button("Сравнить", type="primary"):
        if len(selected) < 1:
            st.warning("Выберите хотя бы одну модель.")
            return
        ordered = ([UNTRAINED] if UNTRAINED in selected else []) + [
            name for name in selected if name != UNTRAINED
        ]
        runs = {name: recognize(name, catalog, text, mode) for name in ordered}
        columns = st.columns(len(ordered))
        sets = {}
        for column, name in zip(columns, ordered):
            rows, elapsed = runs[name]
            sets[name] = {(row["start"], row["end"], row["type"]) for row in rows}
            with column:
                st.markdown(f"**{name}**")
                st.markdown(highlight(text, rows), unsafe_allow_html=True)
                st.write(f"{len(rows)} сущностей, {elapsed:.3f} с")
        found = {}
        for name, spans in sets.items():
            for span in spans:
                found.setdefault(span, []).append(name)
        diff_rows = []
        for (start, end, entity_type), owners in sorted(found.items()):
            if len(owners) == len(ordered):
                continue
            diff_rows.append({
                "models": ", ".join(owners),
                "type": entity_type,
                "text": text[start:end],
                "start": start,
                "end": end,
            })
        st.subheader("Различия")
        st.dataframe(pd.DataFrame(diff_rows), hide_index=True, width="stretch")


def page_train():
    st.subheader("Обучение")
    st.caption(
        "Запуск идёт в фоне на cuda:0 и не смотрит на test. "
        "Графики читаются из истории эпох. Имя потом видно в сравнении."
    )
    datasets = available_datasets()
    status = training_status()
    busy = bool(status and status.get("running"))
    if not datasets:
        st.error("В папке data нет датасета со split train, dev и test.")
        training_monitor()
        return
    dataset_labels = [label for label, _path in datasets]
    with st.form("train-form"):
        name = st.text_input("Имя модели", "ruBERT-base, новый прогон")
        dataset_label = st.selectbox("Датасет", dataset_labels)
        mode_label = st.selectbox(
            "Режим разметки",
            MODE_CHOICES,
            help="Вложенный учит три слоя. Плоский учит только lvl_0. Оценивать потом можно в любом режиме.",
        )
        preset = st.selectbox("Исходная модель", list(PRESETS))
        custom = st.text_input("Или свой идентификатор Hugging Face", "")
        epochs = st.number_input("Эпохи", min_value=1, max_value=30, value=12)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-3, value=2e-5, format="%e")
        batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=8)
        loss = st.selectbox("Loss", ["ce", "weighted"])
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
        submitted = st.form_submit_button("Запустить", disabled=busy, type="primary")
    if submitted:
        model_name = custom.strip() or PRESETS[preset]
        dataset = dict(datasets)[dataset_label]
        try:
            output = start_training(
                name, model_name, int(epochs), float(lr), int(batch_size), loss, int(seed),
                str(dataset), MODE_CODE[mode_label],
            )
            st.success(f"Обучение запущено: {output.name}")
        except RuntimeError as error:
            st.error(str(error))
    training_monitor()


@st.fragment(run_every=2)
def training_monitor():
    status = training_status()
    if not status:
        st.info("Сейчас ничего не обучается.")
        return
    output = Path(status["output"])
    if status.get("running"):
        st.warning(f"Идёт обучение «{status['name']}». GPU занята.")
    else:
        st.success(f"Обучение «{status['name']}» завершилось. Модель в списке сравнения.")
    history = read_history(output)
    if history is not None and not history.empty:
        epoch = int(history["epoch"].iloc[-1])
        total = int(status.get("epochs") or epoch)
        st.progress(min(epoch / total, 1.0), text=f"Эпоха {epoch} из {total}")
        st.line_chart(history, x="epoch", y="train_loss", height=220)
        st.line_chart(history, x="epoch", y=["dev_f1", "dev_macro_f1"], height=220)
        st.dataframe(history, hide_index=True, width="stretch")
    log_path = Path(status["log"])
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        st.code("\n".join(lines[-25:]) or "Первых строк лога ещё нет.")


def render_report(report, model_name):
    strict = report["strict"]
    if report.get("efficiency"):
        cost = report["efficiency"]
        st.caption(
            f"Параметров {cost['parameters'] / 1e6:.1f} млн. "
            f"Инференс этой выборки: {cost['inference_seconds']:.2f} с на {cost['documents']} документов, "
            f"пик памяти {cost['peak_memory_bytes'] / 1024**2:.0f} МиБ."
        )
    interval = report["bootstrap"]["strict_f1"]
    split_title = SPLIT_TITLE.get(report["split"], report["split"])
    st.subheader(f"{model_name}, {split_title}")
    st.caption("Ответ засчитан, только если совпали тип сущности, начало и конец.")
    precision, recall, f1 = st.columns(3)
    precision.metric("Точность", f"{strict['precision']:.3f}")
    recall.metric("Полнота", f"{strict['recall']:.3f}")
    f1.metric("F1", f"{strict['f1']:.3f}")
    st.caption(
        f"Если перемешать документы этой выборки, F1 обычно попадает в "
        f"{interval['low']:.3f}–{interval['high']:.3f}. Это один прогон обучения, не несколько."
    )

    kinds = report["error_kinds"]
    reference = strict["gold_entities"]
    st.subheader("Куда делись сущности из эталона")
    st.dataframe(pd.DataFrame([
        {"что произошло": "Совпало полностью", "сколько": kinds.get("tp", 0)},
        {"что произошло": "Тип верный, границы съехали", "сколько": kinds.get("boundary", 0)},
        {"что произошло": "Текст пересекается, тип другой", "сколько": kinds.get("boundary_and_type", 0)},
        {"что произошло": "Границы те же, тип другой", "сколько": kinds.get("wrong_type", 0)},
        {"что произошло": "Рядом нет никакой находки", "сколько": kinds.get("missed", 0)},
    ]), hide_index=True, width="stretch")
    st.caption(
        f"В эталоне {reference} сущностей. "
        f"Отдельно модель поставила {kinds.get('spurious', 0)} находок там, где в эталоне сущности нет."
    )

    st.subheader("Качество по типам")
    st.caption("Чем меньше примеров, тем менее надёжна цифра F1 у этого типа.")
    per_type = pd.DataFrame([
        {
            "тип": name,
            "примеров": row["support"],
            "точность": round(row["precision"], 3),
            "полнота": round(row["recall"], 3),
            "F1": round(row["f1"], 3),
        }
        for name, row in report["per_type"].items()
    ]).sort_values(["примеров", "тип"], ascending=[False, True])
    st.dataframe(per_type, hide_index=True, width="stretch")

    st.subheader("Где модель ошибается")
    confused, missed, invented = st.columns(3)
    confused.markdown("**Путает один тип с другим**")
    confused.dataframe(top_confusion(report), hide_index=True, width="stretch")
    missed.markdown("**Чаще всего не находит**")
    missed.dataframe(missed_table(report), hide_index=True, width="stretch")
    invented.markdown("**Чаще всего ставит лишнее**")
    invented.dataframe(invented_table(report), hide_index=True, width="stretch")

    with st.expander("Подробнее"):
        macro = report["bootstrap"]["strict_macro_f1"]
        relaxed = report["relaxed_overlap"]
        nested = report["nested"]
        mode = report.get("annotation_mode", "nested")
        st.dataframe(pd.DataFrame([
            {"показатель": "macro F1", "значение": round(strict["macro_f1"], 3)},
            {"показатель": "macro F1, интервал", "значение": f"{macro['low']:.3f}–{macro['high']:.3f}"},
            {"показатель": "weighted F1", "значение": round(strict["weighted_f1"], 3)},
            {"показатель": "мягкий F1", "значение": round(relaxed["f1"], 3)},
            {"показатель": "совпали тип и начало", "значение": round(report["same_start"]["f1"], 3)},
            {"показатель": "совпали тип и конец", "значение": round(report["same_end"]["f1"], 3)},
            {"показатель": "непрерывных в разметке", "значение": nested["continuous"]},
            {"показатель": "в эталоне", "значение": nested["flat_gold"]},
            {"показатель": "снято внутри слоя", "значение": nested.get("within_level_removed", 0)},
            {"показатель": "разорванных", "значение": nested["discontinuous"]},
        ]), hide_index=True, width="stretch")
        if mode == "nested" and report.get("per_level"):
            level_rows = []
            for name, row in report["per_level"].items():
                level_rows.append({
                    "слой": name,
                    "precision": round(row["precision"], 3),
                    "recall": round(row["recall"], 3),
                    "F1": round(row["f1"], 3),
                    "эталон": row["tp"] + row["fn"],
                })
            st.dataframe(pd.DataFrame(level_rows), hide_index=True, width="stretch")
        length_left, length_middle, length_right = st.columns(3)
        length_left.markdown("**Длина в символах**")
        length_left.dataframe(pd.DataFrame(report["by_char_length"]), hide_index=True, width="stretch")
        length_middle.markdown("**Длина в токенах**")
        length_middle.dataframe(pd.DataFrame(report["by_token_length"]), hide_index=True, width="stretch")
        length_right.markdown("**Длина документа**")
        length_right.dataframe(pd.DataFrame(report["by_doc_length"]), hide_index=True, width="stretch")


def render_study(dataset: Path, split: str, mode: str):
    rows = study.seed_rows(dataset, split, mode, "base")
    ready = [row for row in rows if row["flat_f1"] is not None and row["nested_f1"] is not None]
    if ready:
        st.subheader("Base, несколько seed")
        st.caption(
            "Один и тот же рецепт: learning rate 2e-5, batch 8, 12 эпох. "
            "Разница — F1 вложенного обучения минус F1 плоского, на одних и тех же документах."
        )
        table = []
        for row in ready:
            paired = row.get("paired")
            table.append({
                "seed": row["seed"],
                "плоский F1": round(row["flat_f1"], 3),
                "вложенный F1": round(row["nested_f1"], 3),
                "разница": None if not paired else round(paired["observed_diff"], 3),
                "интервал разницы": None if not paired else f"{paired['low']:.3f}–{paired['high']:.3f}",
            })
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        flat_stats = study.mean_std([row["flat_f1"] for row in ready])
        nested_stats = study.mean_std([row["nested_f1"] for row in ready])
        diff_values = [row["paired"]["observed_diff"] for row in ready if row.get("paired")]
        diff_stats = study.mean_std(diff_values)
        text = (
            f"Среднее плоского F1 {flat_stats['mean']:.3f} ± {flat_stats['std']:.3f}. "
            f"Среднее вложенного F1 {nested_stats['mean']:.3f} ± {nested_stats['std']:.3f}."
        )
        if diff_stats:
            text += f" Средняя разница {diff_stats['mean']:.3f} ± {diff_stats['std']:.3f}."
        st.caption(text)
    sweep = study.tiny_sweep(dataset, split, mode)
    if sweep:
        st.subheader("Tiny2, подбор шага")
        st.caption("Batch 16. Кривая каждой модели открывается ниже, в блоке выбранной модели.")
        st.dataframe(pd.DataFrame([
            {
                "модель": row["name"],
                "режим обучения": MODE_LABELS.get(row["mode"], row["mode"]),
                "lr": row["lr"],
                "лучшая эпоха": None if not row["history"] else row["history"]["best_epoch"],
                "dev F1": None if not row["history"] else round(row["history"]["best_dev_f1"], 3),
                "последняя dev F1": None if not row["history"] else round(row["history"]["last_dev_f1"], 3),
                "test F1": None if row["test_f1"] is None else round(row["test_f1"], 3),
            }
            for row in sweep
        ]), hide_index=True, width="stretch")


def page_results(catalog):
    st.subheader("Метрики")
    if not catalog:
        st.info("Пока нет сохранённых моделей.")
        return
    datasets = available_datasets()
    if not datasets:
        st.error("В папке data нет датасета со split train, dev и test.")
        return
    dataset_labels = [label for label, _path in datasets]
    dataset_label = st.selectbox("Датасет", dataset_labels)
    dataset = dict(datasets)[dataset_label]
    mode_label = st.radio(
        "Режим оценки",
        MODE_CHOICES,
        horizontal=True,
        help="Любую модель можно посчитать в обоих режимах. Плоский сравнивает lvl_0 с плоским эталоном.",
    )
    mode = MODE_CODE[mode_label]
    split = st.radio("Выборка", ["dev", "test"], horizontal=True, index=1)
    if split == "test":
        st.caption("Итоговая оценка. По этой выборке параметры не подбирают.")
    names = list(catalog)
    overview = []
    for name in names:
        report = load_report(catalog[name], split, dataset, mode)
        trained = MODE_LABELS.get(trained_mode(catalog[name]), "вложенный")
        if not report:
            overview.append({
                "модель": name,
                "обучалась": trained,
                "точность": "ещё не посчитано",
                "полнота": "",
                "F1": "",
            })
            continue
        band = report["bootstrap"]["strict_f1"]
        overview.append({
            "модель": name,
            "обучалась": trained,
            "точность": round(report["strict"]["precision"], 3),
            "полнота": round(report["strict"]["recall"], 3),
            "F1": round(report["strict"]["f1"], 3),
            "F1, интервал": f"{band['low']:.3f}–{band['high']:.3f}",
        })
    st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch")
    render_study(dataset, split, mode)
    selected = st.selectbox("Подробно", names)
    path = catalog[selected]
    history = read_history(path)
    if history is not None and not history.empty:
        st.subheader("Обучение")
        st.line_chart(history, x="epoch", y="train_loss", height=220)
        st.line_chart(history, x="epoch", y=["dev_f1", "dev_macro_f1"], height=220)
    report = load_report(path, split, dataset, mode)
    if report:
        render_report(report, selected)
    else:
        st.info("Подробный отчёт для этой пары модели и выборки ещё не посчитан.")
    busy = training_status()
    if busy and busy.get("running"):
        st.warning("Пока идёт обучение, запустить оценку невозможно: GPU занята.")
        return
    if st.button(f"Посчитать подробную оценку на {split}"):
        with st.spinner("Идёт расчёт предсказаний и интервалов. Обучение при этом не запускается."):
            rich_eval.save_report(path, split, str(dataset), draws=1000, mode=mode)
        st.rerun()


def main():
    st.set_page_config(page_title="NEREL sandbox", layout="wide")
    st.title("NEREL: обучение и сравнение")
    st.caption(
        "Один и тот же датасет можно учить и проверять плоско или по трём слоям. "
        "Модель, обученная на вложенных сущностях, считается и на плоском эталоне: туда входят только предсказания lvl_0. "
        "Текст не уходит с этого компьютера."
    )
    catalog = dict(named_models())
    page = st.sidebar.radio("Раздел", ["Сравнение", "Обучение", "Метрики"])
    if page == "Сравнение":
        page_playground(catalog)
    elif page == "Обучение":
        page_train()
    else:
        page_results(catalog)


if __name__ == "__main__":
    main()
