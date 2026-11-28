from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_golden import (
    DEFAULT_PER_CLASS_THRESHOLDS,
    run_golden_evaluation,
    save_report,
)
from src.constants import CODES_ORDER
from src.model import NL2SQLQualityModel
from src.postprocess import postprocess_predictions
from src.schema_validator import SchemaValidator

st.set_page_config(
    page_title="NL2SQL Quality Gate",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_REPORT_FILE = MODELS_DIR / "metrics_report.json"
GOLDEN_TEST_172_FILE = PROJECT_ROOT / "data" / "golden_test_172.jsonl"
GOLDEN_REPORT_172_FILE = MODELS_DIR / "golden_test_report_172.json"

CODE_CATEGORIES: dict[str, str] = {
    "UNKNOWN_TABLE": "Схема БД",
    "UNKNOWN_COLUMN": "Схема БД",
    "SCOPE": "Вне темы БД",
    "ATTACH": "Контекст (Местоимения)",
    "VAGUE": "Семантика (Размытость)",
    "MISSING_VALUE": "Спецификация (Значение)",
    "MISSING_TIME": "Период дат",
    "MISSING_CRITERION": "Безопасность (WHERE)",
    "RANKING_SPEC": "Ранжирование (Топ / Сортировка)",
    "MISSING_AGG": "Семантика (Агрегация)",
    "MISSING_SORT": "Сортировка",
    "MISSING_LIMIT": "Лимит строк",
}

SAMPLE_SCHEMAS: dict[str, str] = {
    "Сотрудники и отделы (HR)": (
        "TABLE departments: department_id INT PRIMARY KEY, name VARCHAR, floor INT | "
        "TABLE employees: employee_id INT PRIMARY KEY, name VARCHAR, salary INT, department_id INT, status VARCHAR"
    ),
    "Электронная коммерция (Shop)": (
        "TABLE customers: id INT PRIMARY KEY, name VARCHAR, phone VARCHAR, email VARCHAR, city VARCHAR | "
        "TABLE products: product_id INT PRIMARY KEY, name VARCHAR, price INT, category_id INT | "
        "TABLE orders: order_id INT PRIMARY KEY, customer_id INT, order_date DATE, total_amount INT, status VARCHAR"
    ),
    "Авиакомпании и рейсы (Flights)": (
        "TABLE airports: airport_id INT PRIMARY KEY, city VARCHAR, code VARCHAR | "
        "TABLE flights: flight_id INT PRIMARY KEY, flight_number VARCHAR, departure_date DATE, price INT, status VARCHAR"
    ),
    "Медицинская клиника (Hospital)": (
        "TABLE doctors: doctor_id INT PRIMARY KEY, name VARCHAR, specialty VARCHAR, salary INT | "
        "TABLE appointments: appointment_id INT PRIMARY KEY, doctor_id INT, patient_id INT, appointment_date DATE"
    ),
}

STATUS_LABELS: dict[str, str] = {
    "clear": "CLEAR (Понятный)",
    "incomplete": "INCOMPLETE (Неполный)",
    "ambiguous": "AMBIGUOUS (Неоднозначный)",
}


def get_available_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.pt"), key=lambda model_path: (model_path.name != "best_model.pt", model_path.name))


def load_tokenizer(model_name: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)


@st.cache_resource
def load_runtime_artifacts(checkpoint_path_str: str) -> tuple[NL2SQLQualityModel | None, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        return None, None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "cointegrated/rubert-tiny2")
    codes = config.get("codes_order", CODES_ORDER)

    tokenizer = load_tokenizer(model_name)
    model = NL2SQLQualityModel(model_name=model_name, num_codes=len(codes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def run_prediction(
    model: NL2SQLQualityModel,
    tokenizer: Any,
    question: str,
    db_type: str,
    schema: str,
) -> tuple[list[float], float]:
    device = next(model.parameters()).device
    prompt = f"СУБД: {db_type}. Запрос: {question}"
    encoded = tokenizer(prompt, schema, truncation="only_second", max_length=512, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in encoded.items()}

    start_time = time.perf_counter()
    with torch.no_grad():
        probabilities = model.predict_probabilities(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return probabilities.squeeze(0).cpu().tolist(), latency_ms


def apply_typo(text: str) -> str:
    words = text.split()
    if not words:
        return text
    target_index = random.randint(0, len(words) - 1)
    selected_word = words[target_index]
    if len(selected_word) > 4:
        swap_index = random.randint(1, len(selected_word) - 3)
        words[target_index] = (
            selected_word[:swap_index]
            + selected_word[swap_index + 1]
            + selected_word[swap_index]
            + selected_word[swap_index + 2:]
        )
    elif len(selected_word) > 2:
        words[target_index] = selected_word + selected_word[-1]
    return " ".join(words)


def build_chart_dataframe(codes: list[str], probabilities: list[float], threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Дефект": codes,
            "Вероятность (%)": [round(probability * 100.0, 1) for probability in probabilities],
            "Статус": ["Превышен порог" if probability >= threshold else "Ниже порога" for probability in probabilities],
        }
    )


def render_probability_chart(codes: list[str], probabilities: list[float], threshold: float) -> None:
    chart_data = build_chart_dataframe(codes, probabilities, threshold)
    color_scale = alt.Scale(domain=["Превышен порог", "Ниже порога"], range=["#dc2626", "#2563eb"])
    bar_chart = alt.Chart(chart_data).mark_bar(cornerRadius=3).encode(
        x=alt.X("Вероятность (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Дефект:N", sort="-x"),
        color=alt.Color("Статус:N", scale=color_scale, legend=alt.Legend(title="Статус порога")),
        tooltip=["Дефект", "Вероятность (%)", "Статус"],
    ).properties(height=320)
    threshold_rule = alt.Chart(pd.DataFrame({"x": [threshold * 100.0]})).mark_rule(
        color="#ea580c", strokeDash=[4, 4], size=2
    ).encode(x="x:Q")
    st.altair_chart(bar_chart + threshold_rule, use_container_width=True)


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_golden_summary(report: dict[str, Any]) -> None:
    status_col, exact_col, f1_col, latency_col = st.columns(4)
    status_col.metric("Точность статуса", format_percentage(report.get("status_accuracy", 0.0)))
    exact_col.metric("Точное совпадение (EM)", format_percentage(report.get("code_subset_accuracy", 0.0)))
    f1_col.metric("Macro F1", format_percentage(report.get("macro_f1", 0.0)))
    latency_col.metric("Задержка", f"{report.get('avg_latency', 0.0):.2f} мс")


def render_golden_matrix(confusion_matrix: list[list[int]]) -> None:
    st.markdown("##### Матрица ошибок бизнес-статусов")
    matrix_df = pd.DataFrame(
        confusion_matrix,
        index=["Факт: CLEAR", "Факт: INCOMPLETE", "Факт: AMBIGUOUS"],
        columns=["Прогноз: CLEAR", "Прогноз: INCOMPLETE", "Прогноз: AMBIGUOUS"],
    )
    st.dataframe(matrix_df)


def build_class_metric_row(code: str, stats: dict[str, Any], pr_auc: float | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Код дефекта": code,
        "Precision": format_percentage(stats.get("precision", 0.0)),
        "Recall": format_percentage(stats.get("recall", 0.0)),
        "F1-Score": format_percentage(stats.get("f1", 0.0)),
    }
    if pr_auc is not None:
        row["PR-AUC"] = format_percentage(pr_auc)
    row["Support"] = int(stats.get("support", 0))
    return row


def render_golden_code_table(codes_order: list[str], per_class_metrics: dict[str, Any]) -> None:
    st.markdown(f"##### Детализация по {len(codes_order)} кодам дефектов")
    rows = []
    for code in codes_order:
        row = build_class_metric_row(code, per_class_metrics.get(code, {}))
        row["Категория"] = CODE_CATEGORIES.get(code, "Общий")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))


def render_golden_record_item(record: dict[str, Any]) -> None:
    is_matched = record["status_match"] and record["code_exact_match"]
    prefix = "[OK]" if is_matched else "[РАСХОЖДЕНИЕ]"
    title = f"{prefix} {record['id']}: {record['question']}"
    with st.expander(title, expanded=not is_matched):
        st.markdown(f"**Ожидаемый статус:** `{record['expected_status']}`")
        st.markdown(f"**Предсказанный статус:** `{record['pred_status']}`")
        st.markdown(f"**Ожидаемые коды:** `{record['expected_codes']}`")
        st.markdown(f"**Предсказанные коды:** `{record['pred_codes']}`")
        st.markdown(f"**Схема:** `{record.get('schema', '')}`")
        active_probs = {
            code: f"{prob * 100:.1f}%"
            for code, prob in record["probabilities"].items()
            if prob >= 0.15 or code in record["expected_codes"]
        }
        st.markdown(f"**Вероятности кодов:** `{active_probs}`")


def render_golden_mismatches(records: list[dict[str, Any]]) -> None:
    st.markdown("Детальный анализ примеров")
    mismatches = [record for record in records if not (record["status_match"] and record["code_exact_match"])]
    view_mode = st.radio(
        "Фильтр отображения:",
        [f"Только расхождения ({len(mismatches)})", f"Все примеры ({len(records)})"],
        horizontal=True,
    )
    items_to_show = mismatches if "Только расхождения" in view_mode else records
    for record in items_to_show:
        render_golden_record_item(record)


def reset_calibrated_thresholds(codes_order: list[str]) -> None:
    for code in codes_order:
        st.session_state[f"calib_slider_{code}"] = DEFAULT_PER_CLASS_THRESHOLDS.get(code, 0.50)


def render_calibrated_sliders(codes_order: list[str]) -> dict[str, float]:
    custom_thresholds: dict[str, float] = {}
    with st.expander("Настройка порогов по классам", expanded=True):
        st.button("Сбросить к автонастройке", on_click=reset_calibrated_thresholds, args=(codes_order,))
        for code in codes_order:
            slider_key = f"calib_slider_{code}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = DEFAULT_PER_CLASS_THRESHOLDS.get(code, 0.50)
            custom_thresholds[code] = st.slider(code, min_value=0.05, max_value=0.95, step=0.05, key=slider_key)
    return custom_thresholds


def select_golden_threshold(codes_order: list[str]) -> float | dict[str, float]:
    threshold_mode = st.radio("Режим порога:", ["Фиксированный", "Калиброванный"], horizontal=True, key="golden_thresh_mode")
    if threshold_mode == "Фиксированный":
        return st.slider("Общий порог отсечения:", 0.1, 0.9, 0.5, 0.05, key="golden_threshold")
    return render_calibrated_sliders(codes_order)


def execute_golden_run(
    model: NL2SQLQualityModel | None, tokenizer: Any, codes_order: list[str],
    threshold: float | dict[str, float], state_key: str,
) -> None:
    if model is None or tokenizer is None:
        st.error("Модель не найдена или не загружена.")
        return
    if not GOLDEN_TEST_172_FILE.exists():
        st.error(f"Файл эталонного теста не найден: {GOLDEN_TEST_172_FILE}")
        return
    with st.spinner("Выполняется тестирование по 172 эталонным запросам..."):
        report = run_golden_evaluation(model, tokenizer, codes_order, GOLDEN_TEST_172_FILE, threshold)
        save_report(GOLDEN_REPORT_172_FILE, report)
        st.session_state[state_key] = report
        st.success("Тестирование успешно завершено!")


def load_or_run_golden_report(
    model: NL2SQLQualityModel | None, tokenizer: Any, codes_order: list[str],
    threshold: float | dict[str, float], model_key: str,
) -> dict[str, Any] | None:
    state_key = f"golden_report_172_{model_key}"
    if st.button("Запустить тест", type="primary", key="golden_run_btn"):
        execute_golden_run(model, tokenizer, codes_order, threshold, state_key)
    if state_key in st.session_state:
        return st.session_state[state_key]
    if GOLDEN_REPORT_172_FILE.exists():
        with open(GOLDEN_REPORT_172_FILE, "r", encoding="utf-8") as report_file:
            return json.load(report_file)
    return None


def render_golden_tab(model: NL2SQLQualityModel | None, tokenizer: Any, config: dict[str, Any], model_key: str) -> None:
    codes_order = config.get("codes_order", CODES_ORDER)
    st.text_input(
        "Файл эталонного теста:",
        value=f"{GOLDEN_TEST_172_FILE.name} (172 примера)",
        disabled=True,
        key="golden_file_display",
    )
    threshold = select_golden_threshold(codes_order)
    report = load_or_run_golden_report(model, tokenizer, codes_order, threshold, model_key)

    if report is None:
        st.info("Нажмите кнопку «Запустить тест», чтобы выполнить проверку модели.")
        return

    st.markdown("---")
    render_golden_summary(report)
    render_golden_matrix(report["confusion_matrix"])
    render_golden_code_table(codes_order, report.get("per_class_metrics", {}))
    render_golden_mismatches(report.get("records", []))


def render_benchmark_summary(report: dict[str, Any]) -> None:
    f1_col, exact_col, false_clear_col, ece_col = st.columns(4)
    multilabel = report.get("multilabel", {})
    clarity = report.get("clarity", {})
    calibration = report.get("calibration", {})
    f1_col.metric("Macro F1", format_percentage(multilabel.get("macro_f1", 0.0)))
    exact_col.metric("Exact Match", format_percentage(multilabel.get("subset_accuracy", 0.0)))
    false_clear_col.metric("False Clear", f"{clarity.get('false_clear_rate', 0.0):.2f}%")
    ece_col.metric("ECE", f"{calibration.get('expected_calibration_error', 0.0):.4f}")


def render_benchmark_tab() -> None:
    if not METRICS_REPORT_FILE.exists():
        st.info("Файл metrics_report.json не найден.")
        return
    with open(METRICS_REPORT_FILE, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    render_benchmark_summary(report)
    st.markdown("##### Детализация по классам дефектов")
    per_class = report.get("per_class", {})
    pr_auc_data = report.get("pr_auc", {}).get("per_class_pr_auc", {})
    rows = [build_class_metric_row(code, stats, pr_auc_data.get(code, 0.0)) for code, stats in per_class.items()]
    st.dataframe(pd.DataFrame(rows))


def on_schema_preset_change() -> None:
    selected_name = st.session_state.get("playground_schema_preset")
    if selected_name in SAMPLE_SCHEMAS:
        st.session_state["playground_schema_text"] = SAMPLE_SCHEMAS[selected_name]


def render_playground_inputs() -> tuple[str, str, str, float, bool, bool]:
    if "playground_schema_text" not in st.session_state:
        st.session_state["playground_schema_text"] = next(iter(SAMPLE_SCHEMAS.values()))

    st.selectbox("Заготовленная схема БД:", list(SAMPLE_SCHEMAS.keys()), key="playground_schema_preset", on_change=on_schema_preset_change)
    schema_text = st.text_area("Схема базы данных (DDL):", height=100, key="playground_schema_text")
    selected_db = st.selectbox("СУБД:", ["postgresql", "mysql", "sqlserver"], key="playground_db")
    threshold = st.slider("Порог отсечения:", 0.1, 0.9, 0.5, 0.05, key="playground_threshold")
    user_query = st.text_area("Запрос пользователя:", height=90, placeholder="Введите текст запроса", key="playground_query_text")
    check_clicked = st.button("Проверить запрос", type="primary")
    typo_clicked = st.button("Внести опечатку")
    return user_query, schema_text, selected_db, threshold, check_clicked, typo_clicked


def render_detected_remarks(result: dict[str, Any]) -> None:
    st.markdown("##### Выявленные замечания:")
    if not result["detected_codes"]:
        st.success("Запрос понятен и готов к генерации SQL.")
        return
    for code, reason, clarification in zip(
        result["detected_codes"], result["reasons"], result["suggested_clarifications"]
    ):
        with st.expander(code, expanded=True):
            st.markdown(f"**Причина:** {reason}")
            st.markdown(f"**Рекомендация:** {clarification}")


def render_playground_results(
    result: dict[str, Any], codes_order: list[str], probabilities: list[float], threshold: float, latency_ms: float
) -> None:
    st.markdown("---")
    status_col, score_col, count_col, latency_col = st.columns(4)
    status = result["ambiguity_status"]
    status_col.metric("Статус", STATUS_LABELS.get(status, status))
    score_col.metric("Оценка полноты", f"{result['completeness_score']:.2f} / 1.00")
    count_col.metric("Дефектов найдено", len(result["detected_codes"]))
    latency_col.metric("Время анализа", f"{latency_ms:.2f} мс")
    st.markdown("Вероятности по дефектам:")
    render_probability_chart(codes_order, probabilities, threshold)
    render_detected_remarks(result)


def render_playground_tab(model: NL2SQLQualityModel | None, tokenizer: Any, config: dict[str, Any]) -> None:
    user_query, schema_text, selected_db, threshold, check_clicked, typo_clicked = render_playground_inputs()
    if typo_clicked and user_query:
        user_query = apply_typo(user_query)
        st.info(f"Запрос с опечаткой: {user_query}")
        check_clicked = True
    if not check_clicked:
        return
    if not user_query.strip() or model is None or tokenizer is None:
        st.warning("Проверьте ввод запроса и наличие загруженной модели.")
        return
    codes_order = config.get("codes_order", CODES_ORDER)
    syntax_codes = SchemaValidator(schema_text).validate_query(user_query)
    probabilities, latency_ms = run_prediction(model, tokenizer, user_query, selected_db, schema_text)
    result = postprocess_predictions(probabilities, codes_order, thresholds=threshold, syntax_codes=syntax_codes)
    render_playground_results(result, codes_order, probabilities, threshold, latency_ms)


def select_model_checkpoint() -> tuple[str, Path] | None:
    available_models = get_available_models()
    if not available_models:
        st.error("В папке backend/models не найдено ни одного файла модели (*.pt).")
        return None
    model_names = [model_path.name for model_path in available_models]
    selected_name = st.selectbox("Выбор модели:", model_names, key="selected_model_file")
    selected_path = next(model_path for model_path in available_models if model_path.name == selected_name)
    return selected_name, selected_path


def main() -> None:
    selection = select_model_checkpoint()
    if selection is None:
        return
    selected_model_name, selected_model_path = selection
    model, tokenizer, config = load_runtime_artifacts(str(selected_model_path))
    tab_playground, tab_golden, tab_metrics = st.tabs([
        "Проверка запроса", "Эталонный тест (Golden Test)", "Метрики обучения"
    ])
    with tab_playground:
        render_playground_tab(model, tokenizer, config)
    with tab_golden:
        render_golden_tab(model, tokenizer, config, selected_model_name)
    with tab_metrics:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
