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

from src.constants import CODES_ORDER
from src.model import NL2SQLQualityModel
from src.postprocess import postprocess_predictions

st.set_page_config(
    page_title="NL2SQL Quality Gate",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pt"
METRICS_REPORT_FILE = PROJECT_ROOT / "models" / "metrics_report.json"

SAMPLE_SCHEMAS: dict[str, str] = {
    "Сотрудники и отделы (HR)": (
        "TABLE departments: department_id INT PRIMARY KEY, name VARCHAR, floor INT | "
        "TABLE employees: employee_id INT PRIMARY KEY, name VARCHAR, salary INT, department_id INT, status VARCHAR"
    ),
    "Авиакомпании и рейсы (Flights)": (
        "TABLE airports: airport_id INT PRIMARY KEY, city VARCHAR, code VARCHAR | "
        "TABLE flights: flight_id INT PRIMARY KEY, flight_number VARCHAR, departure_date DATE, price INT, status VARCHAR"
    ),
    "Медицинская клиника (Hospital)": (
        "TABLE doctors: doctor_id INT PRIMARY KEY, name VARCHAR, specialty VARCHAR, salary INT | "
        "TABLE appointments: appointment_id INT PRIMARY KEY, doctor_id INT, patient_id INT, appointment_date DATE"
    ),
    "Электронная коммерция (Shop)": (
        "TABLE products: product_id INT PRIMARY KEY, name VARCHAR, price INT, category_id INT | "
        "TABLE orders: order_id INT PRIMARY KEY, customer_id INT, order_date DATE, total_amount INT, status VARCHAR"
    ),
}

PRESET_QUERIES: list[dict[str, str]] = [
    {
        "label": "1. [CLEAR] Полный выбор данных без условий",
        "query": "Покажи всех сотрудников",
        "schema_key": "Сотрудники и отделы (HR)",
        "db_type": "postgresql",
    },
    {
        "label": "2. [CLEAR] Выборка с точным условием WHERE",
        "query": "Покажи сотрудников где salary равен 50000",
        "schema_key": "Сотрудники и отделы (HR)",
        "db_type": "postgresql",
    },
    {
        "label": "3. [INCOMPLETE] Пропущен временной интервал и лимит",
        "query": "Выведи свежие заказы",
        "schema_key": "Электронная коммерция (Shop)",
        "db_type": "postgresql",
    },
    {
        "label": "4. [INCOMPLETE] Не указано поле сортировки топа",
        "query": "Покажи топ 5 врачей",
        "schema_key": "Медицинская клиника (Hospital)",
        "db_type": "mysql",
    },
    {
        "label": "5. [INCOMPLETE] Изменение всех строк без условия WHERE",
        "query": "Обнови статус всех сотрудников на активен",
        "schema_key": "Сотрудники и отделы (HR)",
        "db_type": "postgresql",
    },
    {
        "label": "6. [AMBIGUOUS] Несуществующая таблица в схеме",
        "query": "Покажи список из таблицы warehouses где id=10",
        "schema_key": "Электронная коммерция (Shop)",
        "db_type": "sqlserver",
    },
    {
        "label": "7. [AMBIGUOUS] Несуществующий атрибут в таблице",
        "query": "Выведи врачей отсортированных по internal_rating",
        "schema_key": "Медицинская клиника (Hospital)",
        "db_type": "postgresql",
    },
    {
        "label": "8. [AMBIGUOUS] Размытые субъективные определения",
        "query": "Покажи самых лучших и прибыльных сотрудников",
        "schema_key": "Сотрудники и отделы (HR)",
        "db_type": "postgresql",
    },
    {
        "label": "9. [AMBIGUOUS] Запрос вне контекста базы данных",
        "query": "Какая завтра погода в Москве?",
        "schema_key": "Сотрудники и отделы (HR)",
        "db_type": "postgresql",
    },
]


@st.cache_resource
def load_runtime_artifacts() -> tuple[NL2SQLQualityModel | None, Any, dict[str, Any]]:
    if not MODEL_CHECKPOINT.exists():
        return None, None, {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "cointegrated/rubert-tiny2")
    codes = config.get("codes_order", CODES_ORDER)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    inputs = tokenizer(
        prompt,
        schema,
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model.predict_probabilities(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return probs.squeeze(0).cpu().tolist(), latency_ms


def apply_typo(text: str) -> str:
    words = text.split()
    if not words:
        return text
    target_idx = random.randint(0, len(words) - 1)
    w = words[target_idx]
    if len(w) > 4:
        idx = random.randint(1, len(w) - 3)
        words[target_idx] = w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:]
    elif len(w) > 2:
        words[target_idx] = w + w[-1]
    return " ".join(words)


def render_probability_chart(codes: list[str], probs: list[float], threshold: float) -> None:
    data = pd.DataFrame(
        {
            "Дефект": codes,
            "Вероятность (%)": [round(p * 100.0, 1) for p in probs],
            "Статус": ["Превышен порог" if p >= threshold else "Ниже порога" for p in probs],
        }
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Вероятность (%):Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Дефект:N", sort="-x"),
            color=alt.Color(
                "Статус:N",
                scale=alt.Scale(
                    domain=["Превышен порог", "Ниже порога"],
                    range=["#dc2626", "#2563eb"],
                ),
                legend=alt.Legend(title="Статус порога"),
            ),
            tooltip=["Дефект", "Вероятность (%)", "Статус"],
        )
        .properties(height=340)
    )
    rule = alt.Chart(pd.DataFrame({"x": [threshold * 100.0]})).mark_rule(
        color="#ea580c", strokeDash=[4, 4], size=2
    ).encode(x="x:Q")

    st.altair_chart(chart + rule, use_container_width=True)


def render_benchmark_tab() -> None:
    st.subheader("Показатели качества модели (контрольная выборка)")
    if not METRICS_REPORT_FILE.exists():
        st.info("Файл metrics_report.json не найден.")
        return

    with open(METRICS_REPORT_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    multi = report.get("multilabel", {})
    clarity = report.get("clarity", {})
    calib = report.get("calibration", {})

    c1.metric("Macro F1", f"{multi.get('macro_f1', 0.0) * 100:.2f}%")
    c2.metric("Exact Match", f"{multi.get('subset_accuracy', 0.0) * 100:.2f}%")
    c3.metric("Ложный пропуск (False Clear)", f"{clarity.get('false_clear_rate', 0.0):.2f}%")
    c4.metric("Ошибка калибровки (ECE)", f"{calib.get('expected_calibration_error', 0.0):.4f}")

    st.markdown("##### Детализация по классам дефектов")
    per_class = report.get("per_class", {})
    pr_auc_data = report.get("pr_auc", {}).get("per_class_pr_auc", {})

    rows = []
    for code, stats in per_class.items():
        rows.append(
            {
                "Код дефекта": code,
                "Точность (Precision)": f"{stats.get('precision', 0.0) * 100:.2f}%",
                "Полнота (Recall)": f"{stats.get('recall', 0.0) * 100:.2f}%",
                "F1-Score": f"{stats.get('f1', 0.0) * 100:.2f}%",
                "PR-AUC": f"{pr_auc_data.get(code, 0.0) * 100:.2f}%",
                "Примеров (Support)": int(stats.get("support", 0)),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_playground_tab(model: NL2SQLQualityModel | None, tokenizer: Any, config: dict[str, Any]) -> None:
    st.sidebar.header("Параметры проверки")
    selected_db = st.sidebar.selectbox("СУБД:", ["postgresql", "mysql", "sqlserver"])
    threshold = st.sidebar.slider("Порог отсечения:", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Готовые сценарии:")
    preset_labels = ["-- Ввести свой запрос --"] + [p["label"] for p in PRESET_QUERIES]
    selected_preset_label = st.sidebar.selectbox("Выбор сценария:", preset_labels)

    current_query = ""
    current_schema = SAMPLE_SCHEMAS["Сотрудники и отделы (HR)"]

    if selected_preset_label != "-- Ввести свой запрос --":
        matched = next(p for p in PRESET_QUERIES if p["label"] == selected_preset_label)
        current_query = matched["query"]
        current_schema = SAMPLE_SCHEMAS.get(matched["schema_key"], current_schema)
        selected_db = matched["db_type"]

    col_input, col_schema = st.columns([1.2, 1.0])
    with col_input:
        user_query = st.text_area(
            "Запрос пользователя:",
            value=current_query,
            height=100,
            placeholder="Введите текст запроса",
        )
        b_check, b_typo = st.columns(2)
        btn_check = b_check.button("Проверить запрос", type="primary", use_container_width=True)
        btn_typo = b_typo.button("Внести опечатку", use_container_width=True)

    with col_schema:
        schema_text = st.text_area(
            "Схема базы данных (DDL):",
            value=current_schema,
            height=100,
        )

    if btn_typo and user_query:
        user_query = apply_typo(user_query)
        st.info(f"Запрос с опечаткой: {user_query}")
        btn_check = True

    if btn_check:
        if not user_query.strip():
            st.warning("Введите текст запроса.")
            return
        if model is None or tokenizer is None:
            st.error("Файл модели best_model.pt не найден.")
            return

        codes_order = config.get("codes_order", CODES_ORDER)
        probs, latency_ms = run_prediction(model, tokenizer, user_query, selected_db, schema_text)
        result = postprocess_predictions(probs, codes_order, thresholds=threshold)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        status = result["ambiguity_status"]
        status_label = {"clear": "CLEAR (Понятный)", "incomplete": "INCOMPLETE (Неполный)", "ambiguous": "AMBIGUOUS (Неоднозначный)"}
        m1.metric("Статус", status_label.get(status, status))
        m2.metric("Оценка полноты", f"{result['completeness_score']:.2f} / 1.00")
        m3.metric("Дефектов найдено", len(result["detected_codes"]))
        m4.metric("Время анализа", f"{latency_ms:.2f} мс")

        res_col1, res_col2 = st.columns([1.1, 1.0])
        with res_col1:
            st.markdown("##### Вероятности по дефектам:")
            render_probability_chart(codes_order, probs, threshold)

        with res_col2:
            st.markdown("##### Выявленные замечания:")
            if result["detected_codes"]:
                for code, reason, clar in zip(
                    result["detected_codes"], result["reasons"], result["suggested_clarifications"]
                ):
                    with st.expander(code, expanded=True):
                        st.markdown(f"**Причина:** {reason}")
                        st.markdown(f"**Рекомендация:** {clar}")
            else:
                st.success("Запрос понятен и готов к генерации SQL.")


def main() -> None:
    st.title("NL2SQL Quality Gate — Панель тестирования")
    model, tokenizer, config = load_runtime_artifacts()
    tab_play, tab_metrics = st.tabs(["Проверка запроса", "Метрики и калибровка"])

    with tab_play:
        render_playground_tab(model, tokenizer, config)

    with tab_metrics:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
