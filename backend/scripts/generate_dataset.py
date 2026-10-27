from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    CLARIFICATION_TEMPLATES,
    REASON_TEMPLATES,
)
from src.postprocess import compute_completeness_score, derive_clarity

DATA_DIR = PROJECT_ROOT / "data"
SCHEMAS_FILE = DATA_DIR / "schemas.jsonl"
OUTPUT_DATASET = DATA_DIR / "dataset_v3.jsonl"
OUTPUT_TRAIN = DATA_DIR / "train.jsonl"
OUTPUT_VAL = DATA_DIR / "val.jsonl"
OUTPUT_TEST = DATA_DIR / "test.jsonl"

TABLE_RE = re.compile(r"TABLE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)(?=\s*\|\s*TABLE\s|$)", re.I)

TABLE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "airlines": ("авиакомпании", "авиалинии"),
    "airports": ("аэропорты", "аэродромы"),
    "aircraft": ("самолеты", "воздушные суда"),
    "routes": ("маршруты", "направления"),
    "flights": ("рейсы", "полеты"),
    "passengers": ("пассажиры", "клиенты"),
    "bookings": ("бронирования", "брони"),
    "tickets": ("билеты", "посадочные талоны"),
    "customers": ("клиенты", "покупатели"),
    "orders": ("заказы", "покупки"),
    "products": ("товары", "продукты"),
    "categories": ("категории", "разделы"),
    "doctors": ("врачи", "доктора"),
    "patients": ("пациенты", "больные"),
    "appointments": ("приемы", "записи"),
    "departments": ("отделы", "подразделения"),
    "employees": ("сотрудники", "работники"),
    "salaries": ("зарплаты", "оклады"),
    "courses": ("курсы", "дисциплины"),
    "students": ("студенты", "учащиеся"),
    "hotels": ("отели", "гостиницы"),
    "rooms": ("номера", "комнаты"),
    "accounts": ("счета", "аккаунты"),
    "transactions": ("транзакции", "переводы"),
    "cars": ("автомобили", "машины"),
    "warehouses": ("склады", "складские комплексы"),
    "users": ("пользователи", "аккаунты"),
}

COLUMN_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "id": ("ID", "идентификатор", "номер"),
    "name": ("имя", "название"),
    "price": ("цена", "стоимость"),
    "amount": ("сумма", "размер"),
    "salary": ("зарплата", "оклад"),
    "date": ("дата",),
    "status": ("статус", "состояние"),
    "rating": ("рейтинг", "оценка"),
    "city": ("город",),
    "quantity": ("количество",),
}

VAGUE_WORDS = [
    "лучшие",
    "популярные",
    "хорошие",
    "выгодные",
    "высокие",
    "низкие",
    "дешевые",
    "дорогие",
    "опытные",
]
NONEXISTENT_TABLES = [
    "warehouses",
    "suppliers",
    "audit_logs",
    "discounts",
    "bonus_programs",
    "couriers",
]
NONEXISTENT_COLUMNS = [
    "loyalty_level",
    "discount_percentage",
    "internal_rating",
    "tax_number",
    "bonus_points",
]
STATUSES = ["активен", "завершен", "в обработке", "отменен", "новый", "архивирован", "оплачен"]


def parse_columns(block: str) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for raw in block.split(","):
        c = raw.strip()
        if not c or c.upper().startswith(
            ("FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "KEY ")
        ):
            continue
        parts = c.split()
        col_name = parts[0].strip('`"')
        col_type = parts[1].upper() if len(parts) > 1 else "VARCHAR"
        columns.append((col_name, col_type))
    return columns


def parse_schema_tables(schema_text: str) -> dict[str, list[tuple[str, str]]]:
    tables: dict[str, list[tuple[str, str]]] = {}
    for table_name, block in TABLE_RE.findall(schema_text):
        cols = parse_columns(block)
        if cols:
            tables[table_name] = cols
    return tables


def get_ru_table_name(table_name: str) -> str:
    cleaned = table_name.lower().rstrip("s")
    for key, variants in TABLE_TRANSLATIONS.items():
        if key in table_name.lower() or cleaned in key:
            return random.choice(variants)
    return table_name.replace("_", " ")


def get_ru_col_name(col_name: str) -> str:
    cleaned = col_name.lower()
    for key, variants in COLUMN_TRANSLATIONS.items():
        if key == cleaned or key in cleaned:
            return random.choice(variants)
    return col_name.replace("_", " ")


def generate_clear_sample(
    table: str, col: str, val_str: str, val_id: int
) -> tuple[str, str, list[str]]:
    roll = random.random()
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    if roll < 0.50:
        return "SELECT", f"Покажи {t_ru} где {c_ru} равен '{val_str}'", []
    if roll < 0.70:
        return "INSERT", f"Добавь запись в {t_ru}: {col}='{val_str}', id={val_id}", []
    if roll < 0.85:
        return "UPDATE", f"Обнови {c_ru} на '{val_str}' в {t_ru} где id={val_id}", []
    return "DELETE", f"Удали запись из таблицы {t_ru} с ID {val_id}", []


def generate_incomplete_sample(
    table: str, col: str, val_str: str, val_id: int
) -> tuple[str, str, list[str]]:
    roll = random.random()
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    if roll < 0.25:
        return "INSERT", f"Добавь нового {t_ru}", ["MISSING_VALUE"]
    if roll < 0.50:
        return "UPDATE", f"Обнови статус всех {t_ru} на '{val_str}'", ["MISSING_CRITERION"]
    if roll < 0.65:
        return "SELECT", f"Покажи {t_ru} за последний месяц", ["MISSING_TIME"]
    if roll < 0.80:
        return "SELECT", f"Покажи топ 5 {t_ru}", ["MISSING_SORT"]
    if roll < 0.92:
        return "SELECT", f"Выведи самые дорогие {t_ru}", ["MISSING_LIMIT"]
    return "SELECT", f"Посчитай {c_ru} по категориям в {t_ru}", ["MISSING_AGG"]


def generate_ambiguous_sample(table: str, col: str, val_id: int) -> tuple[str, str, list[str]]:
    roll = random.random()
    t_ru = get_ru_table_name(table)
    if roll < 0.25:
        fake_table = random.choice(NONEXISTENT_TABLES)
        return "SELECT", f"Покажи список из таблицы {fake_table} где id={val_id}", ["UNKNOWN_TABLE"]
    if roll < 0.50:
        fake_col = random.choice(NONEXISTENT_COLUMNS)
        return (
            "SELECT",
            f"Выведи {t_ru} отсортированных по {get_ru_col_name(fake_col)}",
            ["UNKNOWN_COLUMN"],
        )
    if roll < 0.75:
        return "SELECT", f"Покажи {random.choice(VAGUE_WORDS)} {t_ru}", ["VAGUE"]
    if roll < 0.90:
        return "SELECT", f"Выведи {t_ru} для каждой связанной сущности", ["SCOPE"]
    return "SELECT", f"Покажи {t_ru} клиентов из Москвы со скидкой более 10%", ["ATTACH"]


def create_record(
    index: int,
    question: str,
    schema: str,
    db_type: str,
    db_id: str,
    intent: str,
    codes: list[str],
    split: str,
) -> dict[str, Any]:
    clean_codes = sorted(list(set(codes)))
    return {
        "id": f"qg-{index:06d}",
        "question": question.strip(),
        "db_schema": schema,
        "db_type": db_type,
        "db_id": db_id,
        "intent": intent,
        "clarity": derive_clarity(clean_codes),
        "codes": clean_codes,
        "completeness_score": compute_completeness_score(clean_codes),
        "reasons": [REASON_TEMPLATES[c] for c in clean_codes if c in REASON_TEMPLATES],
        "suggested_clarifications": [
            CLARIFICATION_TEMPLATES[c] for c in clean_codes if c in CLARIFICATION_TEMPLATES
        ],
        "split": split,
    }


def split_schemas(
    schemas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(schemas)
    random.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.75)
    val_end = int(n * 0.90)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_records_for_pool(
    schema_pool: list[dict[str, Any]],
    split_name: str,
    target_count: int,
    start_index: int,
) -> list[dict[str, Any]]:
    records = []
    pool_size = len(schema_pool)
    for i in range(target_count):
        schema_entry = schema_pool[i % pool_size]
        parsed = parse_schema_tables(schema_entry["db_schema"])
        if not parsed:
            continue
        table = random.choice(list(parsed.keys()))
        col_name, _ = random.choice(parsed[table])
        val_str = random.choice(STATUSES)
        val_id = random.randint(1, 150)

        category_roll = random.random()
        if category_roll < 0.32:
            intent, q, codes = generate_clear_sample(table, col_name, val_str, val_id)
        elif category_roll < 0.70:
            intent, q, codes = generate_incomplete_sample(table, col_name, val_str, val_id)
        else:
            intent, q, codes = generate_ambiguous_sample(table, col_name, val_id)

        rec = create_record(
            start_index + i,
            q,
            schema_entry["db_schema"],
            schema_entry.get("db_type", "postgresql"),
            schema_entry["name"],
            intent,
            codes,
            split_name,
        )
        records.append(rec)
    return records


def main() -> None:
    random.seed(42)
    with open(SCHEMAS_FILE, "r", encoding="utf-8") as file:
        schemas = [json.loads(line) for line in file if line.strip()]

    train_schemas, val_schemas, test_schemas = split_schemas(schemas)
    train_data = generate_records_for_pool(train_schemas, "train", 7500, 1)
    val_data = generate_records_for_pool(val_schemas, "val", 1500, 7501)
    test_data = generate_records_for_pool(test_schemas, "test", 1000, 9001)

    all_data = train_data + val_data + test_data
    write_jsonl(OUTPUT_DATASET, all_data)
    write_jsonl(OUTPUT_TRAIN, train_data)
    write_jsonl(OUTPUT_VAL, val_data)
    write_jsonl(OUTPUT_TEST, test_data)


if __name__ == "__main__":
    main()
