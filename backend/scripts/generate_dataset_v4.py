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

from src.constants import CLARIFICATION_TEMPLATES, REASON_TEMPLATES
from src.postprocess import compute_completeness_score, derive_clarity

DATA_DIR = PROJECT_ROOT / "data"
SCHEMAS_FILE = DATA_DIR / "schemas.jsonl"
OUTPUT_DATASET = DATA_DIR / "dataset_v4.jsonl"
OUTPUT_TRAIN = DATA_DIR / "train.jsonl"
OUTPUT_VAL = DATA_DIR / "val.jsonl"
OUTPUT_TEST = DATA_DIR / "test.jsonl"

TABLE_RE = re.compile(r"TABLE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)(?=\s*\|\s*TABLE\s|$)", re.I)

TABLE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "airlines": ("авиакомпании", "авиалинии", "перевозчики"),
    "airports": ("аэропорты", "аэродромы", "терминалы"),
    "aircraft": ("самолеты", "воздушные суда", "лайнеры"),
    "routes": ("маршруты", "направления", "рейсовые пути"),
    "flights": ("рейсы", "полеты", "вылеты"),
    "passengers": ("пассажиры", "клиенты", "путешественники"),
    "bookings": ("бронирования", "брони", "заказы билетов"),
    "tickets": ("билеты", "посадочные талоны"),
    "customers": ("клиенты", "покупатели", "заказчики", "пользователи"),
    "orders": ("заказы", "покупки", "сделки"),
    "products": ("товары", "продукты", "позиции", "номенклатура"),
    "categories": ("категории", "разделы", "группы товаров"),
    "doctors": ("врачи", "доктора", "специалисты", "медики"),
    "patients": ("пациенты", "больные", "посетители клиники"),
    "appointments": ("приемы", "записи", "визиты к врачу"),
    "departments": ("отделы", "подразделения", "департаменты"),
    "employees": ("сотрудники", "работники", "персонал", "штат"),
    "salaries": ("зарплаты", "оклады", "выплаты сотрудникам"),
    "courses": ("курсы", "дисциплины", "учебные программы"),
    "students": ("студенты", "учащиеся", "слушатели"),
    "hotels": ("отели", "гостиницы", "хостелы"),
    "rooms": ("номера", "комнаты", "апартаменты"),
    "accounts": ("счета", "аккаунты", "профили пользователей"),
    "transactions": ("транзакции", "переводы", "платежи", "операции"),
    "cars": ("автомобили", "машины", "транспортные средства"),
    "warehouses": ("склады", "складские комплексы", "хранилища"),
    "users": ("пользователи", "клиенты", "учетные записи"),
}

COLUMN_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "id": ("ID", "идентификатор", "номер", "код"),
    "name": ("имя", "название", "наименование", "фио"),
    "price": ("цена", "стоимость", "тариф"),
    "amount": ("сумма", "размер платежа", "объем"),
    "salary": ("зарплата", "оклад", "ставка"),
    "date": ("дата", "время", "период"),
    "status": ("статус", "состояние", "текущий статус"),
    "rating": ("рейтинг", "оценка", "балл"),
    "city": ("город", "населенный пункт"),
    "quantity": ("количество", "число единиц", "остаток"),
}

VAGUE_WORDS = [
    "лучшие", "популярные", "хорошие", "выгодные", "высокие",
    "низкие", "дешевые", "дорогие", "опытные", "надежные",
    "эффективные", "прибыльные", "успешные", "ключевые", "перспективные"
]

NONEXISTENT_TABLES = [
    "warehouses", "suppliers", "audit_logs", "discounts",
    "bonus_programs", "couriers", "reviews", "tariffs", "partners"
]

NONEXISTENT_COLUMNS = [
    "loyalty_level", "discount_percentage", "internal_rating",
    "tax_number", "bonus_points", "vip_status", "performance_score"
]

STATUSES = ["активен", "завершен", "в обработке", "отменен", "новый", "архивирован", "оплачен"]

OUT_OF_SCOPE_QUERIES = [
    "Какая погода в Москве завтра?",
    "Сколько стоит доллар на бирже?",
    "Напиши рецепт приготовления пиццы с грибами",
    "Как настроить роутер для домашнего интернета?",
    "Кто выиграл вчерашний футбольный матч?",
    "Сочини поздравительное стихотворение коллеге",
    "Переведи этот абзац на немецкий язык",
    "Какой сегодня курс биткоина к евро?",
    "Расскажи биографию Альберта Эйнштейна",
    "Как починить протекающий кран в ванной?"
]


def parse_columns(block: str) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for raw in block.split(","):
        c = raw.strip()
        if not c or c.upper().startswith(("FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "KEY ")):
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


def inject_typo(text: str) -> str:
    if len(text) < 8 or random.random() > 0.15:
        return text
    idx = random.randint(3, len(text) - 4)
    choice = random.random()
    if choice < 0.4:
        return text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]
    if choice < 0.7:
        return text[:idx] + text[idx + 1:]
    return text[:idx] + text[idx] + text[idx:]


def generate_unconditional_select(t_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи всех {t_ru}",
        f"Выгрузи всю таблицу {t_ru}",
        f"Выведи полный список {t_ru}",
        f"Предоставь справочник всех {t_ru}",
        f"Дай все записи из таблицы {t_ru} без фильтрации",
        f"Выведи всех {t_ru} без ограничений",
        f"Покажи целиком весь каталог {t_ru}",
        f"Сформируй полный перечень {t_ru}",
        f"Выгрузи базу всех {t_ru}",
        f"Дай мне всю информацию по всем {t_ru}",
    ]
    return "SELECT", random.choice(templates), []


def generate_conditional_select(t_ru: str, c_ru: str, val_str: str, val_id: int) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {t_ru} где {c_ru} равен '{val_str}'",
        f"Найди {t_ru} с {c_ru} '{val_str}'",
        f"Выведи данные по {t_ru} у которых {c_ru} имеет значение '{val_str}'",
        f"Отфильтруй {t_ru} по условию {c_ru}='{val_str}'",
        f"Выгрузи запись из {t_ru} с ID {val_id}",
        f"Предоставь информацию о {t_ru} где идентификатор равен {val_id}",
        f"Найди в таблице {t_ru} строку с номером {val_id}",
    ]
    return "SELECT", random.choice(templates), []


def generate_clear_modification(table: str, col: str, t_ru: str, c_ru: str, val_str: str, val_id: int) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.50:
        templates = [
            f"Добавь запись в {t_ru}: {col}='{val_str}', id={val_id}",
            f"Вставь новую строку в {t_ru} со значениями id={val_id} и {col}='{val_str}'",
            f"Создай нового {t_ru} с параметрами {col}='{val_str}', id={val_id}",
        ]
        return "INSERT", random.choice(templates), []
    if roll < 0.80:
        templates = [
            f"Обнови {c_ru} на '{val_str}' в {t_ru} где id={val_id}",
            f"Измени значение {c_ru} на '{val_str}' для записи {t_ru} с номером {val_id}",
            f"Установи {c_ru}='{val_str}' в таблице {t_ru} по ID {val_id}",
        ]
        return "UPDATE", random.choice(templates), []
    templates = [
        f"Удали запись из таблицы {t_ru} с ID {val_id}",
        f"Удали строку из {t_ru} где идентификатор равен {val_id}",
        f"Сотри {t_ru} с номером {val_id}",
    ]
    return "DELETE", random.choice(templates), []


def generate_clear_sample(table: str, col: str, val_str: str, val_id: int) -> tuple[str, str, list[str]]:
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    roll = random.random()
    if roll < 0.35:
        return generate_unconditional_select(t_ru)
    if roll < 0.70:
        return generate_conditional_select(t_ru, c_ru, val_str, val_id)
    return generate_clear_modification(table, col, t_ru, c_ru, val_str, val_id)


def generate_single_incomplete(t_ru: str, c_ru: str, val_str: str) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.18:
        opts = [f"Добавь нового {t_ru}", f"Вставь запись в таблицу {t_ru}", f"Создай {t_ru}"]
        return "INSERT", random.choice(opts), ["MISSING_VALUE"]
    if roll < 0.36:
        opts = [
            f"Обнови статус всех {t_ru} на '{val_str}'",
            f"Измени данные в {t_ru} на '{val_str}'",
            f"Удали всех {t_ru} из базы",
            f"Сотри записи в таблице {t_ru}",
        ]
        return "UPDATE", random.choice(opts), ["MISSING_CRITERION"]
    if roll < 0.54:
        opts = [
            f"Покажи {t_ru} за последний месяц",
            f"Выведи операции по {t_ru} за недавнее время",
            f"Скинь свежие {t_ru}",
            f"Покажи данные по {t_ru} за прошлый период",
        ]
        return "SELECT", random.choice(opts), ["MISSING_TIME"]
    if roll < 0.72:
        opts = [
            f"Покажи топ 5 {t_ru}",
            f"Выведи 10 лучших {t_ru}",
            f"Кто среди {t_ru} занимает первые три места",
            f"Дай рейтинг пяти {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_SORT"]
    if roll < 0.88:
        opts = [
            f"Выведи самые дорогие {t_ru}",
            f"Покажи максимальные {t_ru} по стоимости",
            f"Скинь самые дешевые {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_LIMIT"]
    opts = [
        f"Посчитай {c_ru} по категориям в {t_ru}",
        f"Выведи аналитику по {c_ru} для каждого {t_ru}",
        f"Сгруппируй {t_ru} и дай статистику по полю {c_ru}",
    ]
    return "SELECT", random.choice(opts), ["MISSING_AGG"]


def generate_combined_incomplete(t_ru: str) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.35:
        opts = [
            f"Выведи недавние {t_ru}",
            f"Покажи свежие {t_ru} из базы",
            f"Скинь последние изменения по {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_TIME", "MISSING_LIMIT"]
    if roll < 0.70:
        opts = [
            f"Покажи топ {t_ru} за прошлый период",
            f"Выведи рейтинг {t_ru} за последнее время",
            f"Кто лидирует среди {t_ru} за недавний срок",
        ]
        return "SELECT", random.choice(opts), ["MISSING_TIME", "MISSING_SORT"]
    opts = [
        f"Выведи рекордные {t_ru} за прошлый месяц",
        f"Покажи пиковые значения {t_ru} за недавние дни",
    ]
    return "SELECT", random.choice(opts), ["MISSING_TIME", "MISSING_LIMIT", "MISSING_SORT"]


def generate_incomplete_sample(table: str, col: str, val_str: str) -> tuple[str, str, list[str]]:
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    if random.random() < 0.30:
        return generate_combined_incomplete(t_ru)
    return generate_single_incomplete(t_ru, c_ru, val_str)


def generate_ambiguous_sample(table: str, col: str, val_id: int) -> tuple[str, str, list[str]]:
    roll = random.random()
    t_ru = get_ru_table_name(table)
    if roll < 0.20:
        fake_table = random.choice(NONEXISTENT_TABLES)
        opts = [
            f"Покажи список из таблицы {fake_table} где id={val_id}",
            f"Выгрузи все записи из несуществующего реестра {fake_table}",
            f"Найди в таблице {fake_table} данные по коду {val_id}",
        ]
        return "SELECT", random.choice(opts), ["UNKNOWN_TABLE"]
    if roll < 0.40:
        fake_col = random.choice(NONEXISTENT_COLUMNS)
        opts = [
            f"Выведи {t_ru} отсортированных по {get_ru_col_name(fake_col)}",
            f"Отфильтруй {t_ru} по полю {get_ru_col_name(fake_col)} равному 100",
            f"Покажи {get_ru_col_name(fake_col)} для всех {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["UNKNOWN_COLUMN"]
    if roll < 0.65:
        w = random.choice(VAGUE_WORDS)
        opts = [
            f"Покажи {w} {t_ru}",
            f"Выгрузи список наиболее {w} {t_ru}",
            f"Найди {w} позиции среди {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["VAGUE"]
    if roll < 0.85:
        return "SELECT", random.choice(OUT_OF_SCOPE_QUERIES), ["SCOPE"]
    opts = [
        f"Покажи {t_ru} клиентов из Москвы со скидкой более 10%",
        f"Выведи {t_ru} врачей клиники с зарплатой выше среднего",
        f"Найди {t_ru} отделов компании с сотрудниками на испытательном сроке",
    ]
    return "SELECT", random.choice(opts), ["ATTACH"]


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
    final_question = inject_typo(question.strip())
    return {
        "id": f"qg-{index:06d}",
        "question": final_question,
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
        if category_roll < 0.33:
            intent, q, codes = generate_clear_sample(table, col_name, val_str, val_id)
        elif category_roll < 0.70:
            intent, q, codes = generate_incomplete_sample(table, col_name, val_str)
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
