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
OUTPUT_DATASET = DATA_DIR / "dataset_v7.jsonl"
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
    "events": ("события", "логи", "записи активности"),
    "sales": ("продажи", "сделки", "торговые операции"),
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
    "created_at": ("дата создания", "время добавления"),
    "sale_date": ("дата продажи", "день сделки"),
    "hire_date": ("дата найма", "день приема на работу"),
}

VAGUE_WORDS = [
    "лучшие", "популярные", "хорошие", "выгодные", "высокие",
    "низкие", "дешевые", "дорогие", "опытные", "надежные",
    "эффективные", "прибыльные", "успешные", "ключевые", "перспективные",
    "старые", "свежие", "крупные", "слабые", "активные", "важные", "недавние",
    "нормальные", "стандартные", "адекватные",
]

NONEXISTENT_TABLES = [
    "warehouses", "suppliers", "audit_logs", "discounts",
    "bonus_programs", "couriers", "reviews", "tariffs", "partners", "subscriptions",
]

NONEXISTENT_COLUMNS = [
    "loyalty_level", "discount_percentage", "internal_rating",
    "tax_number", "bonus_points", "vip_status", "performance_score",
    "birthdate", "discount", "margin", "delivery_cost",
]

STATUSES = ["активен", "завершен", "в обработке", "отменен", "новый", "архивирован", "оплачен"]

OUT_OF_SCOPE_QUERIES = [
    "Какая погода в Москве завтра?",
    "Напиши рецепт приготовления пиццы с грибами",
    "Как настроить роутер для домашнего интернета?",
    "Кто выиграл вчерашний футбольный матч?",
    "Сочини поздравительное стихотворение коллеге",
    "Переведи этот абзац на немецкий язык",
    "Какой сегодня курс биткоина к евро?",
    "Расскажи биографию Альберта Эйнштейна",
    "Как починить протекающий кран в ванной?",
    "Сколько градусов на улице в Санкт-Петербурге?",
    "Сколько будет два плюс два?",
    "Посоветуй хороший научно-фантастический фильм",
    "Когда началось восстание декабристов в России?",
    "Как правильно варить гречневую кашу на молоке?",
    "Назови столицу Австралии и ее население",
]

CONSTRAINT_PREFIXES = ("FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "KEY ")


def parse_columns(block: str) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for raw_column in block.split(","):
        cleaned = raw_column.strip()
        if not cleaned or cleaned.upper().startswith(CONSTRAINT_PREFIXES):
            continue
        parts = cleaned.split()
        column_name = parts[0].strip('`"')
        column_type = parts[1].upper() if len(parts) > 1 else "VARCHAR"
        columns.append((column_name, column_type))
    return columns


def parse_schema(schema_text: str) -> dict[str, list[tuple[str, str]]]:
    return {
        table_name.lower(): parse_columns(block)
        for table_name, block in TABLE_RE.findall(schema_text)
    }


def load_schemas(schemas_path: Path) -> list[dict[str, Any]]:
    with open(schemas_path, "r", encoding="utf-8") as schemas_file:
        return [json.loads(line) for line in schemas_file if line.strip()]


def translate_name(name: str, dictionary: dict[str, tuple[str, ...]]) -> str:
    key = name.lower()
    if key in dictionary:
        return random.choice(dictionary[key])
    return name


def generate_unconditional_select(table_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи всех {table_ru}",
        f"Выведи список всех {table_ru} из базы",
        f"Предоставь информацию о {table_ru}",
        f"Выгрузи таблицу {table_ru}",
        f"Дай список всех записей в таблице {table_ru}",
        f"Выведи все {table_ru}",
        f"Покажи данные по {table_ru}",
        f"Выгрузи базу {table_ru} полностью",
    ]
    return "SELECT", random.choice(templates), []


def generate_counting_select(table_ru: str, column_ru: str, value_str: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Сколько {table_ru} зарегистрировано в системе?",
        f"Посчитай количество {table_ru}",
        f"Сколько всего {table_ru} есть в базе?",
        f"Сколько {table_ru} имеют {column_ru} '{value_str}'?",
        f"Какое общее число {table_ru}?",
    ]
    return "SELECT", random.choice(templates), []


def generate_conditional_select(
    table_ru: str, column_ru: str, value_str: str, value_id: int
) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {table_ru} где {column_ru} равен '{value_str}'",
        f"Выведи {table_ru} с идентификатором {value_id}",
        f"Найди {table_ru} у которых {column_ru}='{value_str}'",
        f"Предоставь информацию о {table_ru} где идентификатор равен {value_id}",
        f"Найди в таблице {table_ru} строку с номером {value_id}",
        f"Покажи заказы клиента {value_id}",
    ]
    return "SELECT", random.choice(templates), []


def generate_bounded_recent_select(table_ru: str) -> tuple[str, str, list[str]]:
    top_limit = random.choice([3, 5, 10, 20, 50])
    year_value = random.choice([2023, 2024])
    templates = [
        f"Покажи последние {top_limit} {table_ru} за {year_value} год, отсортированных по дате создания",
        f"Выведи последние {top_limit} записей {table_ru} за {year_value} год, сортировка по убыванию",
        f"Покажи топ-{top_limit} {table_ru} по сумме за январь {year_value}",
        f"Выкати список {table_ru} со статусом 'активен', сортировка по имени, лимит {top_limit}",
        f"Дай первые {top_limit} {table_ru}, отсортированных по дате",
    ]
    return "SELECT", random.choice(templates), []


def generate_valid_join_pronoun(first_table_ru: str, second_table_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {first_table_ru} и их {second_table_ru}",
        f"Выведи всех {first_table_ru} и связанные с ними {second_table_ru}",
        f"Предоставь список {first_table_ru} вместе с их {second_table_ru}",
        f"Найди {first_table_ru} и покажи их {second_table_ru} за текущий год",
        f"Выгрузи {first_table_ru} и принадлежащие им {second_table_ru}",
    ]
    return "SELECT", random.choice(templates), []


def generate_valid_grouping_select(table_ru: str, column_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Сгруппируй {table_ru} по {column_ru} и посчитай количество",
        f"Посчитай сумму по {column_ru} для каждого {table_ru}",
        "Сгруппируй платежи по месяцам и рассчитай итог по сумме",
        "Посчитай общую сумму зарплат по отделам сотрудников",
    ]
    return "SELECT", random.choice(templates), []


def generate_insert_modification(column: str, table_ru: str, value_str: str, value_id: int) -> tuple[str, str, list[str]]:
    templates = [
        f"Добавь запись в {table_ru}: {column}='{value_str}', id={value_id}",
        f"Вставь новую строку в {table_ru} со значениями id={value_id} и {column}='{value_str}'",
        f"Создай нового {table_ru} с параметрами {column}='{value_str}', id={value_id}",
    ]
    return "INSERT", random.choice(templates), []


def generate_update_or_delete_modification(
    table_ru: str, column_ru: str, value_str: str, value_id: int
) -> tuple[str, str, list[str]]:
    if random.random() < 0.50:
        templates = [
            f"Обнови {column_ru} на '{value_str}' в {table_ru} где id={value_id}",
            f"Измени значение {column_ru} на '{value_str}' для записи {table_ru} с номером {value_id}",
            f"Установи {column_ru}='{value_str}' в таблице {table_ru} по ID {value_id}",
        ]
        return "UPDATE", random.choice(templates), []
    templates = [
        f"Удали запись из таблицы {table_ru} с ID {value_id}",
        f"Удали строку из {table_ru} где идентификатор равен {value_id}",
        f"Сотри {table_ru} с номером {value_id}",
    ]
    return "DELETE", random.choice(templates), []


def generate_clear_modification(
    column: str, table_ru: str, column_ru: str, value_str: str, value_id: int
) -> tuple[str, str, list[str]]:
    if random.random() < 0.35:
        return generate_insert_modification(column, table_ru, value_str, value_id)
    return generate_update_or_delete_modification(table_ru, column_ru, value_str, value_id)


def generate_clear_sample(
    table: str, column: str, value_str: str, value_id: int, other_table: str
) -> tuple[str, str, list[str]]:
    table_ru = translate_name(table, TABLE_TRANSLATIONS)
    column_ru = translate_name(column, COLUMN_TRANSLATIONS)
    other_table_ru = translate_name(other_table, TABLE_TRANSLATIONS)
    roll = random.random()
    if roll < 0.20:
        return generate_unconditional_select(table_ru)
    if roll < 0.35:
        return generate_counting_select(table_ru, column_ru, value_str)
    if roll < 0.50:
        return generate_conditional_select(table_ru, column_ru, value_str, value_id)
    if roll < 0.65:
        return generate_bounded_recent_select(table_ru)
    if roll < 0.80:
        return generate_valid_join_pronoun(table_ru, other_table_ru)
    if roll < 0.90:
        return generate_valid_grouping_select(table_ru, column_ru)
    return generate_clear_modification(column, table_ru, column_ru, value_str, value_id)


def generate_missing_value_sample(table_ru: str, column_ru: str) -> tuple[str, str, list[str]]:
    if random.random() < 0.75:
        templates = [
            f"Покажи {table_ru} со статусом",
            f"Выведи {table_ru} где {column_ru} равен",
            f"Найди {table_ru} с {column_ru} больше",
            f"Выгрузи {table_ru} у которых {column_ru} меньше",
            f"Скинь список {table_ru} с {column_ru} от",
            f"Выведи {table_ru} где {column_ru} после",
            f"Покажи {table_ru} по категории",
            f"Отфильтруй {table_ru} со статусом",
        ]
        return "SELECT", random.choice(templates), ["MISSING_VALUE"]
    insert_templates = [f"Добавь нового {table_ru}", f"Вставь запись в таблицу {table_ru}", f"Создай {table_ru}"]
    return "INSERT", random.choice(insert_templates), ["MISSING_VALUE"]


def generate_missing_criterion_sample(table_ru: str, value_str: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Удали {table_ru}",
        f"Сотри все записи в таблице {table_ru}",
        f"Удали данные из {table_ru}",
        f"Обнови зарплаты сотрудников в {table_ru}",
        f"Обнови {table_ru} установив статус '{value_str}'",
        f"Отфильтруй {table_ru} по условию",
        f"Очисти таблицу {table_ru}",
    ]
    selected_text = random.choice(templates)
    operation = "UPDATE" if "Обнови" in selected_text else "DELETE"
    return operation, selected_text, ["MISSING_CRITERION"]


def generate_missing_time_sample(table_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {table_ru} за последний месяц",
        f"Выведи операции по {table_ru} за недавнее время",
        f"Скинь свежие {table_ru}",
        f"Покажи недавние {table_ru} из базы",
        f"Дай свежие события по {table_ru}",
    ]
    return "SELECT", random.choice(templates), ["MISSING_TIME"]


def generate_ranking_spec_sample(table_ru: str) -> tuple[str, str, list[str]]:
    top_limit = random.choice([3, 5, 10])
    templates = [
        f"Покажи топ {top_limit} {table_ru}",
        f"Выведи {top_limit} лучших {table_ru}",
        f"Кто среди {table_ru} занимает первые {top_limit} места",
        f"Дай рейтинг {top_limit} {table_ru}",
        f"Выведи самые дорогие {table_ru}",
        f"Покажи максимальные {table_ru} по стоимости",
        f"Скинь наиболее прибыльные {table_ru}",
        f"Выгрузи верхушку списка {table_ru}",
        f"Кто наторговал больше всех среди {table_ru}?",
        f"Выведи топ {table_ru}",
    ]
    return "SELECT", random.choice(templates), ["RANKING_SPEC"]


def generate_combined_incomplete(table_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Выведи недавние {table_ru}",
        f"Покажи свежие {table_ru} из базы",
        f"Скинь последние изменения по {table_ru}",
        f"Дай недавние логи {table_ru}",
        f"Выведи рекордные {table_ru} за прошлый месяц",
        f"Покажи пиковые значения {table_ru} за недавние дни",
    ]
    return "SELECT", random.choice(templates), ["MISSING_TIME", "RANKING_SPEC"]


def generate_incomplete_sample(table: str, column: str, value_str: str) -> tuple[str, str, list[str]]:
    table_ru = translate_name(table, TABLE_TRANSLATIONS)
    column_ru = translate_name(column, COLUMN_TRANSLATIONS)
    if random.random() < 0.25:
        return generate_combined_incomplete(table_ru)
    roll = random.random()
    if roll < 0.25:
        return generate_missing_value_sample(table_ru, column_ru)
    if roll < 0.50:
        return generate_missing_criterion_sample(table_ru, value_str)
    if roll < 0.75:
        return generate_missing_time_sample(table_ru)
    return generate_ranking_spec_sample(table_ru)


def generate_unknown_table_sample(value_id: int) -> tuple[str, str, list[str]]:
    fake_table = random.choice(NONEXISTENT_TABLES)
    templates = [
        f"Покажи список из таблицы {fake_table} где id={value_id}",
        f"Выгрузи все записи из несуществующего реестра {fake_table}",
        f"Найди в таблице {fake_table} данные по коду {value_id}",
        f"Покажи данные из таблицы {fake_table}",
    ]
    return "SELECT", random.choice(templates), ["UNKNOWN_TABLE"]


def generate_unknown_column_sample(table_ru: str) -> tuple[str, str, list[str]]:
    fake_column = random.choice(NONEXISTENT_COLUMNS)
    fake_column_ru = translate_name(fake_column, COLUMN_TRANSLATIONS)
    templates = [
        f"Выведи {table_ru} отсортированных по {fake_column_ru}",
        f"Отфильтруй {table_ru} по полю {fake_column_ru} равному 100",
        f"Покажи поле {fake_column} из таблицы {table_ru}",
        f"Выведи поле {fake_column} для всех {table_ru}",
    ]
    return "SELECT", random.choice(templates), ["UNKNOWN_COLUMN"]


def generate_vague_sample(table_ru: str) -> tuple[str, str, list[str]]:
    vague_word = random.choice(VAGUE_WORDS)
    templates = [
        f"Покажи {vague_word} {table_ru}",
        f"Выгрузи список наиболее {vague_word} {table_ru}",
        f"Найди {vague_word} позиции среди {table_ru}",
        f"Покажи самых {vague_word} {table_ru}",
        f"Выведи {table_ru} с высокой зарплатой",
        f"Сделай общую сводку и аналитику по {table_ru}",
        f"Выведи статистику по {table_ru}",
        f"Посчитай показатели эффективности по {table_ru}",
    ]
    return "SELECT", random.choice(templates), ["VAGUE"]


def generate_attach_sample(table_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи их {table_ru}",
        f"Выведи его последние {table_ru}",
        f"Удали их из таблицы {table_ru}",
        "Отфильтруй их по сумме",
        "Покажи данные по нему",
        f"Найди этот {table_ru}",
    ]
    return "SELECT", random.choice(templates), ["ATTACH"]


def generate_ambiguous_sample(table: str, value_id: int) -> tuple[str, str, list[str]]:
    table_ru = translate_name(table, TABLE_TRANSLATIONS)
    roll = random.random()
    if roll < 0.22:
        return generate_unknown_table_sample(value_id)
    if roll < 0.44:
        return generate_unknown_column_sample(table_ru)
    if roll < 0.70:
        return generate_vague_sample(table_ru)
    if roll < 0.85:
        return "SELECT", random.choice(OUT_OF_SCOPE_QUERIES), ["SCOPE"]
    return generate_attach_sample(table_ru)


def build_sample_record(
    sample_id: int, operation: str, question: str, schema_text: str, db_type: str, codes: list[str]
) -> dict[str, Any]:
    status = derive_clarity(codes)
    return {
        "id": f"gen-v7-{sample_id:05d}", "question": question, "operation": operation,
        "schema": schema_text, "db_schema": schema_text, "db_type": db_type,
        "clarity": status, "expected_status": status, "codes": codes, "expected_codes": codes,
        "completeness_score": compute_completeness_score(codes),
        "reasons": [REASON_TEMPLATES[code] for code in codes if code in REASON_TEMPLATES],
        "suggested_clarifications": [CLARIFICATION_TEMPLATES[code] for code in codes if code in CLARIFICATION_TEMPLATES],
    }


def pick_query_by_clarity(
    table_name: str, column_name: str, value_str: str, value_id: int, other_table: str
) -> tuple[str, str, list[str]]:
    dice = random.random()
    if dice < 0.35:
        return generate_clear_sample(table_name, column_name, value_str, value_id, other_table)
    if dice < 0.70:
        return generate_incomplete_sample(table_name, column_name, value_str)
    return generate_ambiguous_sample(table_name, value_id)


def generate_single_record(schemas_data: list[dict[str, Any]], sample_id: int) -> dict[str, Any] | None:
    schema_entry = random.choice(schemas_data)
    schema_text = schema_entry.get("db_schema") or schema_entry.get("schema", "")
    tables_dict = parse_schema(schema_text)
    if not tables_dict:
        return None
    table_name = random.choice(list(tables_dict.keys()))
    columns = tables_dict[table_name]
    column_name = columns[0][0] if columns else "id"
    other_table = random.choice(list(tables_dict.keys()))
    operation, question, codes = pick_query_by_clarity(
        table_name, column_name, random.choice(STATUSES), random.randint(1, 999), other_table
    )
    return build_sample_record(sample_id, operation, question, schema_text, schema_entry.get("db_type", "sqlite"), codes)


def write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset_v7(target_size: int = 15000) -> None:
    random.seed(42)
    schemas_data = load_schemas(SCHEMAS_FILE)
    print(f"Загружено схем: {len(schemas_data)}. Генерация датасета v7 на {target_size} примеров...")

    samples: list[dict[str, Any]] = []
    sample_id = 1
    while len(samples) < target_size:
        record = generate_single_record(schemas_data, sample_id)
        if record is not None:
            samples.append(record)
            sample_id += 1

    random.shuffle(samples)
    write_jsonl_file(OUTPUT_DATASET, samples)
    write_jsonl_file(OUTPUT_TRAIN, samples[:11000])
    write_jsonl_file(OUTPUT_VAL, samples[11000:13000])
    write_jsonl_file(OUTPUT_TEST, samples[13000:])
    print(f"Датасет v7 ({len(samples)} строк) и сплиты успешно сохранены.")


if __name__ == "__main__":
    generate_dataset_v7(15000)
