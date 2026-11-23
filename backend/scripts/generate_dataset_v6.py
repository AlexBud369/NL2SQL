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
OUTPUT_DATASET = DATA_DIR / "dataset_v6.jsonl"
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
    "старые", "свежие", "крупные", "слабые", "активные", "важные", "недавние"
]

NONEXISTENT_TABLES = [
    "warehouses", "suppliers", "audit_logs", "discounts",
    "bonus_programs", "couriers", "reviews", "tariffs", "partners"
]

NONEXISTENT_COLUMNS = [
    "loyalty_level", "discount_percentage", "internal_rating",
    "tax_number", "bonus_points", "vip_status", "performance_score",
    "birthdate", "discount"
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
    "Назови столицу Австралии и ее население"
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


def parse_schema(schema_text: str) -> dict[str, list[tuple[str, str]]]:
    tables: dict[str, list[tuple[str, str]]] = {}
    for table_name, block in TABLE_RE.findall(schema_text):
        tables[table_name.lower()] = parse_columns(block)
    return tables


def load_schemas(schemas_path: Path) -> list[dict[str, Any]]:
    with open(schemas_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_ru_table_name(table_name: str) -> str:
    key = table_name.lower()
    if key in TABLE_TRANSLATIONS:
        return random.choice(TABLE_TRANSLATIONS[key])
    return table_name


def get_ru_col_name(col_name: str) -> str:
    key = col_name.lower()
    if key in COLUMN_TRANSLATIONS:
        return random.choice(COLUMN_TRANSLATIONS[key])
    return col_name


# ---------------------------------------------------------------------------
# CLEAR Generators (включая реляционные JOIN местоимения и валидные SELECT *)
# ---------------------------------------------------------------------------

def generate_unconditional_select(t_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи всех {t_ru}",
        f"Выведи список всех {t_ru} из базы",
        f"Предоставь информацию о {t_ru}",
        f"Выгрузи таблицу {t_ru}",
        f"Дай список всех записей в таблице {t_ru}",
        f"Выведи все {t_ru}",
    ]
    return "SELECT", random.choice(templates), []


def generate_counting_select(t_ru: str, c_ru: str, val_str: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Сколько {t_ru} зарегистрировано в системе?",
        f"Посчитай количество {t_ru}",
        f"Сколько всего {t_ru} есть в базе?",
        f"Сколько {t_ru} имеют {c_ru} '{val_str}'?",
        f"Какое общее число {t_ru}?",
    ]
    return "SELECT", random.choice(templates), []


def generate_conditional_select(t_ru: str, c_ru: str, val_str: str, val_id: int) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {t_ru} где {c_ru} равен '{val_str}'",
        f"Выведи {t_ru} с идентификатором {val_id}",
        f"Найди {t_ru} у которых {c_ru}='{val_str}'",
        f"Предоставь информацию о {t_ru} где идентификатор равен {val_id}",
        f"Найди в таблице {t_ru} строку с номером {val_id}",
    ]
    return "SELECT", random.choice(templates), []


def generate_bounded_recent_select(t_ru: str) -> tuple[str, str, list[str]]:
    n = random.choice([3, 5, 10, 20, 50])
    year = random.choice([2023, 2024])
    templates = [
        f"Покажи последние {n} {t_ru} за {year} год, отсортированных по дате создания",
        f"Выведи последние {n} записей {t_ru} за {year} год, сортировка по убыванию",
        f"Покажи топ-{n} {t_ru} по сумме за январь {year}",
        f"Выкати список {t_ru} со статусом 'активен', сортировка по имени, лимит {n}",
        f"Дай первые {n} {t_ru}, отсортированных по дате",
    ]
    return "SELECT", random.choice(templates), []


def generate_valid_join_pronoun(t_ru1: str, t_ru2: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {t_ru1} и их {t_ru2}",
        f"Выведи всех {t_ru1} и связанные с ними {t_ru2}",
        f"Предоставь список {t_ru1} вместе с их {t_ru2}",
        f"Найди {t_ru1} и покажи их {t_ru2} за текущий год",
        f"Выгрузи {t_ru1} и принадлежащие им {t_ru2}",
    ]
    return "SELECT", random.choice(templates), []


def generate_clear_modification(table: str, col: str, t_ru: str, c_ru: str, val_str: str, val_id: int) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.35:
        templates = [
            f"Добавь запись в {t_ru}: {col}='{val_str}', id={val_id}",
            f"Вставь новую строку в {t_ru} со значениями id={val_id} и {col}='{val_str}'",
            f"Создай нового {t_ru} с параметрами {col}='{val_str}', id={val_id}",
        ]
        return "INSERT", random.choice(templates), []
    if roll < 0.65:
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


def generate_clear_sample(table: str, col: str, val_str: str, val_id: int, other_table: str) -> tuple[str, str, list[str]]:
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    t_ru2 = get_ru_table_name(other_table)
    roll = random.random()
    if roll < 0.20:
        return generate_unconditional_select(t_ru)
    if roll < 0.35:
        return generate_counting_select(t_ru, c_ru, val_str)
    if roll < 0.55:
        return generate_conditional_select(t_ru, c_ru, val_str, val_id)
    if roll < 0.70:
        return generate_bounded_recent_select(t_ru)
    if roll < 0.85:
        return generate_valid_join_pronoun(t_ru, t_ru2)
    return generate_clear_modification(table, col, t_ru, c_ru, val_str, val_id)


# ---------------------------------------------------------------------------
# INCOMPLETE Generators (целевое исправление MISSING_VALUE, LIMIT, SORT)
# ---------------------------------------------------------------------------

def generate_missing_value_select(t_ru: str, c_ru: str) -> tuple[str, str, list[str]]:
    templates = [
        f"Покажи {t_ru} со статусом",
        f"Выведи {t_ru} где {c_ru} равен",
        f"Найди {t_ru} с {c_ru} больше",
        f"Выгрузи {t_ru} у которых {c_ru} меньше",
        f"Скинь список {t_ru} с {c_ru} от",
        f"Выведи {t_ru} где {c_ru} после",
        f"Покажи {t_ru} по категории",
        f"Отфильтруй {t_ru} со статусом",
    ]
    return "SELECT", random.choice(templates), ["MISSING_VALUE"]


def generate_single_incomplete(t_ru: str, c_ru: str, val_str: str) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.25:
        # MISSING_VALUE: оборванные условия фильтрации или неполная вставка
        if random.random() < 0.75:
            return generate_missing_value_select(t_ru, c_ru)
        opts = [f"Добавь нового {t_ru}", f"Вставь запись в таблицу {t_ru}", f"Создай {t_ru}"]
        return "INSERT", random.choice(opts), ["MISSING_VALUE"]

    if roll < 0.45:
        # MISSING_CRITERION: строго для деструктивных операций без WHERE или явного "отфильтруй"
        opts = [
            f"Удали {t_ru}",
            f"Сотри все записи в таблице {t_ru}",
            f"Удали данные из {t_ru}",
            f"Обнови зарплаты сотрудников в {t_ru}",
            f"Обнови {t_ru} установив статус '{val_str}'",
            f"Отфильтруй {t_ru} по условию",
        ]
        return "UPDATE" if "Обнови" in opts[0] else "DELETE", random.choice(opts), ["MISSING_CRITERION"]

    if roll < 0.65:
        # MISSING_TIME: относительные временные метки без интервала
        opts = [
            f"Покажи {t_ru} за последний месяц",
            f"Выведи операции по {t_ru} за недавнее время",
            f"Скинь свежие {t_ru}",
            f"Покажи недавние {t_ru} из базы",
            f"Дай свежие события по {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_TIME"]

    if roll < 0.82:
        # MISSING_SORT: топ с цифрой, но без метрики сортировки
        n = random.choice([3, 5, 10])
        opts = [
            f"Покажи топ {n} {t_ru}",
            f"Выведи {n} лучших {t_ru}",
            f"Кто среди {t_ru} занимает первые {n} места",
            f"Дай рейтинг {n} {t_ru}",
            f"Выведи первые {n} {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_SORT"]

    # MISSING_LIMIT: направление сортировки есть, а цифры среза нет
    opts = [
        f"Выведи самые дорогие {t_ru}",
        f"Покажи максимальные {t_ru} по стоимости",
        f"Скинь самые дешевые {t_ru}",
        f"Дай наиболее прибыльные {t_ru}",
        f"Покажи {t_ru} с наибольшей ценой",
    ]
    return "SELECT", random.choice(opts), ["MISSING_LIMIT"]


def generate_combined_incomplete(t_ru: str) -> tuple[str, str, list[str]]:
    roll = random.random()
    if roll < 0.35:
        opts = [
            f"Выведи недавние {t_ru}",
            f"Покажи свежие {t_ru} из базы",
            f"Скинь последние изменения по {t_ru}",
            f"Дай недавние логи {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_TIME", "MISSING_LIMIT"]

    if roll < 0.70:
        # Комбинация превосходной степени без цифры: и лимит, и сортировка пропущены
        opts = [
            f"Кто наторговал больше всех среди {t_ru}?",
            f"Кто среди {t_ru} заработал больше всего?",
            f"Покажи лидеров среди {t_ru}",
            f"Выведи топ {t_ru}",
            f"Кто среди {t_ru} занимает первое место",
            f"Покажи рекордсменов по {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["MISSING_LIMIT", "MISSING_SORT"]

    opts = [
        f"Выведи рекордные {t_ru} за прошлый месяц",
        f"Покажи пиковые значения {t_ru} за недавние дни",
        f"Кто лидировал среди {t_ru} за последний период",
    ]
    return "SELECT", random.choice(opts), ["MISSING_TIME", "MISSING_LIMIT", "MISSING_SORT"]


def generate_incomplete_sample(table: str, col: str, val_str: str) -> tuple[str, str, list[str]]:
    t_ru, c_ru = get_ru_table_name(table), get_ru_col_name(col)
    if random.random() < 0.35:
        return generate_combined_incomplete(t_ru)
    return generate_single_incomplete(t_ru, c_ru, val_str)


# ---------------------------------------------------------------------------
# AMBIGUOUS Generators (VAGUE, ATTACH без контекста, UNKNOWN, SCOPE, AGG)
# ---------------------------------------------------------------------------

def generate_ambiguous_sample(table: str, col: str, val_id: int) -> tuple[str, str, list[str]]:
    roll = random.random()
    t_ru = get_ru_table_name(table)
    if roll < 0.20:
        fake_table = random.choice(NONEXISTENT_TABLES)
        opts = [
            f"Покажи список из таблицы {fake_table} где id={val_id}",
            f"Выгрузи все записи из несуществующего реестра {fake_table}",
            f"Найди в таблице {fake_table} данные по коду {val_id}",
            f"Покажи данные из таблицы {fake_table}",
        ]
        return "SELECT", random.choice(opts), ["UNKNOWN_TABLE"]

    if roll < 0.40:
        fake_col = random.choice(NONEXISTENT_COLUMNS)
        opts = [
            f"Выведи {t_ru} отсортированных по {get_ru_col_name(fake_col)}",
            f"Отфильтруй {t_ru} по полю {get_ru_col_name(fake_col)} равному 100",
            f"Покажи поле {fake_col} из таблицы {t_ru}",
            f"Выведи поле {fake_col} для всех {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["UNKNOWN_COLUMN"]

    if roll < 0.60:
        w = random.choice(VAGUE_WORDS)
        opts = [
            f"Покажи {w} {t_ru}",
            f"Выгрузи список наиболее {w} {t_ru}",
            f"Найди {w} позиции среди {t_ru}",
            f"Покажи самых {w} {t_ru}",
            f"Выведи {t_ru} с высокой зарплатой",
        ]
        return "SELECT", random.choice(opts), ["VAGUE"]

    if roll < 0.75:
        return "SELECT", random.choice(OUT_OF_SCOPE_QUERIES), ["SCOPE"]

    if roll < 0.90:
        # ATTACH: строго изолированные местоимения БЕЗ упоминания сущности в том же предложении
        opts = [
            f"Покажи их {t_ru}",
            f"Выведи его последние {t_ru}",
            f"Удали их из таблицы {t_ru}",
            f"Отфильтруй их по сумме",
            f"Покажи данные по нему",
            f"Найди этот {t_ru}",
        ]
        return "SELECT", random.choice(opts), ["ATTACH"]

    # MISSING_AGG: путаница скалярного итога и списка
    opts = [
        f"Посчитай {t_ru}",
        f"Сколько составляют {t_ru}",
        f"Суммируй {t_ru}",
        f"Выведи среднее по {t_ru}",
    ]
    return "SELECT", random.choice(opts), ["MISSING_AGG"]


# ---------------------------------------------------------------------------
# Основной пайплайн генерации датасета v6
# ---------------------------------------------------------------------------

def generate_dataset_v6(target_size: int = 15000) -> None:
    random.seed(42)
    schemas_data = load_schemas(SCHEMAS_FILE)
    print(f"Загружено схем: {len(schemas_data)}. Генерация датасета v6 на {target_size} примеров...")

    samples: list[dict[str, Any]] = []

    # Распределение баланса классов:
    # ~35% CLEAR, ~35% INCOMPLETE, ~30% AMBIGUOUS
    p_clear = 0.35
    p_incomplete = 0.35

    id_counter = 1
    while len(samples) < target_size:
        schema_entry = random.choice(schemas_data)
        schema_text = schema_entry.get("db_schema") or schema_entry.get("schema", "")
        db_type = schema_entry.get("db_type", "sqlite")
        tables_dict = parse_schema(schema_text)

        if not tables_dict:
            continue

        table_name = random.choice(list(tables_dict.keys()))
        columns = tables_dict[table_name]
        col_name = columns[0][0] if columns else "id"
        other_table = random.choice(list(tables_dict.keys()))

        val_str = random.choice(STATUSES)
        val_id = random.randint(1, 999)

        dice = random.random()
        if dice < p_clear:
            op, q, codes = generate_clear_sample(table_name, col_name, val_str, val_id, other_table)
        elif dice < p_clear + p_incomplete:
            op, q, codes = generate_incomplete_sample(table_name, col_name, val_str)
        else:
            op, q, codes = generate_ambiguous_sample(table_name, col_name, val_id)

        status = derive_clarity(codes)
        score = compute_completeness_score(codes)
        reasons = [REASON_TEMPLATES[c] for c in codes if c in REASON_TEMPLATES]
        clarifications = [CLARIFICATION_TEMPLATES[c] for c in codes if c in CLARIFICATION_TEMPLATES]

        item = {
            "id": f"gen-v6-{id_counter:05d}",
            "question": q,
            "operation": op,
            "schema": schema_text,
            "db_schema": schema_text,
            "db_type": db_type,
            "clarity": status,
            "expected_status": status,
            "codes": codes,
            "expected_codes": codes,
            "completeness_score": score,
            "reasons": reasons,
            "suggested_clarifications": clarifications,
        }
        samples.append(item)
        id_counter += 1

    random.shuffle(samples)

    # Сохраняем полный dataset_v6.jsonl
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Полный датасет сохранен в {OUTPUT_DATASET} ({len(samples)} строк)")

    # Сплиты: 11000 train, 2000 val, 2000 test
    train_split = samples[:11000]
    val_split = samples[11000:13000]
    test_split = samples[13000:]

    for path, split_data, name in [
        (OUTPUT_TRAIN, train_split, "Train"),
        (OUTPUT_VAL, val_split, "Val"),
        (OUTPUT_TEST, test_split, "Test"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name} сплит сохранен в {path} ({len(split_data)} строк)")

    print("Генерация датасета v6 успешно завершена!")


if __name__ == "__main__":
    generate_dataset_v6(15000)
