import json
from pathlib import Path

SCHEMAS_PATH = Path("data/schemas.jsonl")
QUESTIONS_PATH = Path("data/questions.jsonl")
EXISTING_PATH = Path("data/al_data_ru.jsonl")
OUTPUT_PATH = Path("data/all_data.jsonl")

AMBIGUOUS_CODES = {
    "VAGUE",
    "SCOPE",
    "ATTACH",
    "UNKNOWN_TABLE",
    "UNKNOWN_COLUMN",
}

MISSING_CODES = {
    "MISSING_TIME",
    "MISSING_CRITERION",
    "MISSING_SORT",
    "MISSING_AGG",
    "MISSING_LIMIT",
}

PENALTIES = {
    "MISSING_TIME": 0.35,
    "MISSING_CRITERION": 0.25,
    "MISSING_SORT": 0.20,
    "MISSING_AGG": 0.25,
    "MISSING_LIMIT": 0.15,
    "VAGUE": 0.30,
    "SCOPE": 0.35,
    "ATTACH": 0.30,
    "UNKNOWN_TABLE": 0.50,
    "UNKNOWN_COLUMN": 0.40,
}

REASON_TEMPLATES = {
    "MISSING_TIME": "Не указан период времени",
    "MISSING_CRITERION": "Не указано, по какому признаку фильтровать",
    "MISSING_SORT": "Не указано, по какому полю сортировать",
    "MISSING_AGG": "Не указана агрегация (сумма, среднее или количество)",
    "MISSING_LIMIT": "Не указано количество строк в выборке",
    "VAGUE": "Есть размытое слово, требующее уточнения",
    "SCOPE": "Непонятно, какие сущности нужно соединить",
    "ATTACH": "Непонятно, к чему относится уточнение",
    "UNKNOWN_TABLE": "Название таблицы можно понять по-разному",
    "UNKNOWN_COLUMN": "Название колонки можно понять по-разному",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Файл {path} не найден, пропускаю")
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_schemas(path: Path) -> dict[str, str]:
    schemas = load_jsonl(path)
    return {item["name"]: item["db_schema"] for item in schemas}


def compute_label_status(codes: list[str]) -> str:
    if not codes:
        return "clear"
    if any(code in AMBIGUOUS_CODES for code in codes):
        return "ambiguous"
    if any(code in MISSING_CODES for code in codes):
        return "incomplete"
    return "clear"


def compute_completeness_score(codes: list[str]) -> float:
    score = 1.0
    for code in codes:
        score -= PENALTIES.get(code, 0.0)
    return max(0.0, min(1.0, round(score, 2)))


def compute_reasons(codes: list[str]) -> list[str]:
    return [REASON_TEMPLATES[code] for code in codes if code in REASON_TEMPLATES]


def enrich(item: dict, schemas: dict[str, str]) -> dict | None:
    schema_name = item.pop("schema_name", None)
    if schema_name is None:
        return None

    if schema_name not in schemas:
        print(f"Не найдена схема для {schema_name}, пропускаю: {item['question']}")
        return None

    codes = item.get("missing_dimensions", [])
    item["db_schema"] = schemas[schema_name]
    item["label_status"] = compute_label_status(codes)
    item["completeness_score"] = compute_completeness_score(codes)
    item["reasons"] = compute_reasons(codes)

    ordered = {
        "question": item["question"],
        "db_type": item["db_type"],
        "db_schema": item["db_schema"],
        "label_status": item["label_status"],
        "completeness_score": item["completeness_score"],
        "missing_dimensions": item["missing_dimensions"],
        "reasons": item["reasons"],
    }
    return ordered


def main():
    schemas = load_schemas(SCHEMAS_PATH)
    print(f"Загружено схем: {len(schemas)}")

    questions = load_jsonl(QUESTIONS_PATH)
    print(f"Загружено новых вопросов: {len(questions)}")

    enriched = []
    for item in questions:
        result = enrich(item, schemas)
        if result is not None:
            enriched.append(result)

    print(f"Обогащено новых примеров: {len(enriched)}")

    existing = load_jsonl(EXISTING_PATH)
    print(f"Загружено существующих примеров: {len(existing)}")

    combined = existing + enriched

    seen = set()
    unique = []
    for item in combined:
        key = item.get("question", "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    print(f"После удаления дублей: {len(unique)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Результат сохранён в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
