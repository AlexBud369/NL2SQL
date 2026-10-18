import json
import random

SCHEMAS_FILE = "backend/data/schemas.jsonl"
OUTPUT_FILE = "backend/data/synthetic_queries.jsonl"

TEMPLATES = [
    # SELECT Clear (30%)
    {"q": "Покажи всех {table}", "clarity": "clear", "codes": []},
    {"q": "Выведи все записи из {table}", "clarity": "clear", "codes": []},
    {"q": "Найди все {table} с ID 5", "clarity": "clear", "codes": []},
    {"q": "Покажи {table} за 2024 год", "clarity": "clear", "codes": []},
    # SELECT Incomplete (10%)
    {"q": "Покажи {table}", "clarity": "incomplete", "codes": ["MISSING_CRITERION"]},
    {"q": "Выведи {table} недавно", "clarity": "incomplete", "codes": ["MISSING_TIME"]},
    {
        "q": "Покажи топ-10 {table}",
        "clarity": "incomplete",
        "codes": ["MISSING_SORT", "MISSING_LIMIT"],
    },
    # SELECT Ambiguous (10%)
    {"q": "Покажи активных {table}", "clarity": "ambiguous", "codes": ["VAGUE"]},
    {"q": "Покажи лучших {table}", "clarity": "ambiguous", "codes": ["VAGUE"]},
    {"q": "Покажи {table} и другие", "clarity": "ambiguous", "codes": ["SCOPE"]},
    # INSERT Clear (10%)
    {
        "q": "Добавь нового {table} с именем Иван и датой 2024-01-01",
        "clarity": "clear",
        "codes": [],
    },
    {"q": "Вставь запись в {table}: ID 1, статус active", "clarity": "clear", "codes": []},
    # INSERT Incomplete (10%)
    {"q": "Добавь нового {table}", "clarity": "incomplete", "codes": ["MISSING_VALUE"]},
    {
        "q": "Вставь запись в {table} без указания дат",
        "clarity": "incomplete",
        "codes": ["MISSING_VALUE"],
    },
    # UPDATE Clear (10%)
    {"q": "Обнови {table} с ID 1, установи статус active", "clarity": "clear", "codes": []},
    {"q": "Измени имя в {table} на Петр", "clarity": "clear", "codes": ["MISSING_WHERE"]},
    # UPDATE Incomplete (10%)
    {"q": "Обнови {table}", "clarity": "incomplete", "codes": ["MISSING_VALUE", "MISSING_WHERE"]},
    {
        "q": "Измени статус в {table} на 'завершен'",
        "clarity": "incomplete",
        "codes": ["MISSING_WHERE"],
    },
    # UPDATE Ambiguous (5%)
    {"q": "Обнови хорошие {table}", "clarity": "ambiguous", "codes": ["VAGUE"]},
    # DELETE Clear (10%)
    {"q": "Удали {table} с ID 5", "clarity": "clear", "codes": []},
    {"q": "Удали все {table}", "clarity": "clear", "codes": []},
    # DELETE Incomplete (10%)
    {"q": "Удали {table}", "clarity": "incomplete", "codes": ["MISSING_CRITERION"]},
    {"q": "Удали {table} из базы", "clarity": "incomplete", "codes": ["MISSING_CRITERION"]},
    # DELETE Ambiguous (5%)
    {"q": "Удали старые {table}", "clarity": "ambiguous", "codes": ["VAGUE"]},
]


def extract_tables(schema_text):
    tables = []
    parts = schema_text.split("|")
    for part in parts:
        part = part.strip()
        if part.startswith("TABLE "):
            table_name = part.split(":")[0].replace("TABLE ", "").strip()
            tables.append(table_name)
    return tables


def main():
    schemas = []
    with open(SCHEMAS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                schemas.append(json.loads(line))

    print(f"Загружено схем: {len(schemas)}")

    generated_data = []

    for schema in schemas:
        db_type = "postgresql"
        schema_text = schema["db_schema"]
        tables = extract_tables(schema_text)

        if not tables:
            continue

        for _ in range(4):
            template = random.choice(TEMPLATES)
            table = random.choice(tables)

            question = template["q"].format(table=table)

            record = {
                "question": question,
                "db_schema": schema_text,
                "db_type": db_type,
                "clarity": template["clarity"],
                "codes": template["codes"],
            }
            generated_data.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in generated_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Сгенерировано {len(generated_data)} запросов и сохранено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
