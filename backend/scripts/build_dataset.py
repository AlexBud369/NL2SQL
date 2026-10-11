import json
import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/raw/ambrosia.csv")
OUT_PATH = Path("data/all_data.jsonl")
SAMPLES_PER_TYPE = 60
TARGET_TYPES = ["attachment", "scope", "vagueness"]
DEFAULT_DB_TYPE = "postgresql"


def extract_schema_from_dump(db_dump: str) -> str:
    pattern = r"CREATE\s+TABLE\s+[`\"]?(\w+)[`\"]?\s*\((.*?)\);"
    matches = re.findall(pattern, db_dump, flags=re.DOTALL | re.IGNORECASE)

    tables = []
    for table_name, columns_block in matches:
        cols = []
        for line in columns_block.split(","):
            line = " ".join(line.split())
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(("FOREIGN KEY", "PRIMARY KEY (", "UNIQUE (")):
                continue
            cols.append(line)
        tables.append(f"TABLE {table_name}: {', '.join(cols)}")
    return " | ".join(tables)


def map_ambig_type(ambig_type: str) -> str:
    return {
        "attachment": "ATTACH",
        "scope": "SCOPE",
        "vagueness": "VAGUE",
    }.get(str(ambig_type).lower(), "UNKNOWN")


def build_ambrosia_example(row: pd.Series) -> dict | None:
    schema = extract_schema_from_dump(row["db_dump"])
    if not schema:
        return None

    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        return None

    is_ambig = str(row["question_type"]).lower() == "ambig"

    if is_ambig:
        return {
            "question": question.strip(),
            "db_type": DEFAULT_DB_TYPE,
            "db_schema": schema,
            "label_status": "ambiguous",
            "completeness_score": 0.4,
            "missing_dimensions": [map_ambig_type(row["ambig_type"])],
            "reasons": [f"Тип неоднозначности: {row['ambig_type']}"],
        }

    return {
        "question": question.strip(),
        "db_type": DEFAULT_DB_TYPE,
        "db_schema": schema,
        "label_status": "clear",
        "completeness_score": 1.0,
        "missing_dimensions": [],
        "reasons": [],
    }


OWN_EXAMPLES = [
    {
        "question": "Сколько заказов было в январе 2025 года?",
        "db_type": "postgresql",
        "db_schema": "TABLE customers: customer_id INT PRIMARY KEY, name VARCHAR, region VARCHAR, registration_date DATE | TABLE orders: order_id INT PRIMARY KEY, customer_id INT FOREIGN KEY REFERENCES customers(customer_id), order_date DATE, total_amount DECIMAL, status VARCHAR | TABLE products: product_id INT PRIMARY KEY, name VARCHAR, category VARCHAR, price DECIMAL | TABLE order_items: order_id INT, product_id INT, quantity INT",
        "label_status": "clear",
        "completeness_score": 1.0,
        "missing_dimensions": [],
        "reasons": [],
    },
    {
        "question": "Покажи продажи",
        "db_type": "postgresql",
        "db_schema": "TABLE customers: customer_id INT PRIMARY KEY, name VARCHAR, region VARCHAR, registration_date DATE | TABLE orders: order_id INT PRIMARY KEY, customer_id INT FOREIGN KEY REFERENCES customers(customer_id), order_date DATE, total_amount DECIMAL, status VARCHAR | TABLE products: product_id INT PRIMARY KEY, name VARCHAR, category VARCHAR, price DECIMAL | TABLE order_items: order_id INT, product_id INT, quantity INT",
        "label_status": "incomplete",
        "completeness_score": 0.3,
        "missing_dimensions": ["MISSING_TIME"],
        "reasons": ["Не указан период времени"],
    },
    {
        "question": "Покажи топ-10 клиентов",
        "db_type": "postgresql",
        "db_schema": "TABLE customers: customer_id INT PRIMARY KEY, name VARCHAR, region VARCHAR, registration_date DATE | TABLE orders: order_id INT PRIMARY KEY, customer_id INT FOREIGN KEY REFERENCES customers(customer_id), order_date DATE, total_amount DECIMAL, status VARCHAR | TABLE products: product_id INT PRIMARY KEY, name VARCHAR, category VARCHAR, price DECIMAL | TABLE order_items: order_id INT, product_id INT, quantity INT",
        "label_status": "ambiguous",
        "completeness_score": 0.4,
        "missing_dimensions": ["MISSING_SORT"],
        "reasons": ["Не указано, по какому полю считать топ"],
    },
]


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["ambig_type"].isin(TARGET_TYPES)]

    parts = []
    for ambig_type in TARGET_TYPES:
        subset = df[df["ambig_type"] == ambig_type]
        parts.append(
            subset.sample(
                n=min(SAMPLES_PER_TYPE, len(subset)),
                random_state=42,
            )
        )
    sampled = pd.concat(parts, ignore_index=True)

    examples = []
    for _, row in sampled.iterrows():
        example = build_ambrosia_example(row)
        if example:
            examples.append(example)

    examples.extend(OWN_EXAMPLES)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Сохранено {len(examples)} примеров в {OUT_PATH}")


if __name__ == "__main__":
    main()
