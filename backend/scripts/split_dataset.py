import json
import os
import random
from collections import defaultdict

INPUT_FILE = "backend/data/synthetic_queries.jsonl"
TRAIN_FILE = "backend/data/train.jsonl"
VAL_FILE = "backend/data/val.jsonl"
TRAIN_RATIO = 0.8


def split_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    schemas_data = defaultdict(list)
    for item in data:
        schema_key = item["db_schema"]
        schemas_data[schema_key].append(item)

    unique_schemas = list(schemas_data.keys())
    random.seed(42)
    random.shuffle(unique_schemas)

    split_idx = int(len(unique_schemas) * TRAIN_RATIO)
    train_schemas = set(unique_schemas[:split_idx])
    val_schemas = set(unique_schemas[split_idx:])

    train_data = []
    val_data = []

    for schema, items in schemas_data.items():
        if schema in train_schemas:
            train_data.extend(items)
        else:
            val_data.extend(items)

    random.shuffle(train_data)
    random.shuffle(val_data)

    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Всего схем: {len(unique_schemas)}")
    print(f"Схем в Train: {len(train_schemas)} | Схем в Val: {len(val_schemas)}")
    print(f"Строк в Train: {len(train_data)}")
    print(f"Строк в Val: {len(val_data)}")


if __name__ == "__main__":
    split_dataset()
