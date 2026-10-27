from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
SCHEMAS_FILE = DATA_DIR / "schemas.jsonl"
DATASET_V2_FILE = DATA_DIR / "dataset_v2.jsonl"
SCHEMAS_BACKUP = DATA_DIR / "schemas_old.jsonl"

TABLE_RE = re.compile(r"TABLE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)(?=\s*\|\s*TABLE\s|$)", re.I)
DB_TYPES = ["postgresql", "mysql", "sqlserver"]


def read_schema_file(path: Path) -> dict[str, str]:
    schemas_dict: dict[str, str] = {}
    if not path.exists():
        return schemas_dict
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                name = item.get("name") or item.get("db_id")
                schema = item.get("db_schema")
                if name and schema:
                    schemas_dict[schema.strip()] = name
    return schemas_dict


def sanitize_name(name: str, used_names: set[str], fallback_index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or f"schema_{fallback_index}"
    candidate = cleaned
    counter = 1
    while candidate in used_names:
        counter += 1
        candidate = f"{cleaned}_{counter}"
    used_names.add(candidate)
    return candidate


def build_schema_records(schemas_dict: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for idx, (schema_text, raw_name) in enumerate(schemas_dict.items(), 1):
        if not TABLE_RE.findall(schema_text):
            continue
        unique_name = sanitize_name(raw_name, used_names, idx)
        db_type = DB_TYPES[idx % len(DB_TYPES)]
        records.append({"name": unique_name, "db_type": db_type, "db_schema": schema_text})

    return records


def backup_and_save(target_file: Path, backup_file: Path, records: list[dict[str, Any]]) -> None:
    if target_file.exists() and not backup_file.exists():
        target_file.rename(backup_file)
    with open(target_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    combined_schemas = read_schema_file(SCHEMAS_FILE)
    combined_schemas.update(read_schema_file(DATASET_V2_FILE))

    valid_records = build_schema_records(combined_schemas)
    backup_and_save(SCHEMAS_FILE, SCHEMAS_BACKUP, valid_records)


if __name__ == "__main__":
    main()
