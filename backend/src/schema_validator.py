from __future__ import annotations

import re

TABLE_PATTERN = re.compile(r"TABLE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)(?=\s*\|\s*TABLE\s|$)", re.I)
CONSTRAINT_PREFIXES = ("FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "KEY ")


class SchemaValidator:
    def __init__(self, db_schema: str):
        self.db_schema = db_schema
        self.tables: dict[str, list[str]] = self._parse_schema(db_schema)

    def _clean_column_token(self, raw_token: str) -> str | None:
        trimmed = raw_token.strip()
        if not trimmed or trimmed.upper().startswith(CONSTRAINT_PREFIXES):
            return None
        return trimmed.split()[0].strip('`"').lower()

    def _parse_columns_block(self, columns_block: str) -> list[str]:
        columns: list[str] = []
        for raw_column in columns_block.split(","):
            column_name = self._clean_column_token(raw_column)
            if column_name:
                columns.append(column_name)
        return columns

    def _parse_schema(self, schema_text: str) -> dict[str, list[str]]:
        parsed_schema: dict[str, list[str]] = {}
        for table_name, block in TABLE_PATTERN.findall(schema_text):
            parsed_schema[table_name.lower()] = self._parse_columns_block(block)
        return parsed_schema

    def _validate_table_names(self, tables_to_check: list[str]) -> list[str]:
        for table in tables_to_check:
            if table.lower() not in self.tables:
                return ["UNKNOWN_TABLE"]
        return []

    def _validate_column_names(
        self, columns_to_check: list[str], target_table: str | None
    ) -> list[str]:
        if target_table and target_table.lower() in self.tables:
            available_columns = set(self.tables[target_table.lower()])
        else:
            available_columns = {col for cols in self.tables.values() for col in cols}

        for column in columns_to_check:
            if column.lower() not in available_columns:
                return ["UNKNOWN_COLUMN"]
        return []

    def validate_entities(
        self,
        mentioned_tables: list[str] | None = None,
        mentioned_columns: list[str] | None = None,
        target_table: str | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if mentioned_tables:
            errors.extend(self._validate_table_names(mentioned_tables))
        if mentioned_columns:
            errors.extend(self._validate_column_names(mentioned_columns, target_table))
        return sorted(list(set(errors)))
