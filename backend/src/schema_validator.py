from __future__ import annotations

import re

TABLE_PATTERN = re.compile(r"TABLE\s+([A-Za-z0-9_]+)\s*:\s*(.*?)(?=\s*\|\s*TABLE\s|$)", re.I)
CONSTRAINT_PREFIXES = ("FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "KEY ")

SQL_KEYWORDS = frozenset({
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "group", "by", "order", "having", "limit", "offset", "insert", "into",
    "values", "update", "set", "delete", "table", "as", "and", "or", "not",
    "in", "is", "null", "like", "between", "distinct", "count", "sum", "avg",
    "min", "max", "primary", "key", "foreign", "references", "int", "varchar",
    "boolean", "date", "timestamp", "decimal", "float", "true", "false",
    "cancelled", "admin", "hr", "shop", "hospital", "flights", "id",
    "всех", "базы", "данных", "список"
})

TABLE_KW_RE = re.compile(r"(?:таблиц[аыеу]|из\s+таблицы|в\s+таблиц[еу])\s+([a-zA-Z0-9_]+)", re.I)
COLUMN_KW_RE = re.compile(r"(?:пол[еяю]|колонк[аеуи]|полю|атрибут[аы]?|по)\s+([a-zA-Z0-9_]+)", re.I)


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

    def validate_query(self, question: str) -> list[str]:
        errors: set[str] = set()
        all_tables = set(self.tables.keys())
        all_columns = {col for cols in self.tables.values() for col in cols}

        for match in TABLE_KW_RE.finditer(question):
            token = match.group(1).lower()
            if token not in SQL_KEYWORDS and token not in all_tables:
                errors.add("UNKNOWN_TABLE")

        for match in COLUMN_KW_RE.finditer(question):
            token = match.group(1).lower()
            if token not in SQL_KEYWORDS and token not in all_tables and token not in all_columns:
                errors.add("UNKNOWN_COLUMN")

        return sorted(list(errors))
