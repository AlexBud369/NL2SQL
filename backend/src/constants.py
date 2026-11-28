from __future__ import annotations

# 9 ортогональных классов дефектов качества NL2SQL запросов:
# 1. Ошибки схемы и домена (CRITICAL): UNKNOWN_TABLE, UNKNOWN_COLUMN, SCOPE
# 2. Ошибки синтаксиса и безопасности: MISSING_VALUE, MISSING_CRITERION
# 3. Ошибки семантики и контекста: VAGUE, ATTACH, MISSING_TIME
# 4. Неполнота ранжирования / среза (объединенные SORT + LIMIT): RANKING_SPEC
CODES_ORDER: list[str] = [
    "UNKNOWN_TABLE",
    "UNKNOWN_COLUMN",
    "SCOPE",
    "VAGUE",
    "ATTACH",
    "MISSING_TIME",
    "MISSING_VALUE",
    "MISSING_CRITERION",
    "RANKING_SPEC",
]

AMBIGUOUS_CODES: frozenset[str] = frozenset(
    {
        "UNKNOWN_TABLE",
        "UNKNOWN_COLUMN",
        "SCOPE",
        "VAGUE",
        "ATTACH",
    }
)

INCOMPLETE_CODES: frozenset[str] = frozenset(
    {
        "MISSING_TIME",
        "MISSING_VALUE",
        "MISSING_CRITERION",
        "RANKING_SPEC",
    }
)

PENALTIES: dict[str, float] = {
    "UNKNOWN_TABLE": 0.50,
    "UNKNOWN_COLUMN": 0.40,
    "SCOPE": 0.35,
    "VAGUE": 0.30,
    "ATTACH": 0.30,
    "MISSING_TIME": 0.35,
    "MISSING_VALUE": 0.40,
    "MISSING_CRITERION": 0.25,
    "RANKING_SPEC": 0.25,
    # Для совместимости со старыми моделями:
    "MISSING_AGG": 0.25,
    "MISSING_SORT": 0.20,
    "MISSING_LIMIT": 0.15,
}

REASON_TEMPLATES: dict[str, str] = {
    "UNKNOWN_TABLE": "Указанная таблица отсутствует в выбранной схеме базы данных.",
    "UNKNOWN_COLUMN": "Указанная колонка или атрибут отсутствует в таблице.",
    "SCOPE": "Запрос не относится к тематике базы данных либо содержит неоднозначную область выборки.",
    "VAGUE": "Использовано субъективное или размытое определение, требующее числового порога.",
    "ATTACH": "Неоднозначная синтаксическая привязка: неясно, к какому объекту относится условие.",
    "MISSING_TIME": "Не указан конкретный временной интервал или дата.",
    "MISSING_VALUE": "Отсутствуют обязательные значения для добавления или обновления строки.",
    "MISSING_CRITERION": "Отсутствует критерий или условие фильтрации (WHERE) для деструктивной операции.",
    "RANKING_SPEC": "Не указаны параметры топа/ранжирования (количество записей или поле сортировки).",
    # Для совместимости со старыми моделями:
    "MISSING_AGG": "Не указана функция агрегации.",
    "MISSING_SORT": "Не указано поле сортировки.",
    "MISSING_LIMIT": "Не указан лимит строк.",
}

CLARIFICATION_TEMPLATES: dict[str, str] = {
    "UNKNOWN_TABLE": "Уточните, с какой именно таблицей из схемы требуется выполнить операцию.",
    "UNKNOWN_COLUMN": "Уточните корректное имя атрибута или колонки.",
    "SCOPE": "Сформулируйте запрос в терминах доступных таблиц и полей базы данных.",
    "VAGUE": "Укажите конкретное числовое значение или критерий отбора вместо размытого слова.",
    "ATTACH": "Поясните, к какой конкретно таблице относится условие фильтрации.",
    "MISSING_TIME": "Укажите точные даты начала и конца периода (например, с 01.01.2024 по 31.01.2024).",
    "MISSING_VALUE": "Перечислите значения всех обязательных полей для записи.",
    "MISSING_CRITERION": "Задайте условие фильтрации строк (например, по ID или статусу).",
    "RANKING_SPEC": "Укажите количество записей для выборки (лимит) и поле/направление сортировки.",
    # Для совместимости со старыми моделями:
    "MISSING_AGG": "Укажите функцию агрегации.",
    "MISSING_SORT": "Укажите поле сортировки.",
    "MISSING_LIMIT": "Укажите лимит строк.",
}
