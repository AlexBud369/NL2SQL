from __future__ import annotations

CODES_ORDER: list[str] = [
    "UNKNOWN_TABLE",
    "UNKNOWN_COLUMN",
    "SCOPE",
    "VAGUE",
    "ATTACH",
    "MISSING_TIME",
    "MISSING_VALUE",
    "MISSING_CRITERION",
    "MISSING_AGG",
    "MISSING_SORT",
    "MISSING_LIMIT",
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
        "MISSING_AGG",
        "MISSING_SORT",
        "MISSING_LIMIT",
    }
)

PENALTIES: dict[str, float] = {
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
    "MISSING_VALUE": 0.40,
}

REASON_TEMPLATES: dict[str, str] = {
    "UNKNOWN_TABLE": "Указанная таблица отсутствует в выбранной схеме базы данных.",
    "UNKNOWN_COLUMN": "Указанная колонка или атрибут отсутствует в таблице.",
    "SCOPE": "Не определена область выборки: не указано, как именно объединять или связывать сущности.",
    "VAGUE": "Использовано субъективное или размытое определение, требующее числового порога.",
    "ATTACH": "Неоднозначная синтаксическая привязка: неясно, к какому объекту относится условие.",
    "MISSING_TIME": "Не указан конкретный временной интервал или дата.",
    "MISSING_VALUE": "Отсутствуют обязательные значения для добавления или обновления строки.",
    "MISSING_CRITERION": "Отсутствует критерий или условие фильтрации (WHERE).",
    "MISSING_AGG": "Не указана функция агрегации (сумма, среднее, количество или максимум).",
    "MISSING_SORT": "Не указано поле сортировки для построения рейтинга или топа.",
    "MISSING_LIMIT": "Не указано предельное количество строк для выборки.",
}

CLARIFICATION_TEMPLATES: dict[str, str] = {
    "UNKNOWN_TABLE": "Уточните, с какой именно таблицей из схемы требуется выполнить операцию.",
    "UNKNOWN_COLUMN": "Уточните корректное имя атрибута или колонки.",
    "SCOPE": "Укажите, требуется ли объединить данные через связывание (JOIN) или вывести отдельными списками.",
    "VAGUE": "Укажите конкретное числовое значение или критерий отбора вместо размытого слова.",
    "ATTACH": "Поясните, к какой конкретно таблице относится условие фильтрации.",
    "MISSING_TIME": "Укажите точные даты начала и конца периода (например, с 01.01.2024 по 31.01.2024).",
    "MISSING_VALUE": "Перечислите значения всех обязательных полей для записи.",
    "MISSING_CRITERION": "Задайте условие фильтрации строк (например, по ID или статусу).",
    "MISSING_AGG": "Укажите, какую агрегацию применить: сумму (SUM), среднее (AVG) или количество (COUNT).",
    "MISSING_SORT": "По какому числовому или строковому полю производить сортировку?",
    "MISSING_LIMIT": "Сколько именно записей необходимо вернуть (например, топ 5 или топ 10)?",
}
