from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Обеспечиваем доступ к пакету src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ambiguity.detector import AmbiguityDetectorService

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Инференс модели Quality Gate (Ambiguity & Completeness Detector)"
    )
    parser.add_argument("--query", "-q", type=str, help="Текст запроса на естественном языке")
    parser.add_argument("--db_type", "-d", type=str, default="PostgreSQL", help="Тип СУБД")
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default="users (id, name, age, city), orders (id, user_id, amount, created_at)",
        help="DDL или строковое описание схемы БД",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(PROJECT_ROOT / "models" / "best_model.pt"),
        help="Путь к весам best_model.pt",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    print(f"Загрузка модели из: {checkpoint_path} ...")
    if checkpoint_path == (PROJECT_ROOT / "models" / "best_model.pt"):
        from src.ambiguity.detector import detector_singleton
        detector = detector_singleton
    else:
        detector = AmbiguityDetectorService(checkpoint_path=checkpoint_path)
    print("Модель успешно загружена!\n")

    if args.query:
        # Режим единичного запуска через аргументы командной строки
        run_prediction(detector, args.query, args.db_type, args.schema)
        return

    # Интерактивный режим диалога в терминале
    print("=" * 70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПРОВЕРКИ ЗАПРОСОВ (Для выхода введите 'exit' или 'q')")
    print(f"Текущая схема БД по умолчанию:\n  {args.schema}")
    print("=" * 70)

    current_schema = args.schema
    current_db_type = args.db_type

    while True:
        try:
            print()
            query = input("Введите запрос пользователя > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Выход из программы.")
                break

            run_prediction(detector, query, current_db_type, current_schema)
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из программы.")
            break


def run_prediction(
    detector: AmbiguityDetectorService, query: str, db_type: str, schema: str
) -> None:
    response = detector.analyze(question=query, db_type=db_type, schema=schema)

    status_color = {
        "clear": "\033[92m[CLEAR (Однозначен)]\033[0m",
        "incomplete": "\033[93m[INCOMPLETE (Не полон)]\033[0m",
        "ambiguous": "\033[91m[AMBIGUOUS (Неоднозначен)]\033[0m",
    }.get(response.ambiguity_status, response.ambiguity_status)

    print("-" * 60)
    print(f"Запрос:           {query}")
    print(f"Статус качества:  {status_color}")
    print(f"Индекс полноты:   {response.completeness_score:.2f}")

    if response.missing_dimensions:
        print("Обнаруженные дефекты:")
        for dim in response.missing_dimensions:
            print(f"  • {dim.code}: {dim.description}")
    else:
        print("Дефекты:          Нет (запрос готов к генерации SQL)")

    if response.suggested_clarifications:
        print("Рекомендации пользователю:")
        for sugg in response.suggested_clarifications:
            print(f"  → {sugg}")
    print("-" * 60)


if __name__ == "__main__":
    main()
