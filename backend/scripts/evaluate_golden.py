from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from src.constants import CODES_ORDER
from src.model import NL2SQLQualityModel
from src.postprocess import derive_clarity
from src.schema_validator import SchemaValidator

STATUSES: list[str] = ["clear", "incomplete", "ambiguous"]

DEFAULT_PER_CLASS_THRESHOLDS: dict[str, float] = {
    "UNKNOWN_TABLE": 0.50,
    "UNKNOWN_COLUMN": 0.50,
    "SCOPE": 0.50,
    "MISSING_TIME": 0.40,
    "MISSING_CRITERION": 0.40,
    "VAGUE": 0.40,
    "ATTACH": 0.50,
    "MISSING_VALUE": 0.20,
    "RANKING_SPEC": 0.35,
    "MISSING_AGG": 0.50,
    "MISSING_SORT": 0.25,
    "MISSING_LIMIT": 0.25,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Автоматическое тестирование модели на Golden Test")
    parser.add_argument("--test_file", type=str, default="backend/data/golden_test.jsonl")
    parser.add_argument("--checkpoint", type=str, default="backend/models/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_calibrated", action="store_true", help="Использовать индивидуальные калиброванные пороги")
    parser.add_argument("--export_json", type=str, default="backend/models/golden_test_report.json")
    return parser.parse_args()


def load_model_and_tokenizer(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[NL2SQLQualityModel, Any, list[str]]:
    if not checkpoint_path.exists():
        fallback = checkpoint_path.parent / "best_model_v5.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Файл чекпоинта не найден: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "cointegrated/rubert-tiny2")
    codes = config.get("codes_order", CODES_ORDER)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NL2SQLQualityModel(model_name=model_name, num_codes=len(codes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, codes


def load_golden_samples(data_path: Path) -> list[dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Файл эталонного датасета не найден: {data_path}")
    samples: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append({
                "id": item.get("id", f"gold-{idx:03d}"),
                "question": item.get("question", ""),
                "schema": item.get("schema", item.get("db_schema", "")),
                "db_type": item.get("db_type", "postgresql"),
                "expected_status": item.get("expected_status", item.get("clarity", "clear")),
                "expected_codes": item.get("expected_codes", item.get("codes", [])),
            })
    return samples


def run_inference(
    model: NL2SQLQualityModel,
    tokenizer: Any,
    item: dict[str, Any],
    device: torch.device,
) -> tuple[list[float], float]:
    prompt = f"СУБД: {item['db_type']}. Запрос: {item['question']}"
    inputs = tokenizer(
        prompt,
        item["schema"],
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model.predict_probabilities(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return probs.squeeze(0).cpu().tolist(), latency_ms


def evaluate_sample(
    item: dict[str, Any],
    probs: list[float],
    codes_order: list[str],
    threshold: float | dict[str, float],
    latency_ms: float,
    syntax_codes: list[str] | None = None,
) -> dict[str, Any]:
    neural_codes: list[str] = []
    for prob, code in zip(probs, codes_order):
        cutoff = threshold.get(code, 0.5) if isinstance(threshold, dict) else threshold
        if prob >= cutoff:
            neural_codes.append(code)
    pred_codes = sorted(list(set(neural_codes + (syntax_codes or []))))
    pred_status = derive_clarity(pred_codes)
    exp_status = item["expected_status"]
    exp_codes = item["expected_codes"]
    status_match = pred_status == exp_status
    code_exact_match = set(pred_codes) == set(exp_codes)
    return {
        "id": item["id"],
        "question": item["question"],
        "expected_status": exp_status,
        "pred_status": pred_status,
        "expected_codes": exp_codes,
        "pred_codes": pred_codes,
        "status_match": status_match,
        "code_exact_match": code_exact_match,
        "latency_ms": latency_ms,
        "probabilities": {c: round(p, 4) for c, p in zip(codes_order, probs)},
    }


def compute_status_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    cm = [[0 for _ in range(3)] for _ in range(3)]
    for r in records:
        t_idx = STATUSES.index(r["expected_status"])
        p_idx = STATUSES.index(r["pred_status"])
        cm[t_idx][p_idx] += 1
    acc = sum(cm[i][i] for i in range(3)) / max(1, len(records))
    return {"confusion_matrix": cm, "status_accuracy": acc}


def compute_code_metrics(
    records: list[dict[str, Any]],
    codes_order: list[str],
) -> tuple[dict[str, dict[str, float]], float, float]:
    per_class: dict[str, dict[str, float]] = {}
    f1_list: list[float] = []
    exact_matches = sum(1 for r in records if r["code_exact_match"])
    subset_acc = exact_matches / max(1, len(records))

    for code in codes_order:
        tp = sum(1 for r in records if code in r["pred_codes"] and code in r["expected_codes"])
        fp = sum(1 for r in records if code in r["pred_codes"] and code not in r["expected_codes"])
        fn = sum(1 for r in records if code not in r["pred_codes"] and code in r["expected_codes"])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[code] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        if (tp + fn) > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return per_class, macro_f1, subset_acc


def run_golden_evaluation(
    model: NL2SQLQualityModel,
    tokenizer: Any,
    codes_order: list[str],
    test_file: Path | str = PROJECT_ROOT / "data" / "golden_test.jsonl",
    threshold: float | dict[str, float] = 0.5,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    samples = load_golden_samples(Path(test_file))
    records: list[dict[str, Any]] = []
    for item in samples:
        validator = SchemaValidator(item["schema"])
        syntax_codes = validator.validate_query(item["question"])
        probs, latency_ms = run_inference(model, tokenizer, item, device)
        records.append(
            evaluate_sample(item, probs, codes_order, threshold, latency_ms, syntax_codes)
        )

    status_metrics = compute_status_metrics(records)
    per_class, macro_f1, subset_acc = compute_code_metrics(records, codes_order)
    avg_latency = float(np.mean([r["latency_ms"] for r in records]))

    threshold_label = threshold if isinstance(threshold, (int, float)) else "calibrated"
    return {
        "threshold": threshold_label,
        "threshold_details": threshold if isinstance(threshold, dict) else {"fixed": threshold},
        "total_samples": len(records),
        "status_accuracy": status_metrics["status_accuracy"],
        "code_subset_accuracy": subset_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": status_metrics["confusion_matrix"],
        "per_class_metrics": per_class,
        "avg_latency": avg_latency,
        "records": records,
        "mismatches": [r for r in records if not (r["status_match"] and r["code_exact_match"])],
    }


def display_summary(
    total: int,
    status_acc: float,
    exact_match: float,
    macro_f1: float,
    avg_latency: float,
    threshold: float | dict[str, float],
) -> None:
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ НА GOLDEN TEST")
    print("=" * 70)
    print(f"Всего тестовых примеров:       {total}")
    thresh_str = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "поклассовый (калиброванный)"
    print(f"Порог классификации:          {thresh_str}")
    print(f"Точность бизнес-статуса:      {status_acc * 100:.2f}%")
    print(f"Точное совпадение кодов (EM): {exact_match * 100:.2f}%")
    print(f"Macro F1 по активным кодам:   {macro_f1 * 100:.2f}%")
    print(f"Среднее время ответа (мс):    {avg_latency:.2f} ms")


def display_confusion_matrix(cm: list[list[int]]) -> None:
    print("\nМАТРИЦА ОШИБОК СТАТУСОВ (CONFUSION MATRIX):")
    print("┌─────────────────┬──────────────┬─────────────────┬────────────────┐")
    print("│ Факт \\ Прогноз  │  Pred CLEAR  │ Pred INCOMPLETE │ Pred AMBIGUOUS │")
    print("├─────────────────┼──────────────┼─────────────────┼────────────────┤")
    print(f"│ True CLEAR      │ {cm[0][0]:^12d} │ {cm[0][1]:^15d} │ {cm[0][2]:^14d} │")
    print(f"│ True INCOMPLETE │ {cm[1][0]:^12d} │ {cm[1][1]:^15d} │ {cm[1][2]:^14d} │")
    print(f"│ True AMBIGUOUS  │ {cm[2][0]:^12d} │ {cm[2][1]:^15d} │ {cm[2][2]:^14d} │")
    print("└─────────────────┴──────────────┴─────────────────┴────────────────┘")


def display_code_table(codes_order: list[str], per_class: dict[str, dict[str, float]]) -> None:
    print("\nДЕТАЛИЗАЦИЯ ПО КОДАМ ОШИБОК:")
    print("┌────────────────────┬───────────┬────────┬──────────┬─────────┐")
    print("│ Код дефекта        │ Precision │ Recall │ F1-Score │ Support │")
    print("├────────────────────┼───────────┼────────┼──────────┼─────────┤")
    for code in codes_order:
        data = per_class[code]
        print(
            f"│ {code:<18s} │  {data['precision']*100:6.2f}%  │ {data['recall']*100:5.2f}% │  {data['f1']*100:6.2f}%  │ {int(data['support']):7d} │"
        )
    print("└────────────────────┴───────────┴────────┴──────────┴─────────┘")


def display_mismatches(records: list[dict[str, Any]]) -> None:
    mismatches = [r for r in records if not (r["status_match"] and r["code_exact_match"])]
    print(f"\nСПИСОК ВЫЯВЛЕННЫХ РАСХОЖДЕНИЙ ({len(mismatches)} из {len(records)}):")
    print("-" * 70)
    for m in mismatches:
        print(f"[{m['id']}] Запрос: {m['question']}")
        print(f"  Статус:   ожидался={m['expected_status']} | модель={m['pred_status']} ({'OK' if m['status_match'] else 'ОШИБКА'})")
        print(f"  Коды:     ожидались={m['expected_codes']} | модель={m['pred_codes']}")
        top_probs = {c: p for c, p in m["probabilities"].items() if p >= 0.15 or c in m["expected_codes"]}
        print(f"  Вероятности ключевых кодов: {top_probs}")
        print("-" * 70)


def save_report(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, codes_order = load_model_and_tokenizer(Path(args.checkpoint), device)
    active_thresh = DEFAULT_PER_CLASS_THRESHOLDS if args.use_calibrated else args.threshold
    report = run_golden_evaluation(
        model=model,
        tokenizer=tokenizer,
        codes_order=codes_order,
        test_file=Path(args.test_file),
        threshold=active_thresh,
    )
    display_summary(
        report["total_samples"],
        report["status_accuracy"],
        report["code_subset_accuracy"],
        report["macro_f1"],
        report["avg_latency"],
        active_thresh,
    )
    display_confusion_matrix(report["confusion_matrix"])
    display_code_table(codes_order, report["per_class_metrics"])
    display_mismatches(report["records"])
    save_report(Path(args.export_json), report)
    print(f"\nПолный отчёт сохранён в файл: {args.export_json}")


if __name__ == "__main__":
    main()
