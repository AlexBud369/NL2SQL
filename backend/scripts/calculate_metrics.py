from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CODES_ORDER
from src.dataset import NL2SQLDataCollator, NL2SQLDataset
from src.metrics import (
    compute_clarity_metrics,
    compute_multilabel_metrics,
    compute_per_class_metrics,
    compute_score_metrics,
)
from src.model import NL2SQLQualityModel
from src.postprocess import compute_completeness_score, derive_clarity

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("metrics")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Расчет и вывод метрик модели")
    parser.add_argument("--checkpoint", type=str, default="backend/models/best_model.pt")
    parser.add_argument("--data_file", type=str, default="backend/data/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--export_json", type=str, default="backend/models/metrics_report.json")
    return parser.parse_args()


def load_dataset_scores(path: Path) -> list[float]:
    scores = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                scores.append(float(item.get("completeness_score", 1.0)))
    return scores


def load_model_and_tokenizer(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[NL2SQLQualityModel, Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_name = config.get("model_name", "cointegrated/rubert-tiny2")
    codes = config.get("codes_order", CODES_ORDER)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NL2SQLQualityModel(model_name=model_name, num_codes=len(codes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, config


def run_batch_inference(
    model: NL2SQLQualityModel,
    batch: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
    labels = batch["labels"]

    t0 = time.perf_counter()
    logits = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0 / inputs["input_ids"].size(0)

    binary_preds = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
    return binary_preds, labels.cpu().numpy().astype(int), latency_ms


def evaluate_dataset(
    model: NL2SQLQualityModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    codes_order: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[float], list[float]]:
    all_preds, all_targets, all_true_clarity, all_pred_clarity, all_pred_scores, latencies = [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            preds, targets, per_sample_latency = run_batch_inference(model, batch, device, threshold)
            all_preds.append(preds)
            all_targets.append(targets)
            latencies.extend([per_sample_latency] * len(preds))

            for pred_row, true_clarity in zip(preds, batch["clarities"]):
                detected = [codes_order[i] for i, v in enumerate(pred_row) if v == 1]
                all_pred_clarity.append(derive_clarity(detected))
                all_true_clarity.append(true_clarity)
                all_pred_scores.append(compute_completeness_score(detected))

    return np.vstack(all_preds), np.vstack(all_targets), all_true_clarity, all_pred_clarity, all_pred_scores, latencies


def display_multilabel_results(
    codes_order: list[str],
    per_class: dict[str, dict[str, float]],
    multi_metrics: dict[str, float],
) -> None:
    print("\n1. ТАБЛИЦА МЕТРИК ПО КОДАМ ОШИБОК:")
    print("┌────────────────────┬───────────┬────────┬──────────┬─────────┐")
    print("│ Код ошибки         │ Precision │ Recall │ F1-Score │ Support │")
    print("├────────────────────┼───────────┼────────┼──────────┼─────────┤")
    for code in codes_order:
        data = per_class[code]
        print(f"│ {code:<18s} │  {data['precision']*100:6.2f}%  │ {data['recall']*100:5.2f}% │  {data['f1']*100:6.2f}%  │ {int(data['support']):7d} │")
    print("├────────────────────┼───────────┼────────┼──────────┼─────────┤")
    print(f"│ MACRO F1           │     —     │    —   │  {multi_metrics['macro_f1']*100:6.2f}%  │    —    │")
    print(f"│ MICRO F1           │     —     │    —   │  {multi_metrics['micro_f1']*100:6.2f}%  │    —    │")
    print("└────────────────────┴───────────┴────────┴──────────┴─────────┘")
    print(f"  • Subset Accuracy: {multi_metrics['subset_accuracy']*100:.2f}%")
    print(f"  • Hamming Loss:    {multi_metrics['hamming_loss']:.4f}")


def display_clarity_results(clarity_metrics: dict[str, Any]) -> None:
    cm = clarity_metrics["confusion_matrix"]
    print("\n2. ОЦЕНКА БИЗНЕС-СТАТУСА CLARITY:")
    print(f"  • Accuracy:        {clarity_metrics['clarity_accuracy']*100:.2f}%")
    print(f"  • Macro F1:        {clarity_metrics['clarity_macro_f1']*100:.2f}%")
    print(f"  • False Clear:     {clarity_metrics['false_clear_rate']:.2f}%")
    print(f"  • False Block:     {clarity_metrics['false_block_rate']:.2f}%")
    print("\n  Матрица ошибок (Confusion Matrix):")
    print("  ┌─────────────────┬──────────────┬─────────────────┬────────────────┐")
    print("  │ Факт \\ Прогноз  │  Pred CLEAR  │ Pred INCOMPLETE │ Pred AMBIGUOUS │")
    print("  ├─────────────────┼──────────────┼─────────────────┼────────────────┤")
    print(f"  │ True CLEAR      │ {cm[0][0]:^12d} │ {cm[0][1]:^15d} │ {cm[0][2]:^14d} │")
    print(f"  │ True INCOMPLETE │ {cm[1][0]:^12d} │ {cm[1][1]:^15d} │ {cm[1][2]:^14d} │")
    print(f"  │ True AMBIGUOUS  │ {cm[2][0]:^12d} │ {cm[2][1]:^15d} │ {cm[2][2]:^14d} │")
    print("  └─────────────────┴──────────────┴─────────────────┴────────────────┘")


def display_latency_and_score_results(score_metrics: dict[str, float], latencies: list[float]) -> None:
    print("\n3. ТОЧНОСТЬ ОЦЕНКИ ПОЛНОТЫ:")
    print(f"  • MAE:  {score_metrics['score_mae']:.4f}")
    print(f"  • RMSE: {score_metrics['score_rmse']:.4f}")
    print("\n4. СКОРОСТЬ ИНФЕРЕНСА:")
    print(f"  • Среднее: {np.mean(latencies):.2f} мс")
    print(f"  • P50:     {np.percentile(latencies, 50):.2f} мс")
    print(f"  • P95:     {np.percentile(latencies, 95):.2f} мс")
    print(f"  • P99:     {np.percentile(latencies, 99):.2f} мс")


def save_json_report(path: str, data: dict[str, Any]) -> None:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_data_path(file_path: str) -> Path:
    direct = Path(file_path)
    if direct.exists():
        return direct
    from_root = PROJECT_ROOT / file_path
    if from_root.exists():
        return from_root
    stripped = file_path.removeprefix("backend/").removeprefix("backend\\")
    from_stripped = PROJECT_ROOT / stripped
    if from_stripped.exists():
        return from_stripped
    return direct


def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_data_path(args.checkpoint)
    data_path = resolve_data_path(args.data_file)
    if not checkpoint_path.exists() or not data_path.exists():
        logger.error("Files not found: %s or %s", checkpoint_path, data_path)
        sys.exit(1)

    model, tokenizer, config = load_model_and_tokenizer(checkpoint_path, device)
    codes_order = config.get("codes_order", CODES_ORDER)

    dataset = NL2SQLDataset(data_path, tokenizer, max_length=config.get("max_length", 512))
    collator = NL2SQLDataCollator(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    preds, targets, true_clarity, pred_clarity, pred_scores, latencies = evaluate_dataset(
        model, loader, device, args.threshold, codes_order
    )
    true_scores = load_dataset_scores(data_path)

    multilabel_results = compute_multilabel_metrics(targets, preds)
    per_class_results = compute_per_class_metrics(targets, preds, codes_order)
    clarity_results = compute_clarity_metrics(true_clarity, pred_clarity)
    score_results = compute_score_metrics(true_scores, pred_scores)

    print("\n" + "=" * 78)
    print("      📊 ОТЧЕТ КАЧЕСТВА МОДЕЛИ QUALITY GATE (NL2SQL)")
    print("=" * 78)
    display_multilabel_results(codes_order, per_class_results, multilabel_results)
    display_clarity_results(clarity_results)
    display_latency_and_score_results(score_results, latencies)
    print("=" * 78 + "\n")

    if args.export_json:
        report_payload = {
            "model_name": config.get("model_name"),
            "multilabel": multilabel_results,
            "clarity": clarity_results,
            "score": score_results,
            "per_class": per_class_results,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        }
        save_json_report(args.export_json, report_payload)


if __name__ == "__main__":
    main()
