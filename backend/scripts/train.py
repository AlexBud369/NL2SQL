from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as transformers_logging,
)

transformers_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CODES_ORDER
from src.dataset import NL2SQLDataCollator, NL2SQLDataset
from src.metrics import (
    compute_clarity_metrics,
    compute_multilabel_metrics,
    compute_per_class_metrics,
)
from src.model import NL2SQLQualityModel
from src.postprocess import derive_clarity

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(dataset: NL2SQLDataset, max_weight: float = 15.0) -> torch.Tensor:
    total_count = len(dataset)
    positives = np.zeros(len(dataset.CODES_ORDER), dtype=np.float32)

    for item in dataset.data:
        codes = item.get("codes") if "codes" in item else item.get("expected_codes", [])
        for code in codes:
            if code in dataset.CODES_ORDER:
                positives[dataset.CODES_ORDER.index(code)] += 1.0

    weights = np.zeros(len(dataset.CODES_ORDER), dtype=np.float32)
    for index in range(len(dataset.CODES_ORDER)):
        positive_count = max(positives[index], 1.0)
        negative_count = total_count - positive_count
        weights[index] = min(negative_count / positive_count, max_weight)

    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer_parameters(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_parameters = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if not any(keyword in parameter_name for keyword in no_decay_keywords)
    ]
    no_decay_parameters = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if any(keyword in parameter_name for keyword in no_decay_keywords)
    ]
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def prepare_batch_tensors(
    batch: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key != "labels"
    }
    labels = batch["labels"].to(device)
    return inputs, labels


def apply_gradients(
    model: nn.Module,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> None:
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        return
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def execute_training_step(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    optimizer.zero_grad()
    inputs, labels = prepare_batch_tensors(batch, device)
    use_amp = device.type == "cuda" and scaler is not None
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(**inputs)
        loss = criterion(logits, labels)
    apply_gradients(model, loss, optimizer, scaler, use_amp)
    scheduler.step()
    return float(loss.item())


def train_single_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, device: torch.device, scaler: torch.amp.GradScaler | None, epoch: int, total_epochs: int,
) -> float:
    model.train()
    accumulated_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Эпоха {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True)
    for step, batch in enumerate(progress_bar, 1):
        step_loss = execute_training_step(model, batch, criterion, optimizer, scheduler, device, scaler)
        accumulated_loss += step_loss
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({"loss": f"{accumulated_loss / step:.4f}", "lr": f"{current_lr:.1e}"})
    return accumulated_loss / len(loader)


def extract_clarity_predictions(
    binary_predictions: np.ndarray,
    true_clarities: list[str],
    codes_order: list[str],
) -> tuple[list[str], list[str]]:
    predicted_list: list[str] = []
    true_list: list[str] = []
    for prediction_row, true_clarity in zip(binary_predictions, true_clarities):
        detected_codes = [codes_order[index] for index, flag in enumerate(prediction_row) if flag == 1]
        predicted_list.append(derive_clarity(detected_codes))
        true_list.append(true_clarity)
    return true_list, predicted_list


def collect_validation_predictions(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device, threshold: float, codes_order: list[str],
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    total_loss = 0.0
    all_predictions, all_targets, all_true_clarity, all_pred_clarity = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Валидация", leave=False):
            inputs, labels = prepare_batch_tensors(batch, device)
            logits = model(**inputs)
            total_loss += float(criterion(logits, labels).item())
            binary_preds = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
            all_predictions.append(binary_preds)
            all_targets.append(labels.cpu().numpy().astype(int))
            true_batch, pred_batch = extract_clarity_predictions(binary_preds, batch["clarities"], codes_order)
            all_true_clarity.extend(true_batch)
            all_pred_clarity.extend(pred_batch)
    return total_loss / len(loader), np.vstack(all_predictions), np.vstack(all_targets), all_true_clarity, all_pred_clarity


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    codes_order: list[str],
) -> dict[str, Any]:
    val_loss, predictions, targets, true_clarity, pred_clarity = collect_validation_predictions(
        model, loader, criterion, device, threshold, codes_order
    )
    results = compute_multilabel_metrics(targets, predictions)
    results.update(compute_clarity_metrics(true_clarity, pred_clarity))
    results["per_class"] = compute_per_class_metrics(targets, predictions, codes_order)
    results["val_loss"] = val_loss
    return results


def log_epoch_metrics(epoch: int, train_loss: float, eval_metrics: dict[str, Any]) -> None:
    logger.info("=" * 65)
    logger.info("РЕЗУЛЬТАТЫ ЭПОХИ %d:", epoch)
    logger.info("  Train Loss:       %.4f | Val Loss: %.4f", train_loss, eval_metrics["val_loss"])
    logger.info("  Macro F1:         %.4f (Целевая метрика)", eval_metrics["macro_f1"])
    logger.info("  Subset Accuracy:  %.4f (Exact Match)", eval_metrics["subset_accuracy"])
    logger.info("  Clarity Accuracy: %.4f", eval_metrics["clarity_accuracy"])
    logger.info("  False Clear Rate: %.2f%%", eval_metrics["false_clear_rate"])
    logger.info("=" * 65)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    macro_f1: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "macro_f1": macro_f1,
        "config": config,
    }
    torch.save(payload, path)


def save_metadata(
    path: Path, config: dict[str, Any], metrics: dict[str, Any], epoch: int, macro_f1: float
) -> None:
    metric_keys = ("macro_f1", "micro_f1", "subset_accuracy", "hamming_loss", "clarity_accuracy", "false_clear_rate")
    metadata = {
        "config": config,
        "best_epoch": epoch,
        "best_macro_f1": macro_f1,
        "metrics": {key: metrics[key] for key in metric_keys},
    }
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели NL2SQL Quality Gate")
    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2")
    parser.add_argument("--train_file", type=str, default="backend/data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="backend/data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="backend/models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    return parser.parse_args()


def resolve_data_path(file_path: str) -> Path:
    direct_path = Path(file_path)
    if direct_path.exists():
        return direct_path
    from_root = PROJECT_ROOT / file_path
    if from_root.exists():
        return from_root
    stripped = file_path.removeprefix("backend/").removeprefix("backend\\")
    from_stripped = PROJECT_ROOT / stripped
    if from_stripped.exists():
        return from_stripped
    return direct_path


def create_data_loader(
    dataset: NL2SQLDataset,
    collator: NL2SQLDataCollator,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_training_artifacts(
    arguments: argparse.Namespace,
    train_dataset: NL2SQLDataset,
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any, torch.amp.GradScaler | None]:
    model = NL2SQLQualityModel(
        model_name=arguments.model_name,
        num_codes=len(CODES_ORDER),
        dropout_probability=arguments.dropout,
    ).to(device)
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.AdamW(build_optimizer_parameters(model, arguments.weight_decay), lr=arguments.lr)
    total_steps = len(train_loader) * arguments.epochs
    warmup_steps = int(total_steps * arguments.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    return model, criterion, optimizer, scheduler, scaler


def build_model_config(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": arguments.model_name,
        "num_codes": len(CODES_ORDER),
        "codes_order": CODES_ORDER,
        "threshold": arguments.threshold,
        "max_length": arguments.max_length,
    }


def update_best_checkpoint(
    epoch: int, eval_metrics: dict[str, Any], best_f1: float, patience: int,
    model: nn.Module, optimizer: torch.optim.Optimizer, output_dir: Path, model_config: dict[str, Any],
) -> tuple[float, int]:
    current_f1 = eval_metrics["macro_f1"]
    if current_f1 <= best_f1:
        return best_f1, patience + 1
    logger.info("Найден лучший результат Macro F1: %.4f. Сохранение чекпоинта...", current_f1)
    save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, current_f1, model_config)
    save_metadata(output_dir / "model_config.json", model_config, eval_metrics, epoch, current_f1)
    return current_f1, 0


def run_training_loop(
    arguments: argparse.Namespace, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
    scaler: torch.amp.GradScaler | None, device: torch.device, output_dir: Path,
) -> None:
    best_macro_f1, patience_counter = -1.0, 0
    model_config = build_model_config(arguments)
    for epoch in range(1, arguments.epochs + 1):
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch, arguments.epochs)
        eval_metrics = run_validation(model, val_loader, criterion, device, arguments.threshold, CODES_ORDER)
        log_epoch_metrics(epoch, train_loss, eval_metrics)
        best_macro_f1, patience_counter = update_best_checkpoint(
            epoch, eval_metrics, best_macro_f1, patience_counter, model, optimizer, output_dir, model_config
        )
        if patience_counter >= arguments.early_stopping_patience:
            logger.info("Ранняя остановка на эпохе %d.", epoch)
            break
    logger.info("Обучение завершено. Чекпоинт: %s", output_dir / "best_model.pt")


def prepare_data_loaders(arguments: argparse.Namespace, tokenizer: Any) -> tuple[NL2SQLDataset, DataLoader, DataLoader]:
    train_dataset = NL2SQLDataset(resolve_data_path(arguments.train_file), tokenizer, max_length=arguments.max_length)
    val_dataset = NL2SQLDataset(resolve_data_path(arguments.val_file), tokenizer, max_length=arguments.max_length)
    collator = NL2SQLDataCollator(tokenizer=tokenizer)
    train_loader = create_data_loader(train_dataset, collator, arguments.batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, collator, arguments.batch_size, shuffle=False)
    return train_dataset, train_loader, val_loader


def main() -> None:
    arguments = parse_arguments()
    set_seed(arguments.seed)
    torch.set_num_threads(arguments.cpu_threads)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство для обучения: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    train_dataset, train_loader, val_loader = prepare_data_loaders(arguments, tokenizer)
    model, criterion, optimizer, scheduler, scaler = build_training_artifacts(arguments, train_dataset, train_loader, device)
    run_training_loop(arguments, model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, output_dir)


if __name__ == "__main__":
    main()
