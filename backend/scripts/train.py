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
        for code in item.get("codes", []):
            if code in dataset.CODES_ORDER:
                positives[dataset.CODES_ORDER.index(code)] += 1.0

    weights = np.zeros(len(dataset.CODES_ORDER), dtype=np.float32)
    for index in range(len(dataset.CODES_ORDER)):
        pos = max(positives[index], 1.0)
        neg = total_count - pos
        weights[index] = min(neg / pos, max_weight)

    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer_parameters(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


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
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
    labels = batch["labels"].to(device)

    use_amp = device.type == "cuda" and scaler is not None
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(**inputs)
        loss = criterion(logits, labels)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
    return float(loss.item())


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    total_epochs: int,
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


def collect_validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    codes_order: list[str],
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_true_clarity, all_pred_clarity = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Валидация", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
            labels = batch["labels"].to(device)

            logits = model(**inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            binary_preds = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
            all_preds.append(binary_preds)
            all_targets.append(labels.cpu().numpy().astype(int))

            for pred_row, true_clarity in zip(binary_preds, batch["clarities"]):
                detected = [codes_order[i] for i, val in enumerate(pred_row) if val == 1]
                all_pred_clarity.append(derive_clarity(detected))
                all_true_clarity.append(true_clarity)

    return total_loss / len(loader), np.vstack(all_preds), np.vstack(all_targets), all_true_clarity, all_pred_clarity


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    codes_order: list[str],
) -> dict[str, Any]:
    val_loss, preds, targets, true_clarity, pred_clarity = collect_validation_predictions(
        model, loader, criterion, device, threshold, codes_order
    )
    results = compute_multilabel_metrics(targets, preds)
    results.update(compute_clarity_metrics(true_clarity, pred_clarity))
    results["per_class"] = compute_per_class_metrics(targets, preds, codes_order)
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "macro_f1": macro_f1,
            "config": config,
        },
        path,
    )


def save_metadata(path: Path, config: dict[str, Any], metrics: dict[str, Any], epoch: int, macro_f1: float) -> None:
    data = {
        "config": config,
        "best_epoch": epoch,
        "best_macro_f1": macro_f1,
        "metrics": {
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "subset_accuracy": metrics["subset_accuracy"],
            "hamming_loss": metrics["hamming_loss"],
            "clarity_accuracy": metrics["clarity_accuracy"],
            "false_clear_rate": metrics["false_clear_rate"],
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


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


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_data_path(args.train_file)
    val_path = resolve_data_path(args.val_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство для обучения: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = NL2SQLDataset(train_path, tokenizer, max_length=args.max_length)
    val_dataset = NL2SQLDataset(val_path, tokenizer, max_length=args.max_length)
    collator = NL2SQLDataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = NL2SQLQualityModel(
        model_name=args.model_name,
        num_codes=len(CODES_ORDER),
        dropout_probability=args.dropout,
    ).to(device)

    weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    optimizer_grouped_params = build_optimizer_parameters(model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_macro_f1 = -1.0
    patience_counter = 0
    checkpoint_file = output_dir / "best_model.pt"
    config_file = output_dir / "model_config.json"

    model_config = {
        "model_name": args.model_name,
        "num_codes": len(CODES_ORDER),
        "codes_order": CODES_ORDER,
        "threshold": args.threshold,
        "max_length": args.max_length,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_single_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch, args.epochs
        )
        eval_metrics = run_validation(model, val_loader, criterion, device, args.threshold, CODES_ORDER)
        log_epoch_metrics(epoch, train_loss, eval_metrics)

        current_f1 = eval_metrics["macro_f1"]
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            patience_counter = 0
            logger.info("Найден лучший результат Macro F1: %.4f. Сохранение чекпоинта...", best_macro_f1)
            save_checkpoint(checkpoint_file, model, optimizer, epoch, best_macro_f1, model_config)
            save_metadata(config_file, model_config, eval_metrics, epoch, best_macro_f1)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info("Ранняя остановка на эпохе %d.", epoch)
                break

    logger.info("Обучение завершено. Чекпоинт: %s", checkpoint_file)


if __name__ == "__main__":
    main()
