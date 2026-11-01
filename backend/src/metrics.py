from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def compute_multilabel_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(targets, predictions)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
    }


def compute_per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    codes_order: list[str],
) -> dict[str, dict[str, float]]:
    precisions = precision_score(targets, predictions, average=None, zero_division=0)
    recalls = recall_score(targets, predictions, average=None, zero_division=0)
    f1_scores = f1_score(targets, predictions, average=None, zero_division=0)
    supports = targets.sum(axis=0)

    per_class: dict[str, dict[str, float]] = {}
    for code, p, r, f, s in zip(codes_order, precisions, recalls, f1_scores, supports):
        per_class[code] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": float(s),
        }
    return per_class


def compute_pr_auc_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    codes_order: list[str],
) -> dict[str, Any]:
    per_class_pr_auc: dict[str, float] = {}
    all_scores: list[float] = []

    for idx, code in enumerate(codes_order):
        y_true = targets[:, idx]
        y_score = probabilities[:, idx]
        if y_true.sum() > 0:
            score = float(average_precision_score(y_true, y_score))
        else:
            score = 1.0
        per_class_pr_auc[code] = round(score, 4)
        all_scores.append(score)

    return {
        "macro_pr_auc": round(float(np.mean(all_scores)), 4),
        "per_class_pr_auc": per_class_pr_auc,
    }


def compute_calibration_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> dict[str, float]:
    y_true_flat = targets.flatten()
    y_prob_flat = probabilities.flatten()

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true_flat)

    for i in range(num_bins):
        bin_mask = (y_prob_flat >= bin_boundaries[i]) & (y_prob_flat < bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true_flat[bin_mask])
            bin_conf = np.mean(y_prob_flat[bin_mask])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    brier = float(brier_score_loss(y_true_flat, y_prob_flat))
    return {
        "expected_calibration_error": round(float(ece), 4),
        "brier_score": round(brier, 4),
    }


def compute_clarity_metrics(
    true_clarity: list[str],
    predicted_clarity: list[str],
) -> dict[str, Any]:
    clarity_accuracy = accuracy_score(true_clarity, predicted_clarity)
    clarity_macro_f1 = f1_score(true_clarity, predicted_clarity, average="macro", zero_division=0)

    false_clears = sum(
        1 for t, p in zip(true_clarity, predicted_clarity) if t != "clear" and p == "clear"
    )
    total_defective = sum(1 for t in true_clarity if t != "clear")
    false_clear_rate = (false_clears / max(total_defective, 1)) * 100.0

    false_blocks = sum(
        1 for t, p in zip(true_clarity, predicted_clarity) if t == "clear" and p != "clear"
    )
    total_clear = sum(1 for t in true_clarity if t == "clear")
    false_block_rate = (false_blocks / max(total_clear, 1)) * 100.0

    labels = ["clear", "incomplete", "ambiguous"]
    cm = confusion_matrix(true_clarity, predicted_clarity, labels=labels)

    return {
        "clarity_accuracy": float(clarity_accuracy),
        "clarity_macro_f1": float(clarity_macro_f1),
        "false_clear_rate": float(false_clear_rate),
        "false_block_rate": float(false_block_rate),
        "confusion_matrix": cm.tolist(),
    }


def compute_score_metrics(
    true_scores: list[float],
    predicted_scores: list[float],
) -> dict[str, float]:
    mae = mean_absolute_error(true_scores, predicted_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, predicted_scores))
    return {"score_mae": float(mae), "score_rmse": float(rmse)}
