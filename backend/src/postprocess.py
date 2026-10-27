from __future__ import annotations

from typing import Any

from src.constants import (
    AMBIGUOUS_CODES,
    CLARIFICATION_TEMPLATES,
    INCOMPLETE_CODES,
    PENALTIES,
    REASON_TEMPLATES,
)


def derive_clarity(codes: list[str]) -> str:
    code_set = set(codes)
    if code_set & AMBIGUOUS_CODES:
        return "ambiguous"
    if code_set & INCOMPLETE_CODES:
        return "incomplete"
    return "clear"


def compute_completeness_score(codes: list[str]) -> float:
    total_penalty = sum(PENALTIES.get(code, 0.25) for code in codes)
    return round(max(0.0, min(1.0, 1.0 - total_penalty)), 2)


def get_reasons(codes: list[str]) -> list[str]:
    return [REASON_TEMPLATES[code] for code in codes if code in REASON_TEMPLATES]


def get_clarifications(codes: list[str]) -> list[str]:
    return [CLARIFICATION_TEMPLATES[code] for code in codes if code in CLARIFICATION_TEMPLATES]


def extract_active_codes(
    probabilities: list[float],
    codes_order: list[str],
    thresholds: float | dict[str, float] = 0.5,
) -> list[str]:
    active_codes: list[str] = []
    for probability, code in zip(probabilities, codes_order):
        cutoff = thresholds.get(code, 0.5) if isinstance(thresholds, dict) else thresholds
        if probability >= cutoff:
            active_codes.append(code)
    return active_codes


def postprocess_predictions(
    probabilities: list[float],
    codes_order: list[str],
    thresholds: float | dict[str, float] = 0.5,
    syntax_codes: list[str] | None = None,
) -> dict[str, Any]:
    neural_codes = extract_active_codes(probabilities, codes_order, thresholds)
    merged_codes = sorted(list(set(neural_codes + (syntax_codes or []))))

    return {
        "ambiguity_status": derive_clarity(merged_codes),
        "completeness_score": compute_completeness_score(merged_codes),
        "detected_codes": merged_codes,
        "reasons": get_reasons(merged_codes),
        "suggested_clarifications": get_clarifications(merged_codes),
    }
