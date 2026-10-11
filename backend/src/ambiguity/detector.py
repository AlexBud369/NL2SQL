import logging

from src.schemas import AnalyzeResponse, MissingDimension

logger = logging.getLogger("src.ambiguity")


def analyze(question: str, db_type: str, schema: str) -> AnalyzeResponse:
    """
    ЗАГЛУШКА. Позже здесь будет вызов настоящей модели.
    """
    q = question.lower()

    missing: list[MissingDimension] = []
    reasons: list[str] = []

    if any(w in q for w in ["последн", "недавн", "за месяц"]):
        missing.append(
            MissingDimension(
                code="MISSING_TIME",
                description="Не указан конкретный период времени",
            )
        )
        reasons.append("«Последний месяц» неоднозначен без контекста")

    if "топ" in q and "по" not in q:
        missing.append(
            MissingDimension(
                code="MISSING_SORT",
                description="Не указано, по какому полю сортировать",
            )
        )
        reasons.append("Не указано, по какому критерию считать топ")

    if reasons:
        status = "incomplete"
        score = 0.3
    else:
        status = "clear"
        score = 1.0

    return AnalyzeResponse(
        ambiguity_status=status,
        completeness_score=score,
        missing_dimensions=missing,
        reasons=reasons,
        suggested_clarifications=["Уточните вопрос"] if reasons else [],
    )
