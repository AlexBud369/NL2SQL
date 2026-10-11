import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ambiguity.detector import analyze
from src.config import settings
from src.logging_config import setup_logging
from src.schemas import AnalyzeRequest, AnalyzeResponse

setup_logging()
logger = logging.getLogger("src.main")

app = FastAPI(title="Ambiguity Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
    logger.info("=" * 70)
    logger.info("ВОПРОС: %s", payload.question)
    logger.info("СУБД:   %s", payload.db_type)
    schema_preview = (
        payload.db_schema[:200] + "..." if len(payload.db_schema) > 200 else payload.db_schema
    )
    logger.info("СХЕМА:  %s", schema_preview or "(пустая)")

    result = analyze(payload.question, payload.db_type, payload.db_schema)

    logger.info("СТАТУС: %s", result.ambiguity_status)
    logger.info("ПОЛНОТА: %.2f", result.completeness_score)
    for r in result.reasons:
        logger.info("ПРИЧИНА: %s", r)
    logger.info("=" * 70)

    return result
