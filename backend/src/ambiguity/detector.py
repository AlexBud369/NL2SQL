from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.constants import CODES_ORDER
from src.model import NL2SQLQualityModel
from src.postprocess import postprocess_predictions
from src.schemas import AnalyzeResponse, MissingDimension

logger = logging.getLogger("src.ambiguity")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "models" / "best_model.pt"


class AmbiguityDetectorService:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: NL2SQLQualityModel | None = None
        self.tokenizer: Any = None
        self.codes_order = CODES_ORDER
        self._initialize_service(checkpoint_path)

    def _initialize_service(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", "cointegrated/rubert-tiny2")
        self.codes_order = config.get("codes_order", CODES_ORDER)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = NL2SQLQualityModel(
            model_name=model_name,
            num_codes=len(self.codes_order),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _execute_neural_prediction(self, question: str, db_type: str, schema: str) -> list[float]:
        prompt = f"СУБД: {db_type}. Запрос: {question}"
        inputs = self.tokenizer(
            prompt,
            schema,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probabilities = self.model.predict_probabilities(**inputs)
        return probabilities.squeeze(0).cpu().tolist()

    def analyze(self, question: str, db_type: str, schema: str) -> AnalyzeResponse:
        syntax_codes: list[str] = []

        if self.model is not None and self.tokenizer is not None:
            probabilities = self._execute_neural_prediction(question, db_type, schema)
        else:
            probabilities = [0.0] * len(self.codes_order)

        result = postprocess_predictions(
            probabilities=probabilities,
            codes_order=self.codes_order,
            thresholds=0.5,
            syntax_codes=syntax_codes,
        )

        missing_dimensions = [
            MissingDimension(code=code, description=reason)
            for code, reason in zip(result["detected_codes"], result["reasons"])
        ]

        return AnalyzeResponse(
            ambiguity_status=result["ambiguity_status"],
            completeness_score=result["completeness_score"],
            missing_dimensions=missing_dimensions,
            reasons=result["reasons"],
            suggested_clarifications=result["suggested_clarifications"],
        )


detector_singleton = AmbiguityDetectorService()


def analyze(question: str, db_type: str, schema: str) -> AnalyzeResponse:
    return detector_singleton.analyze(question, db_type, schema)
