from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.constants import CODES_ORDER


class NL2SQLDataset(Dataset):
    CODES_ORDER: list[str] = CODES_ORDER

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: list[dict[str, Any]] = self._load_data(self.file_path)

    def _load_data(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def __len__(self) -> int:
        return len(self.data)

    def _format_input_prompt(self, item: dict[str, Any]) -> str:
        db_type = item.get("db_type", "postgresql")
        question = item.get("question", "")
        return f"СУБД: {db_type}. Запрос: {question}"

    def _create_target_tensor(self, codes: list[str]) -> torch.Tensor:
        target = torch.zeros(len(self.CODES_ORDER), dtype=torch.float32)
        for code in codes:
            if code in self.CODES_ORDER:
                target[self.CODES_ORDER.index(code)] = 1.0
        return target

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        prompt = self._format_input_prompt(item)
        schema = item.get("db_schema", "")

        encoding = self.tokenizer(
            prompt,
            schema,
            truncation="only_second",
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        sample: dict[str, Any] = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": self._create_target_tensor(item.get("codes", [])),
            "raw_id": item.get("id", f"sample_{index}"),
            "clarity": item.get("clarity", "clear"),
        }

        if "token_type_ids" in encoding:
            sample["token_type_ids"] = encoding["token_type_ids"]

        return sample


class NL2SQLDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = self.tokenizer.pad_token_id or 0

    def _compute_padded_length(self, input_sequences: list[list[int]]) -> int:
        longest = max(len(seq) for seq in input_sequences)
        if self.pad_to_multiple_of and longest % self.pad_to_multiple_of != 0:
            return ((longest // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        return longest

    def _pad_sequence(self, sequence: list[int], target_length: int, fill_value: int) -> list[int]:
        return sequence + [fill_value] * (target_length - len(sequence))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        raw_input_ids = [item["input_ids"] for item in batch]
        max_batch_length = self._compute_padded_length(raw_input_ids)

        padded_inputs = [
            self._pad_sequence(item["input_ids"], max_batch_length, self.pad_token_id)
            for item in batch
        ]
        padded_masks = [
            self._pad_sequence(item["attention_mask"], max_batch_length, 0) for item in batch
        ]

        result: dict[str, Any] = {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.stack([item["labels"] for item in batch]),
            "raw_ids": [item["raw_id"] for item in batch],
            "clarities": [item["clarity"] for item in batch],
        }

        if "token_type_ids" in batch[0]:
            padded_tokens = [
                self._pad_sequence(item["token_type_ids"], max_batch_length, 0) for item in batch
            ]
            result["token_type_ids"] = torch.tensor(padded_tokens, dtype=torch.long)

        return result
