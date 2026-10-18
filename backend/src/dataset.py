import json

import torch
from torch.utils.data import Dataset


class NL2SQLDataset(Dataset):
    CODES_ORDER = [
        "UNKNOWN_TABLE",
        "UNKNOWN_COLUMN",
        "SCOPE",
        "VAGUE",
        "ATTACH",
        "MISSING_TIME",
        "MISSING_VALUE",
        "MISSING_CRITERION",
        "MISSING_AGG",
        "MISSING_SORT",
        "MISSING_LIMIT",
    ]

    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["question"],
            item["db_schema"],
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        labels = torch.zeros(len(self.CODES_ORDER), dtype=torch.float)
        for code in item["codes"]:
            if code in self.CODES_ORDER:
                labels[self.CODES_ORDER.index(code)] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "labels": labels,
        }
