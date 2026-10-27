from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class NL2SQLQualityModel(nn.Module):
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        num_codes: int = 11,
        dropout_probability: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_codes = num_codes

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(p=dropout_probability)
        self.classifier = nn.Linear(self.hidden_size, num_codes)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**encoder_inputs)
        cls_representation = encoder_outputs.last_hidden_state[:, 0, :]
        regularized_features = self.dropout(cls_representation)
        return self.classifier(regularized_features)

    def predict_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits)
