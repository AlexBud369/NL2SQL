import torch.nn as nn
from transformers import AutoModel


class NL2SQLQualityModel(nn.Module):
    def __init__(self, model_name: str, num_codes: int = 11):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_codes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
