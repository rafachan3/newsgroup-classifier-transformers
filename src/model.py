"""DistilBERT encoder with a simple dropout + linear classification head."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertForNewsgroups(nn.Module):
    """DistilBERT with ``dropout(768) -> linear(768, num_labels)`` on [CLS] token.

    The pretrained DistilBERT body is left at default configuration; the *head*
    dropout rate is set from the training YAML (e.g. 0.1, 0.2, 0.3) for
    ablations.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        dim = int(self.bert.config.dim)
        self.classifier = nn.Linear(dim, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return unnormalized class logits, shape (batch, num_labels)."""
        out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def save_pretrained(self, path: str | Any, **kwargs: Any) -> None:
        """Delegate encoder save to underlying DistilBERT (optional helper)."""
        p = str(path)
        self.bert.save_pretrained(p, **kwargs)
