"""ruBERT with one token-classification head per nesting level."""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

from levels import NUM_LEVELS


class MultiLevelBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_levels = int(getattr(config, "num_levels", NUM_LEVELS))
        self.bert = BertModel(config, add_pooling_layer=False)
        dropout = getattr(config, "classifier_dropout", None)
        if dropout is None:
            dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.num_labels) for _ in range(self.num_levels)]
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        logits = torch.stack([classifier(hidden) for classifier in self.classifiers], dim=1)
        loss = None
        if labels is not None:
            losses = []
            for level in range(self.num_levels):
                losses.append(nn.functional.cross_entropy(
                    logits[:, level].reshape(-1, logits.size(-1)),
                    labels[:, level].reshape(-1),
                    ignore_index=-100,
                ))
            loss = torch.stack(losses).mean()
        return {"loss": loss, "logits": logits}
