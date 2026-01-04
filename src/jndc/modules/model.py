import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class ConcatCLSModel(BertPreTrainedModel):
    def __init__(self, config, drop_rate=0.5):
        super().__init__(config)
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(drop_rate)
        config.output_hidden_states = True
        self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embeddings = torch.cat([outputs.hidden_states[-i][:, 0] for i in range(1, 5)], dim=1)
        pooled_output = self.dropout(cls_embeddings)
        logits = self.classifier(pooled_output).softmax(dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }
