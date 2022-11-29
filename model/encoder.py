from typing import List

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from model.serializable import SerializableModel


class EntityDescriptionEncoder(SerializableModel):

    def __init__(self, bert_model: str):
        super(EntityDescriptionEncoder, self).__init__()
        self._bert_encoder: PreTrainedModel = AutoModel.from_pretrained(bert_model)
        self._bert_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(bert_model)

    def prepare_inputs(self, descriptions: List[str]) -> BatchEncoding:
        return self._bert_tokenizer(
            descriptions,
            return_tensors='pt',
            truncation=True,
            max_length=self._bert_encoder.config.max_length,
            add_special_tokens=True
        )

    def forward(self, encoding: BatchEncoding) -> Tensor:
        return self._bert_encoder(**encoding.to(self.device)).last_hidden_state[:, 0]  # select [CLS] token representation
