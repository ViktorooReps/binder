from typing import Dict, List, Set, TypeVar

import torch
from torch import Tensor
from torch.nn import Module

from datamodel.example import TypedSpan
from model.encoder import EntityDescriptionEncoder
from model.metric import Metric
from model.serializable import SerializableModel


_Model = TypeVar('_Model', bound=Module)


class InferenceBinder(SerializableModel):

    def __init__(self, description_encoder: EntityDescriptionEncoder, span_encoder: SpanEncoder, metric: Metric):
        super().__init__()
        self._description_encoder = description_encoder
        self._encoded_descriptions: Dict[str, Tensor] = {}

        self._span_encoder = span_encoder
        self._metric = metric

    def train(self: _Model, mode: bool = True) -> _Model:
        raise NotImplementedError

    @torch.no_grad()
    def add_entity_types(self, descriptions: Dict[str, str]) -> None:
        names, texts = map(list, zip(*descriptions.items()))
        entity_representations = self._description_encoder(texts)
        self._encoded_descriptions.update(zip(names, entity_representations))

    @torch.no_grad()
    def forward(self, texts: List[str]) -> List[Set[TypedSpan]]:
        encoder_inputs = self._span_encoder.prepare_inputs(texts)
        span_representations = self._span_encoder()

