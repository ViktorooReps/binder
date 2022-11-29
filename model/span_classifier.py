from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, Union, Optional, TypeVar, Iterable, Set, Dict

import torch
from torch import Tensor, BoolTensor, LongTensor
from torch.nn import Linear, Embedding
from torch.nn.functional import pad
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import EncodingFast

from datamodel.example import Example, strided_split, TypedSpan, collate_examples
from model.encoder import EntityDescriptionEncoder
from model.loss import ContrastiveThresholdLoss
from model.metric import Metric, CosSimilarity
from model.serializable import SerializableModel


@dataclass
class ModelArguments:
    bert_model: str = field(metadata={'help': 'Name of the BERT HuggingFace model to use.'})
    hidden_size: int = field(default=128, metadata={'help': 'Hidden representations of vector spaces.'})
    max_sequence_length: int = field(default=128, metadata={'help': 'Maximum length in tokens of an example.'})
    max_entity_length: int = field(default=30, metadata={'help': 'Maximum length in tokens of an entity.'})
    loss_beta: float = field(default=0.6, metadata={'help': 'Beta coefficient in loss function.'})
    start_coef: float = field(default=0.2, metadata={'help': 'Weight of start loss.'})
    end_coef: float = field(default=0.2, metadata={'help': 'Weight of end loss.'})
    span_coef: float = field(default=0.6, metadata={'help': 'Weight of span loss.'})


_Model = TypeVar('_Model', bound='SpanClassifier')


class SpanClassifier(SerializableModel):

    def __init__(
            self,
            bert_model: str,
            descriptions: List[str],
            unk_entity_type_id: int,
            metric: Metric,
            hidden_size: int = 128,
            max_sequence_length: int = 128,
            max_entity_length: int = 30,
            loss_beta: float = 0.6,
            start_coef: float = 0.2,
            end_coef: float = 0.2,
            span_coef: float = 0.6
    ):
        super(SpanClassifier, self).__init__()

        assert start_coef + end_coef + span_coef == 1.0
        self._metric = metric
        self._unk_entity_label_id = unk_entity_type_id
        self._loss_fn_factory = partial(
            ContrastiveThresholdLoss,
            unk_id=unk_entity_type_id,
            ignore_id=-100,
            reduction='mean',
            beta=loss_beta
        )
        self._loss_fn: Optional[ContrastiveThresholdLoss] = None
        self._start_coef = start_coef
        self._end_coef = end_coef
        self._span_coef = span_coef

        self._token_encoder: PreTrainedModel = AutoModel.from_pretrained(bert_model)
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(bert_model)

        self._hidden_size = hidden_size
        self._encoder_hidden = self._token_encoder.config.hidden_size
        self._max_sequence_length = min(max_sequence_length, self._tokenizer.model_max_length)
        self._max_entity_length = max_entity_length

        self._descriptions_encoder = EntityDescriptionEncoder(bert_model)
        self._entity_projection = Linear(self._encoder_hidden, self._hidden_size)
        self._encoded_descriptions: Optional[BatchEncoding] = None
        self._frozen_entity_representations: Optional[Tensor] = None
        self.add_descriptions(descriptions)

        self._length_embedding = Embedding(self._max_entity_length, self._hidden_size)
        self._span_projection = Linear(self._encoder_hidden * 2 + self._hidden_size, self._hidden_size)

        self._token_start_projection = Linear(self._encoder_hidden, self._hidden_size)
        self._token_end_projection = Linear(self._encoder_hidden, self._hidden_size)
        self._entity_start_projection = Linear(self._encoder_hidden, self._hidden_size)
        self._entity_end_projection = Linear(self._encoder_hidden, self._hidden_size)

    @classmethod
    def from_args(cls: _Model, args: ModelArguments, descriptions: List[str], unk_entity_type_id: int) -> _Model:
        return cls(
            args.bert_model, descriptions, unk_entity_type_id, CosSimilarity(scale=1.0, freeze_scale=False),
            hidden_size=args.hidden_size,
            max_sequence_length=args.max_sequence_length,
            max_entity_length=args.max_entity_length,
            loss_beta=args.loss_beta,
            start_coef=args.start_coef,
            end_coef=args.end_coef,
            span_coef=args.span_coef
        )

    def freeze_descriptions(self):
        if self._frozen_entity_representations is not None:
            return

        with torch.no_grad():
            self._frozen_entity_representations = self._descriptions_encoder(self._encoded_descriptions)

    def add_descriptions(self, descriptions: List[str]) -> None:
        self._encoded_descriptions = self._descriptions_encoder.prepare_inputs(descriptions)
        self._loss_fn = self._loss_fn_factory(len(descriptions))
        self._frozen_entity_representations = None

    def eval(self: _Model) -> _Model:
        super(SpanClassifier, self).eval()
        self.freeze_descriptions()
        return self

    def train(self: _Model, mode: bool = True) -> _Model:
        super(SpanClassifier, self).train(mode)
        self._frozen_entity_representations = None
        return self

    def _get_entity_representations(self) -> Tensor:
        if self._frozen_entity_representations is not None:
            return self._frozen_entity_representations
        return self._descriptions_encoder(self._encoded_descriptions)

    def _get_length_representations(self, token_representations: Tensor) -> Tensor:
        batch_size, sequence_length, representation_dims = token_representations.shape

        length_embeddings = self._length_embedding(torch.arange(self._max_entity_length, device=self.device))
        return length_embeddings.reshape(1, 1, self._max_entity_length, self._hidden_size).repeat(batch_size, sequence_length, 1, 1)

    def _get_span_representations(self, token_representations: Tensor) -> Tuple[Tensor, BoolTensor]:
        batch_size, sequence_length, representation_dims = token_representations.shape

        padding_masks: List[BoolTensor] = []
        span_end_representations: List[Tensor] = []

        for shift in range(self._max_entity_length):  # self._max_entity_length ~ 20-30, so it is fine to not vectorize this
            span_end_representations.append(torch.roll(token_representations, -shift, 1).unsqueeze(-2))

            padding_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=token_representations.device)
            padding_mask[:, -shift:] = False
            padding_masks.append(padding_mask.unsqueeze(-1))

        padding_mask = torch.concat(padding_masks, dim=-1).bool()
        span_end_representation = torch.concat(span_end_representations, dim=-2)
        span_start_representation = token_representations.unsqueeze(-2).repeat(1, 1, self._max_entity_length, 1)

        span_length_embedding = self._get_length_representations(token_representations)

        span_representation = torch.concat([span_start_representation, span_end_representation, span_length_embedding], dim=-1)
        return self._span_projection(span_representation), padding_mask

    def _compute_sim_scores(self, representations: Tensor, entity_representations: Tensor, *, span_level: bool = True) -> Tensor:
        n_classes = len(self._encoded_descriptions.encodings)

        representations = representations.unsqueeze(-2)  # (B, S, M, 1, H) or (B, S, 1, H)
        new_shape = (1, 1, 1, n_classes, self._hidden_size) if span_level else (1, 1, n_classes, self._hidden_size)
        entity_representations = entity_representations.reshape(*new_shape)
        return self._metric(representations, entity_representations)

    def prepare_inputs(
            self,
            texts: List[str],
            entities: List[Set[TypedSpan]],
            *,
            stride: float = 1/8,
            category_mapping: Dict[str, int],
            no_entity_category: str,
    ) -> Iterable[Example]:

        encodings: Optional[List[EncodingFast]] = self._tokenizer(
            texts,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True
        ).encodings
        if encodings is None:
            raise ValueError(f'Tokenizer {self._bert_tokenizer} is not fast! Use fast tokenizer!')

        for text_idx, (encoding, entities) in enumerate(zip(encodings, entities)):
            yield from strided_split(
                text_idx, encoding, entities,
                stride=int(self._max_sequence_length * stride),
                max_sequence_length=self._max_sequence_length,
                max_entity_length=self._max_entity_length,
                category_mapping=category_mapping,
                no_entity_category=no_entity_category,
                cls_token_id=self._tokenizer.cls_token_id
            )

    def collate_examples(self, examples: Iterable[Example]) -> Dict[str, Optional[LongTensor]]:
        return collate_examples(
            examples,
            padding_token_id=self._tokenizer.pad_token_id,
            pad_length=self._max_sequence_length,
            return_batch_examples=False
        )

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> Union[LongTensor, Tuple[Tensor, LongTensor]]:
        """Predicts entity type ids. If true label ids are given, calculates loss as well."""

        batch_size, sequence_length = input_ids.shape
        input_ids = pad(input_ids, [0, 0, 0, self._max_sequence_length - sequence_length], value=self._tokenizer.pad_token_id)

        token_representations = self._token_encoder(input_ids=input_ids.to(self.device)).last_hidden_state  # (B, S, E)
        entity_representations = self._get_entity_representations()  # (C, E)

        span_representations, span_padding = self._get_span_representations(token_representations)
        span_scores = self._compute_sim_scores(span_representations, self._entity_projection(entity_representations))

        batch_size, _, _, n_classes = span_scores.shape

        entity_threshold = span_scores[:, 0, 0].reshape(batch_size, 1, 1, n_classes)  # is a sim score with [CLS] token and entities
        mask = (span_scores > entity_threshold)
        scores_mask = torch.full_like(span_scores, fill_value=-torch.inf)
        scores_mask[mask] = 0  # set to -inf scores that did not pass the threshold

        values, predictions = torch.max(span_scores + scores_mask, dim=-1)
        predictions[values == -torch.inf] = self._unk_entity_label_id

        if labels is None:
            return predictions
        labels = labels.to(self.device)

        # (C, H)
        entity_start_representations = self._entity_start_projection(entity_representations)
        entity_end_representations = self._entity_end_projection(entity_representations)

        # (B, S, H)
        token_start_representations = self._token_start_projection(token_representations)
        token_end_representations = self._token_start_projection(token_representations)

        start_scores = self._compute_sim_scores(token_start_representations, entity_start_representations, span_level=False)
        end_scores = self._compute_sim_scores(token_end_representations, entity_end_representations, span_level=False)

        span_loss = self._loss_fn(span_scores, labels)
        start_loss = self._loss_fn(start_scores.unsqueeze(-2).repeat(1, 1, self._max_entity_length, 1), labels)
        end_loss = self._loss_fn(end_scores.unsqueeze(-2).repeat(1, 1, self._max_entity_length, 1), labels)

        return self._span_coef * span_loss + self._start_coef * start_loss + self._end_coef * end_loss, predictions

