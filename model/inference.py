import logging
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os import cpu_count, environ
from pathlib import Path
from typing import Dict, List, Set, TypeVar, Optional

import torch
from torch import Tensor, LongTensor
from torch.nn import Module
from torch.nn.functional import pad
from torch.onnx import export
from transformers import TensorType
from transformers.convert_graph_to_onnx import quantize
from transformers.onnx import FeaturesManager, OnnxConfig

from datamodel.example import TypedSpan, batch_examples, BatchedExamples
from datamodel.utils import invert, to_numpy
from model.serializable import SerializableModel
from model.span_classifier import SpanClassifier


torch.set_num_threads(cpu_count() // 2)

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count() // 2)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

logger = logging.getLogger(__name__)

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
except:
    logger.warning('Could not import ONNX inference tools!')


_Model = TypeVar('_Model', bound=Module)


class InferenceBinder(SerializableModel):

    def __init__(
            self,
            span_classifier: SpanClassifier,
            category_mapping: Dict[str, int],
            no_entity_category: str,
            max_sequence_length: int,
            max_entity_length: int
    ):
        super().__init__()
        self._classifier = span_classifier
        self._classifier.eval()
        self._classifier.drop_descriptions_encoder()
        self._classifier.train = None

        self._category_mapping = deepcopy(category_mapping)
        self._category_id_mapping = invert(self._category_mapping)
        self._no_entity_category = no_entity_category

        self._max_sequence_length = max_sequence_length
        self._max_entity_length = max_entity_length

    def train(self: _Model, mode: bool = True) -> _Model:
        raise NotImplementedError

    @torch.no_grad()
    def add_entity_types(self, descriptions: Dict[str, str]) -> None:
        names, texts = map(list, zip(*descriptions.items()))
        entity_representations = self._description_encoder(texts)
        self._encoded_descriptions.update(zip(names, entity_representations))

    @torch.no_grad()
    def forward(self, texts: List[str]) -> List[Set[TypedSpan]]:
        stride = 1/8
        examples = list(self._classifier.prepare_inputs(
            texts, [None] * len(texts),
            category_mapping=self._category_mapping,
            no_entity_category=self._no_entity_category,
            stride=stride
        ))

        no_entity_category_id = self._category_mapping[self._no_entity_category]

        predictions_collector = [defaultdict(int) for _ in texts]
        total_predictions = [0 for _ in texts]
        for batch in batch_examples(
                examples,
                batch_size=1,
                collate_fn=partial(self._classifier.collate_examples, return_batch_examples=True)
        ):
            predictions: LongTensor = self._classifier(**batch)
            entities_mask = ((predictions != no_entity_category_id) & (predictions != -100))

            batched_examples: BatchedExamples = batch['examples']

            batch_size, length = predictions.shape
            span_start = pad(
                batched_examples.start_offset,
                [0, self._max_sequence_length - length],
                value=-100
            ).view(batch_size, self._max_sequence_length, 1).repeat(1, 1, self._max_entity_length)

            span_end = []
            for shift in range(self._max_entity_length):  # self._max_entity_length ~ 20-30, so it is fine to not vectorize this
                span_end.append(torch.roll(batched_examples.end_offset, -shift, 1).unsqueeze(-2))
            span_end = torch.concat(span_end, dim=-1)

            entity_text_ids = torch.tensor(batched_examples.text_ids).view(batch_size, 1, 1).repeat(1, self._max_sequence_length, self._max_entity_length)

            chosen_text_ids = entity_text_ids[entities_mask]
            chosen_category_ids = predictions[entities_mask]
            chosen_span_starts = span_start[entities_mask]
            chosen_span_ends = span_end[entities_mask]
            for text_id, category_id, start, end in zip(chosen_text_ids, chosen_category_ids, chosen_span_starts, chosen_span_ends):
                predictions_collector[text_id][TypedSpan(start.item(), end.item(), self._category_id_mapping[category_id.item()])] += 1
                total_predictions[text_id] += 1

        all_entities = [set() for _ in texts]
        for text_id, preds in enumerate(predictions_collector):
            for entity, count_preds in preds.items():
                if count_preds > total_predictions[text_id] / 2:
                    all_entities[text_id].add(entity)

        return all_entities

    def optimize(
            self,
            onnx_dir: Path,
            quant: bool = True,
            opset_version: int = 13,
            do_constant_folding: bool = True
    ) -> None:
        onnx_model_path = onnx_dir.joinpath('model.onnx')

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self._encoder)
        onnx_config: OnnxConfig = model_onnx_config(self._classifier._token_encoder.config)

        model_inputs = onnx_config.generate_dummy_inputs(self._classifier._tokenizer, framework=TensorType.PYTORCH)
        dynamic_axes = {0: 'batch', 1: 'sequence'}
        # export to onnx
        export(
            self._encoder,
            ({'input_ids': model_inputs['input_ids']},),
            f=onnx_model_path.as_posix(),
            verbose=False,
            input_names=('input_ids',),
            output_names=('last_hidden_state',),
            dynamic_axes={'input_ids': dynamic_axes, 'last_hidden_state': dynamic_axes},
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
        )

        if quant:
            onnx_model_path = quantize(onnx_model_path)

        self._classifier._token_encoder = ONNXOptimizedEncoder(onnx_model_path)


class ONNXOptimizedEncoder(Module):

    def __init__(self, onnx_path: Path):
        super().__init__()
        self._onnx_path = onnx_path
        self._session: Optional[InferenceSession] = None

    def __getstate__(self):
        state = deepcopy(self.__dict__)
        state.pop('_session')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._session = None

    def _start_session(self) -> None:
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        self._session = InferenceSession(self._onnx_path.as_posix(), options, providers=['CPUExecutionProvider'])
        self._session.disable_fallback()

    def forward(self, input_ids: LongTensor, **_) -> Dict[str, Tensor]:
        if self._session is None:
            logger.info(f'Starting inference session for {self._onnx_path}.')
            start_time = time.time()
            self._start_session()
            logger.info(f'Inference started in {time.time() - start_time:.4f}s.')

        # Run the model (None = get all the outputs)
        return {
            'last_hidden_state': torch.tensor(self._session.run(
                None,
                {
                    'input_ids': to_numpy(input_ids)
                }
            )[0])
        }
