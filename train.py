import logging
from functools import partial
from pprint import pprint
from typing import Dict, Any

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, HfArgumentParser, TrainingArguments, EvalPrediction

from datamodel.configuration import DatasetArguments, get_datasets, DatasetName, get_descriptions
from datamodel.utils import invert
from model.span_classifier import ModelArguments, SpanClassifier

logger = logging.getLogger(__name__)


def compute_metrics(
        evaluation_results: EvalPrediction,
        category_id_mapping: Dict[int, str],
        no_entity_category_id: int,
) -> Dict[str, float]:

    padding_mask = (evaluation_results.label_ids != -100)

    label_ids = evaluation_results.label_ids[padding_mask]
    predictions = evaluation_results.predictions[padding_mask]

    unique_label_ids = set(np.unique(label_ids[label_ids != no_entity_category_id]))

    labels = sorted(category_id_mapping.keys())
    f1_category_scores = f1_score(label_ids, predictions, average=None, labels=labels, zero_division=0)
    recall_category_scores = recall_score(label_ids, predictions, average=None, labels=labels, zero_division=0)
    precision_category_scores = precision_score(label_ids, predictions, average=None, labels=labels, zero_division=0)

    results: Dict[str, float] = {}
    sum_f1 = 0
    sum_recall = 0
    sum_precision = 0
    for category_id, f1, recall, precision in zip(labels, f1_category_scores, recall_category_scores, precision_category_scores):
        if category_id == no_entity_category_id:
            logger.info(f'O: {f1}, {recall}, {precision}')
            continue

        if category_id not in unique_label_ids:
            logger.info(f'Skipping {category_id}: {f1}, {recall}, {precision}')
            continue

        category = category_id_mapping[category_id]
        results[f'F1_{category}'] = f1
        results[f'Recall_{category}'] = recall
        results[f'Precision_{category}'] = precision

        sum_f1 += f1
        sum_recall += recall
        sum_precision += precision

    num_categories = len(category_id_mapping) - 1

    results['F1_macro'] = sum_f1 / num_categories
    results['Recall_macro'] = sum_recall / num_categories
    results['Precision_macro'] = sum_precision / num_categories
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tb_writer = SummaryWriter()

    parser = HfArgumentParser(dataclass_types=[ModelArguments, DatasetArguments, TrainingArguments])
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    training_args: TrainingArguments
    dataset_args: DatasetArguments
    model_args: ModelArguments

    dataset_name = DatasetName(dataset_args.dataset_name)

    unk_category_id = -1
    unk_category = '<UNK>'

    descriptions = get_descriptions(dataset_name)
    category_names = sorted(descriptions.keys())
    category_descriptions = list(map(descriptions.__getitem__, category_names))
    category_id_mapping = dict(enumerate(category_names))
    category_id_mapping[unk_category_id] = unk_category
    category_mapping = invert(category_id_mapping)

    model: SpanClassifier = SpanClassifier.from_args(model_args, category_descriptions, unk_entity_type_id=unk_category_id)

    example_encoder = partial(model.prepare_inputs, category_mapping=category_mapping, no_entity_category=unk_category)
    train_dataset, dev_dataset, test_dataset = get_datasets(dataset_name, example_encoder=example_encoder)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=model.collate_examples,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=partial(
            compute_metrics,
            category_id_mapping=category_id_mapping,
            no_entity_category_id=-1
        )
    )
    trainer.train()

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    pprint(metrics)

    def normalize(d: Dict[str, Any]) -> Dict[str, str]:
        return {k: str(v) for k, v in d.items()}

    tb_writer.add_hparams(hparam_dict={**normalize(model_args.__dict__), **normalize(training_args.__dict__)}, metric_dict=metrics)
