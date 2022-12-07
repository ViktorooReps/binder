from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import HfArgumentParser

from datamodel.configuration import get_descriptions, DatasetName
from datamodel.reader.nerel import get_dataset_files, read_annotation, read_text
from datamodel.utils import DatasetType
from model.inference import InferenceBinder, evaluate


@dataclass
class EvaluationArguments:
    model: Path = field(metadata={'help': 'Model .pkl file to evaluate'})
    device: str = field(metadata={'help': 'cuda or cpu'})


if __name__ == '__main__':
    parser = HfArgumentParser(dataclass_types=[EvaluationArguments])
    eval_args, = parser.parse_args_into_dataclasses()
    eval_args: EvaluationArguments

    model = InferenceBinder.load(eval_args.model)
    model.to(torch.device(eval_args.device))

    # TODO: unify interface for all datasets

    text_files, annotation_files = get_dataset_files(Path('data/nerel'), DatasetType.TEST)
    test_categories = sorted(get_descriptions(DatasetName.NEREL).keys())

    ground_truth = list(map(read_annotation, annotation_files))
    texts = list(map(read_text, text_files))

    model_predictions = model(texts)

    evaluate(model_predictions, ground_truth, test_categories)
