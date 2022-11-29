from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Tuple, Callable, List, Set, Iterable, Dict

from datamodel.dataset import NERDataset
from datamodel.example import TypedSpan, Example
from datamodel.reader.nerel import read_nerel, read_descriptions
from datamodel.utils import DatasetType


class DatasetName(Enum):
    NEREL = 'nerel'
    ACE2005 = 'ace2005'


DATASET_READERS = {
    DatasetName.NEREL: partial(read_nerel, dataset_dir=Path('data/nerel'))
}

DATASET_DESCRIPTIONS = {
    DatasetName.NEREL: partial(read_descriptions, descriptions_path=Path('data/nerel/descriptions.json'))
}


@dataclass
class DatasetArguments:
    dataset_name: str = field(metadata={'help': f'Dataset name. One of {[dn.value for dn in DatasetName]}'})


def get_datasets(
        dataset_name: DatasetName,
        example_encoder: Callable[[List[str], List[Set[TypedSpan]]], Iterable[Example]],
) -> Tuple[NERDataset, NERDataset, NERDataset]:
    reader = DATASET_READERS[dataset_name]

    return (
        NERDataset(example_encoder(*reader(dataset_type=DatasetType.TRAIN))),
        NERDataset(example_encoder(*reader(dataset_type=DatasetType.DEV))),
        NERDataset(example_encoder(*reader(dataset_type=DatasetType.TEST)))
    )


def get_descriptions(dataset_name: DatasetName) -> Dict[str, str]:
    return DATASET_DESCRIPTIONS[dataset_name]()

