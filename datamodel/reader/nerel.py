import json
from pathlib import Path
from typing import Set, List, Tuple, Dict

from datamodel.example import TypedSpan
from datamodel.utils import DatasetType


def read_annotation(annotation_file: Path) -> Set[TypedSpan]:
    collected_annotations: Set[TypedSpan] = set()
    with open(annotation_file) as f:
        for line in f:
            if line.startswith('T'):
                _, span_info, value = line.strip().split('\t')

                if ';' not in span_info:  # skip multispan
                    category, start, end = span_info.split(' ')
                    collected_annotations.add(TypedSpan(int(start), int(end), category))

    return collected_annotations


def read_text(text_file: Path) -> str:
    with open(text_file) as f:
        return f.read()


def get_dataset_files(
        dataset_dir: Path,
        dataset_type: DatasetType,
        *,
        exclude_filenames: Set[str] = None
) -> Tuple[List[Path], List[Path]]:

    if exclude_filenames is None:
        exclude_filenames = set()

    dataset_dir = dataset_dir.joinpath(dataset_type.value)

    if not dataset_dir.exists():
        raise RuntimeError(f'Dataset directory {dataset_dir} does not exist!')

    if not dataset_dir.is_dir():
        raise RuntimeError(f'Provided path {dataset_dir} is not a directory!')

    def is_not_excluded(file: Path) -> bool:
        return file.with_suffix('').name not in exclude_filenames

    return sorted(filter(is_not_excluded, dataset_dir.glob('*.txt'))), sorted(filter(is_not_excluded, dataset_dir.glob('*.ann')))


def read_nerel(
        dataset_dir: Path,
        dataset_type: DatasetType,
        *,
        exclude_filenames: Set[str] = None
) -> Tuple[List[str], List[Set[TypedSpan]]]:

    text_files, annotation_files = get_dataset_files(dataset_dir, dataset_type, exclude_filenames=exclude_filenames)
    return list(map(read_text, text_files)), list(map(read_annotation, annotation_files))


def read_descriptions(descriptions_path: Path) -> Dict[str, str]:
    with open(descriptions_path) as f:
        return json.load(f)
