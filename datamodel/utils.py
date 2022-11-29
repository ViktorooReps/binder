from enum import Enum
from typing import TypeVar, Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad

_K = TypeVar('_K')
_V = TypeVar('_V')


def invert(d: Dict[_K, _V]) -> Dict[_V, _K]:
    return {v: k for k, v in d.items()}


_Tensor = TypeVar('_Tensor', bound=Tensor)


def pad_images(images: List[_Tensor], *, padding_value: Any = 0.0, padding_length: Tuple[Optional[int], Optional[int]]) -> _Tensor:
    """Pad images to equal length (maximum height and width)."""
    max_height, max_width = padding_length

    shapes = torch.tensor(list(map(lambda t: t.shape, images)), dtype=torch.long).transpose(0, 1)
    max_height = shapes[-2].max() if max_height is None else max_height
    max_width = shapes[-1].max() if max_width is None else max_width

    ignore_dims = len(images[0].shape) - 2

    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        pad(img, [*([0, 0] * ignore_dims), 0, max_width - img.shape[-1], 0, max_height - img.shape[-2]], value=padding_value)
        for img in images
    ]
    return torch.stack(image_batch)


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class DatasetType(str, Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'
