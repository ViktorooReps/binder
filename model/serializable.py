import pickle
from pathlib import Path
from typing import TypeVar, Type

import torch
from torch.nn import Module, Parameter


_ModelType = TypeVar('_ModelType', bound=Module)


class SerializableModel(Module):

    def __init__(self):
        super().__init__()
        self._dummy_param = Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device

    def save(self, save_path: Path) -> None:
        previous_device = self.device
        self.cpu()
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.to(previous_device)

    @classmethod
    def load(cls: Type[_ModelType], load_path: Path) -> _ModelType:
        with open(load_path, 'rb') as f:
            return pickle.load(f)
