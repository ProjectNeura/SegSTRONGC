from os import listdir as _listdir
from os.path import isdir as _isdir
from random import choice as _choice
from typing import Sequence as _Sequence

from cv2 import imread as _imread
from numpy import ndarray as _ndarray
from torch import Tensor as _Tensor, tensor as _tensor

from augmentation.autoaugment import *


def select_target(src: str) -> str:
    return select_target(src) if _isdir(src := f"src/{_choice(_listdir(src))}") else src


def select_targets(src: str, n: int) -> tuple[str, ...]:
    return tuple(select_target(src) for _ in range(n))


def load_targets(targets: _Sequence[str]) -> list[_ndarray]:
    return [_imread(target) for target in targets]


def load_targets_tensor(targets: _Sequence[str]) -> list[_Tensor]:
    return [_tensor(_imread(target)) for target in targets]
