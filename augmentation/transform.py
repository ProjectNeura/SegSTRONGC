from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from os.path import abspath as _abspath
from random import randint as _randint

from albumentations import Compose as _Compose, SafeRotate as _SafeRotate, RandomCrop as _RandomCrop, \
    Resize as _Resize, RandomBrightnessContrast as _RandomBrightnessContrast
from cv2 import imread as _imread
from typing_extensions import override as _override, Literal as _Literal

from augmentation.computational import ndarray as _ndarray, zeros as _zeros, linspace as _linspace, full as _full, \
    concatenate as _concatenate, ones as _ones, repeat as _repeat, expand_dims as _expand_dims, \
    array as _array

_ASSETS_PATH: str = f"{_abspath(__file__)[:-12]}assets"
_SMOKE_TEXTURE: _ndarray = _imread(f"{_ASSETS_PATH}/smoke.jpg")


class TransformBase(object, metaclass=_ABCMeta):
    @_abstractmethod
    def apply(self, img: _ndarray) -> _ndarray:
        raise NotImplementedError

    def __call__(self, img: _ndarray) -> _ndarray:
        return self.apply(img)


class Compose(TransformBase):
    def __init__(self, *transforms: TransformBase) -> None:
        self._transforms: tuple[TransformBase, ...] = transforms

    @_override
    def apply(self, img: _ndarray) -> _ndarray:
        for transform in self._transforms:
            img = transform(img)
        return img


class Smoke(TransformBase):
    def __init__(self, attenuation_factor: float = .2, mode: _Literal["linear", "quadratic"] = "linear",
                 smoke_color: tuple[int, int, int] = (200, 200, 200), maximum: float = .7, step: int = 1) -> None:
        self._attenuation_factor: float = attenuation_factor
        self._mode: _Literal["linear", "quadratic"] = mode
        self._smoke_color: _ndarray = _array(smoke_color)
        self._maximum: float = maximum
        self._step: int = step

    @_override
    def apply(self, img: _ndarray) -> _ndarray:
        height, width, num_channels = img.shape
        if num_channels != 3:
            raise AttributeError("Channel error")
        r, h = width * .5, int(height * .5)
        tex = _array(_Compose([
            _RandomCrop(_randint(int(height * .4), int(height * .9)), _randint(int(width * .4), int(width * .9))),
            _SafeRotate(p=1), _Resize(height, width)])(image=_SMOKE_TEXTURE)["image"])
        smoke_mask = _zeros((height, width))
        for i in range(0, h, self._step):
            d = h - i
            total_length = round(2 * (r ** 2 - d ** 2) ** .5)
            total_length += total_length % 2
            num_steps = round(total_length * self._attenuation_factor)
            num_zeros = int((width - total_length) * .5)
            row = _concatenate((_zeros(num_zeros), _linspace(0, self._maximum, num_steps),
                                _full(total_length - num_steps * 2, self._maximum),
                                _linspace(self._maximum, 0, num_steps), _zeros(num_zeros)))
            if self._mode == "quadratic":
                row **= 2
            smoke_mask[i] = row
        smoke_mask = _repeat(_expand_dims(smoke_mask, -1), 3, 2)
        smoke_mask += smoke_mask[::-1, :, :]
        smoke_mask += smoke_mask * tex / 255
        result = _array(img) * (_ones((height, width, 3)) - smoke_mask) + smoke_mask * self._smoke_color
        return result if isinstance(result, _ndarray) else result.get()


class LowBrightness(TransformBase):
    def __init__(self, brightness_range: tuple[float, float] = (-.9, -.1),
                 contrast_range: tuple[float, float] = (-0.2, 0.2)) -> None:
        self._transform: _RandomBrightnessContrast = _RandomBrightnessContrast(brightness_range, contrast_range, p=1)

    @_override
    def apply(self, img: _ndarray) -> _ndarray:
        return self._transform(image=img)["image"]
