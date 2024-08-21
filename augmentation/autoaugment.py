from torch import tensor as _tensor
from torchvision.transforms.functional import rotate as _rotate, adjust_brightness as _adjust_brightness, \
    adjust_saturation as _adjust_saturation, adjust_contrast as _adjust_contrast, \
    adjust_sharpness as _adjust_sharpness, posterize as _posterize, solarize as _solarize, \
    autocontrast as _autocontrast, equalize as _equalize, invert as _invert

from augmentation.computational import npndarray as _npndarray, nparray as _nparray
from augmentation.transform import TransformBase


class Rotate(TransformBase):
    def __init__(self, angle: float, p: float = 1) -> None:
        super().__init__(p)
        self._angle: float = angle

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_rotate(_tensor(img), self._angle))


class Brightness(TransformBase):
    def __init__(self, brightness_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._brightness_factor: float = brightness_factor

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_adjust_brightness(_tensor(img), 1 + self._brightness_factor))


class Color(TransformBase):
    def __init__(self, saturation_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._saturation_factor: float = saturation_factor

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_adjust_saturation(_tensor(img), 1 + self._saturation_factor))


class Contrast(TransformBase):
    def __init__(self, contrast_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._contrast_factor: float = contrast_factor

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_adjust_contrast(_tensor(img), 1 + self._contrast_factor))


class Sharpness(TransformBase):
    def __init__(self, sharpness_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._sharpness_factor: float = sharpness_factor

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_adjust_sharpness(_tensor(img), 1 + self._sharpness_factor))


class Posterize(TransformBase):
    def __init__(self, bits: int, p: float = 1) -> None:
        super().__init__(p)
        self._bits: int = bits

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_posterize(_tensor(img), self._bits))


class Solarize(TransformBase):
    def __init__(self, threshold: float, p: float = 1) -> None:
        super().__init__(p)
        self._threshold: float = threshold

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_solarize(_tensor(img), self._threshold))


class AutoContrast(TransformBase):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_autocontrast(_tensor(img)))


class Equalize(TransformBase):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_equalize(_tensor(img)))


class Invert(TransformBase):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(_invert(_tensor(img)))
