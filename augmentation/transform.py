from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

from typing_extensions import override as _override, Literal as _Literal

from augmentation.computational import ndarray as _ndarray, zeros as _zeros, linspace as _linspace, full as _full, \
    concatenate as _concatenate, ones as _ones, repeat as _repeat, array as _array, expand_dims as _expand_dims


class TransformBase(object, metaclass=_ABCMeta):
    @_abstractmethod
    def apply(self, img: _ndarray) -> _ndarray:
        raise NotImplementedError

    def __call__(self, img: _ndarray) -> _ndarray:
        return self.apply(img)


class Smoke(TransformBase):
    def __init__(self, attenuation_factor: float = .2, mode: _Literal["linear", "quadratic"] = "linear",
                 smoke_color: tuple[int, int, int] = (255, 255, 255), maximum: float = 1, step: int = 2) -> None:
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
        return img * (_ones((height, width, 3)) - smoke_mask) + smoke_mask * self._smoke_color
