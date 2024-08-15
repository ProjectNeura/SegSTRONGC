import numpy as _true_np

try:
    import cupy as _np
except ImportError:
    import numpy as _np

ndarray: type[_true_np.ndarray] = _true_np.ndarray
array: type[_np.array] = _np.array
nparray: type[_true_np.array] = _true_np.array
zeros: type[_np.zeros] = _np.zeros
ones: type[_np.ones] = _np.ones
linspace: type[_np.linspace] = _np.linspace
full: type[_np.full] = _np.full
concatenate: type[_np.concatenate] = _np.concatenate
dot: type[_np.dot] = _np.dot
repeat: type[_np.repeat] = _np.repeat
expand_dims: type[_np.expand_dims] = _np.expand_dims
