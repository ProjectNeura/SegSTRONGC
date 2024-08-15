try:
    import cupy as _np
except ImportError:
    import numpy as _np

ndarray: type[_np.ndarray] = _np.ndarray
array: type[_np.array] = _np.array
zeros: type[_np.zeros] = _np.zeros
ones: type[_np.ones] = _np.ones
linspace: type[_np.linspace] = _np.linspace
full: type[_np.full] = _np.full
concatenate: type[_np.concatenate] = _np.concatenate
dot: type[_np.dot] = _np.dot
repeat: type[_np.repeat] = _np.repeat
expand_dims: type[_np.expand_dims] = _np.expand_dims
