from numpy import ndarray, logical_and


def calculate_dcs(a: ndarray, b: ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float(2 * logical_and(a, b).sum() / (a.sum() + b.sum()))


def calculate_nsd(a: ndarray, b: ndarray) -> float:
    return abs(a - b).sum() / max(a.sum(), b.sum())
