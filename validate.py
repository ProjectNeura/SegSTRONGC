from typing import Callable

from cv2 import imread, cvtColor, COLOR_RGB2GRAY
from numpy import ndarray, logical_and, load
from rich.progress import Progress

from utils import get_items


def calculate_dcs(a: ndarray, b: ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * logical_and(a, b).sum() / (a.sum() + b.sum()))


def calculate_nsd(a: ndarray, b: ndarray) -> float:
    return abs(a - b).sum() / max(a.sum(), b.sum())


def evaluate(src: str, val: str, method: Callable[[ndarray, ndarray], float]) -> float:
    i = 0
    r = 0
    items = get_items(val)
    with Progress() as progress:
        task = progress.add_task("[white]Evaluating...", total=int(len(items) / 3))
        for path in get_items(val):
            if not path.endswith(".npy"):
                continue
            r += method(load(f"{val}/{path}"),
                        cvtColor(imread(f"{src}/case_{str(i).zfill(4)}.png"), COLOR_RGB2GRAY) / 256)
            i += 1
            progress.update(task, advance=1)
    return r / i


if __name__ == '__main__':
    predicted = "lb"
    print("DCS:", evaluate(f"S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/{predicted}_predicted",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val", calculate_dcs))
    print("NSD:", evaluate(f"S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/{predicted}_predicted",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val", calculate_nsd))
