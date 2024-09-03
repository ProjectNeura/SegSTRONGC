from typing import Callable

from cv2 import imread, cvtColor, COLOR_RGB2GRAY
from medpy.metric.binary import dc
from numpy import ndarray, load, abs as npabs, max as npmax, sum as npsum
from rich.progress import Progress

from utils import get_items


def calculate_dcs(a: ndarray, b: ndarray) -> float:
    return dc((a / npmax(a)) == 1, (b / npmax(b)) == 1)


def calculate_nsd(a: ndarray, b: ndarray) -> float:
    a, b = (a / npmax(a)).astype(int), (b / npmax(b)).astype(int)
    return npsum(npabs(a - b)) / max(npsum(a), npsum(b))


def evaluate(src: str, val: str, method: Callable[[ndarray, ndarray], float]) -> float:
    i = 0
    r = 0
    items = get_items(val)
    with Progress() as progress:
        task = progress.add_task("[white]Evaluating...", total=int(len(items) / 3))
        for path in get_items(val):
            if not path.endswith(".npy"):
                continue
            r += method(load(f"{val}/{path}"), cvtColor(imread(f"{src}/case_{str(i).zfill(4)}.png"), COLOR_RGB2GRAY))
            i += 1
            progress.update(task, advance=1)
    return r / i


if __name__ == '__main__':
    predicted = "smoke"
    print("DCS:", evaluate(f"S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/{predicted}_predicted",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val", calculate_dcs))
    print("NSD:", evaluate(f"S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/{predicted}_predicted",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val", calculate_nsd))
