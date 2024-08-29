from cv2 import imwrite
from numpy import load, uint8

from utils import get_items


def npy2png(src: str, output_dir: str) -> None:
    for path in get_items(src):
        if not path.endswith('.npy'):
            continue
        imwrite(f"{output_dir}/{path.replace('.npy', '.png')}", load(f"{src}/{path}").astype(uint8))


if __name__ == '__main__':
    npy2png("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke/1/0",
            "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke/1/png")
