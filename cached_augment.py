from os import makedirs

from cv2 import imread, imwrite

from augmentation import TransformBase, Smoke, LowBrightness, Blood
from utils import get_items


def augment_with_structure(src: str, output_dir: str, transform: TransformBase) -> None:
    makedirs(output_dir, exist_ok=True)
    for path in get_items(src):
        if not path.endswith(".png"):
            continue
        makedirs(f"{output_dir}/{path[:path.rfind('/')]}", exist_ok=True)
        imwrite(f"{output_dir}/{path}", transform(imread(f"{src}/{path}")))


if __name__ == '__main__':
    augment_with_structure("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke", Smoke())
    augment_with_structure("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/lb", LowBrightness())
    augment_with_structure("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/blood", Blood(20))
