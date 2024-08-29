from os import listdir, makedirs
from os.path import isdir

from cv2 import imread, imwrite

from augmentation import TransformBase, Smoke, LowBrightness, Blood


def get_items(src: str, branch: str = "") -> str | list[str]:
    if not isdir(target := f"{src}/{branch}"):
        return f"{branch}" if target.endswith(".png") or target.endswith(".npy") else []
    r = []
    for f in listdir(target):
        if isinstance(item := get_items(src, f"{branch}/{f}"), str):
            r.append(item)
        else:
            r += item
    return r


def augment_with_structure(src: str, output_dir: str, transform: TransformBase) -> None:
    makedirs(output_dir, exist_ok=True)
    for path in get_items(src):
        makedirs(f"{output_dir}/{path[:path.rfind('/')]}", exist_ok=True)
        imwrite(f"{output_dir}/{path}", transform(imread(f"{src}/{path}")))


if __name__ == '__main__':
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke", Smoke())
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/lb", LowBrightness())
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/blood", Blood(20))
