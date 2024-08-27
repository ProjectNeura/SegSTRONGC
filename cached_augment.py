from os import listdir, makedirs
from os.path import isdir

from cv2 import imread, imwrite
from numpy import ndarray

from augmentation import TransformBase, Smoke, LowBrightness, Blood

Item: type = tuple[str, ndarray]


def augment_item(src: str, transform: TransformBase, branch: str = "") -> Item | list[Item]:
    if not isdir(target := f"{src}/{branch}"):
        if not target.endswith(".png"):
            return []
        return f"{branch}", transform(imread(target))
    r = []
    for f in listdir(target):
        if isinstance(item := augment_item(src, transform, f"{branch}/{f}"), list):
            r += item
        else:
            r.append(item)
    return r


def augment_with_structure(src: str, output_dir: str, transform: TransformBase) -> None:
    makedirs(output_dir, exist_ok=True)
    for path, img in augment_item(src, transform):
        makedirs(f"{output_dir}/{path[:path.rfind('/')]}", exist_ok=True)
        imwrite(f"{output_dir}/{path}", img)


if __name__ == '__main__':
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke", Smoke())
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/lb", LowBrightness())
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/blood", Blood(20))
