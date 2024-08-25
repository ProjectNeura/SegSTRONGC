from os import listdir
from os.path import isdir
from numpy import ndarray
from augmentation import TransformBase, Smoke
from cv2 import imread, imwrite


Item: type = tuple[str, ndarray]


def augment_item(src: str, transform: TransformBase, branch: str = "") -> Item | list[Item]:
    if not isdir(target := f"{src}/{branch}"):
        return f"{branch}", transform(imread(src))
    r = []
    for f in listdir(target):
        if not f.endswith(".png"):
            continue
        if isinstance(item := augment_item(src, transform, f"{branch}/{f}"), list):
            r += item
        else:
            r.append(item)
    return r


def augment_with_structure(src: str, output_dir: str, transform: TransformBase) -> None:
    for path, img in augment_item(src, transform):
        imwrite(f"{output_dir}/{path}", img)


if __name__ == '__main__':
    augment_with_structure("C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val",
                           "C:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke", Smoke())
