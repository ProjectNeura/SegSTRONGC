from os import listdir
from os.path import isdir


def get_items(src: str, branch: str = "") -> str | list[str]:
    if not isdir(target := f"{src}/{branch}"):
        return f"{branch}"
    r = []
    for f in listdir(target):
        if isinstance(item := get_items(src, f"{branch}/{f}"), str):
            r.append(item)
        else:
            r += item
    return r
