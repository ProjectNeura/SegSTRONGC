from os import makedirs
from os.path import exists


def check_or_create(path: str) -> None:
    if not exists(path):
        makedirs(path)


def transfer(origin: str, destination: str) -> None:
    pass
