from os import makedirs, listdir
from os.path import exists
from shutil import copyfile

from cv2 import imwrite
from numpy import load, uint8
from rich.progress import Progress


def check_or_create(path: str) -> None:
    if not exists(path):
        makedirs(path)


def transfer(src: str, destination: str) -> None:
    destination = f"{destination}/nnUNet_raw/Dataset001_SegSTRONG_C"
    check_or_create(destination)
    with open(f"{destination}/dataset.json", "w") as f:
        f.write("{\"channel_names\": {\"0\": \"red\", \"1\": \"green\", \"2\": \"blue\"}, "
                "{\"labels\": {\"background\": 0, \"tool\": 1}, \"numTraining\": 6600, \"file_ending\": \".png\"")
    check_or_create(f"{destination}/imagesTr")
    check_or_create(f"{destination}/labelsTr")
    with Progress() as progress:
        task = progress.add_task("[white]Transferring...", total=3300)
        serial = 0
        src = f"{src}/train"
        for i in listdir(src):
            for j in listdir(f"{src}/{i}"):
                for n in range(300):
                    copyfile(f"{src}/{i}/{j}/regular/left/{n}.png",
                             f"{destination}/imagesTr/{(s := str(serial).zfill(3))}_0000.png")
                    imwrite(f"{destination}/labelsTr/{s}.png",
                            load(f"{src}/{i}/{j}/ground_truth/left/{n}.npy").astype(uint8))
                    serial += 1
                    copyfile(f"{src}/{i}/{j}/regular/right/{n}.png",
                             f"{destination}/imagesTr/{(s := str(serial).zfill(3))}_0000.png")
                    imwrite(f"{destination}/labelsTr/{s}.png",
                            load(f"{src}/{i}/{j}/ground_truth/right/{n}.npy").astype(uint8))
                    serial += 1
                    progress.update(task, advance=1)


if __name__ == '__main__':
    transfer("F:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release",
             "F:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_nnunet")
