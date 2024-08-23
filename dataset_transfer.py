from os import makedirs, listdir
from shutil import copyfile
from os.path import exists
from rich.progress import Progress
from numpy import load
from cv2 import imwrite


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
        for i in listdir(src):
            for j in listdir(f"{src}/{i}"):
                for n in range(300):
                    copyfile(f"{src}/{i}/{j}/regular/left/{n}.png",
                             f"{destination}/imagesTr/{(serial := str(serial).zfill(3))}_0000.png")
                    imwrite(f"{destination}/labelsTr/{serial}.png", load(f"{src}/{i}/{j}/ground_truth/left/{n}.npy"))
                    serial += 1
                    copyfile(f"{src}/{i}/{j}/regular/right/{n}.png",
                             f"{destination}/imagesTr/{(serial := str(serial).zfill(3))}_0000.png")
                    imwrite(f"{destination}/labelsTr/{serial}.png", load(f"{src}/{i}/{j}/ground_truth/right/{n}.npy"))
                    progress.update(task, advance=1)


if __name__ == '__main__':
    transfer("F:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release",
             "F:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_nnunet")
