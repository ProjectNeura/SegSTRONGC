from shutil import copyfile

from utils import get_items


def rearrange(src: str, output_dir: str) -> None:
    i = 0
    for path in get_items(src):
        if not path.endswith(".png"):
            continue
        copyfile(f"{src}/{path}", f"{output_dir}/case_{str(i).zfill(4)}_0000.png")
        i += 1


if __name__ == '__main__':
    rearrange("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke",
              "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/smoke_nnunet")
    rearrange("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/lb",
              "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/lb_nnunet")
    rearrange("S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/blood",
              "S:/SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/blood_nnunet")
