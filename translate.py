from os import listdir

from cv2 import VideoWriter, VideoWriter_fourcc, imread
from numpy import concatenate, load
from rich.progress import Progress

src: str = "F:\\SharedDatasets/SegSTRONGC_release/SegSTRONGC_release/val"
# origin: str = "regular"
# total: int = 3300
origin: str = "bg_change"
total: int = 900

if __name__ == '__main__':
    writer = VideoWriter("out.mp4", VideoWriter_fourcc(*"mp4v"), 30, (1920 * 2, 1080))
    with Progress() as progress:
        task = progress.add_task("[white]Translating...", total=total)
        for i in listdir(src):
            for j in listdir(f"{src}/{i}"):
                for n in range(300):
                    left = imread(f"{src}/{i}/{j}/{origin}/left/{n}.png")
                    left_mask = load(f"{src}/{i}/{j}/ground_truth/left/{n}.npy")
                    left[left_mask == 1] = [255, 0, 0]
                    right = imread(f"{src}/{i}/{j}/{origin}/right/{n}.png")
                    right_mask = load(f"{src}/{i}/{j}/ground_truth/right/{n}.npy")
                    right[right_mask == 1] = [255, 0, 0]
                    image = concatenate((left, right), axis=1)
                    writer.write(image)
                    progress.update(task, advance=1)
    writer.release()
