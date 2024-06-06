import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

FRAMES_PER_SEC = 62.6
FRAMES_WIN = int(FRAMES_PER_SEC * 2)  # 2 seconds


class ESC50Dataset(Dataset):
    def __init__(self, npy_list: str, meta_dir: str, validation: bool = False):
        self.file_lst = npy_list
        self.validation = validation
        self.category_map = self.load_meta(meta_dir)

    def load_meta(self, meta_dir: str) -> dict[str, int]:
        category_map = {}
        csv_file = f"{meta_dir}/esc50.csv"
        with open(csv_file, "r") as fp:
            lines = fp.readlines()
            for line in lines[1:]:  # skip first line (title)
                arr = line.split(",")
                filename, label = arr[0], int(arr[2])
                category_map[Path(filename).stem] = label
        return category_map

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        filename = self.file_lst[index]
        try:
            sound = np.load(filename, mmap_mode="r")
        except Exception as ex:
            print(ex)
            print(f"Failed to load {filename}")

        len_sound = len(sound)
        assert len_sound >= FRAMES_WIN

        if self.validation:
            start = 0
        else:
            start = np.random.randint(len_sound - FRAMES_WIN)

        sound = sound[:, start : start + FRAMES_WIN].astype(np.float16)
        return sound, self.category_map[Path(filename).stem]


if __name__ == "__main__":
    ds = ESC50Dataset(
        [
            "/Users/robin/Downloads/ESC-50-master/npy/1-100032-A-0.npy",
            "/Users/robin/Downloads/ESC-50-master/npy/2-126433-A-17.npy",
            "/Users/robin/Downloads/ESC-50-master/npy/5-253094-D-49.npy",
        ],
        "/Users/robin/Downloads/ESC-50-master/meta/",
    )
    sound, label = ds[2]
    print(sound.shape, label)
    sound, label = ds[0]
    print(sound.shape, label)
