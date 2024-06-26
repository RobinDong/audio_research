import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

MFCC = 64
NR_MELS = 128
FMIN = 20
FMAX = 20000
TARGET_SR = 32000

FRAMES_PER_SEC = 62.6
FRAMES_WIN = int(FRAMES_PER_SEC * 4)  # 4 seconds


class ESC50Dataset(Dataset):
    def __init__(self, config, wave_list: str, meta_dir: str, validation: bool = False):
        self.file_lst = wave_list
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
        sound = np.load(filename)

        len_sound = sound.shape[-1]
        assert len_sound >= FRAMES_WIN, sound.shape

        if self.validation:
            start = 0
        else:
            start = np.random.randint(len_sound - FRAMES_WIN)

        sound = sound[:, start: start + FRAMES_WIN]

        stem = Path(filename).stem.split("_")[0]
        return sound, self.category_map[stem]


if __name__ == "__main__":
    ds = ESC50Dataset(
        [
            "/data/audio/aug_gen/2/2-99955-A-7_1.npy",
            "/data/audio/aug_gen/2/2-99955-A-7_3.npy",
            "/data/audio/aug_gen/5/5-233312-A-28_9.npy",
        ],
        "/data/audio/ESC-50-master/meta/",
    )
    sound, label = ds[2]
    print(sound.shape, label)
    sound, label = ds[0]
    print(sound.shape, label)
