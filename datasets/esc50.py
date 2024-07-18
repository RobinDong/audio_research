import librosa
import numpy as np
import audiomentations

from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from audiomentations import (
    Compose,
    Shift,
    PolarityInversion,
    BitCrush,
)

TARGET_SR = 32000
AUG_P = 0.5
PERIOD_RATIO = 0.6


class ESC50Dataset(Dataset):
    def __init__(self, config, wave_list: str, meta_dir: str, validation: bool = False):
        self.file_lst = wave_list
        self.validation = validation
        self.category_map = self.load_meta(meta_dir)

        if not validation:
            augs = [Shift(p=AUG_P), PolarityInversion(p=AUG_P), BitCrush(p=AUG_P)]
            ops = defaultdict(dict)
            for key, value in config.items():
                op_name, attr_name = key.split(".")
                ops[op_name][attr_name] = value[0]

            for op_name, attrs in ops.items():
                attrs_string = ", ".join([f"{key}={val}" for key, val in attrs.items()])
                attrs_string += f", p={AUG_P}"
                aug = eval(f"audiomentations.{op_name}({attrs_string})")
                augs.append(aug)
            self.augment = Compose(augs)

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

        data, sr = librosa.load(filename)
        sound = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

        len_sound = sound.shape[0]
        FRAMES_WIN = int(len_sound * PERIOD_RATIO)

        if self.validation:
            start = 0
        else:
            start = np.random.randint(len_sound - FRAMES_WIN)

        sound = sound[start: start + FRAMES_WIN]

        stem = Path(filename).stem.split("_")[0]
        if self.validation:
            return sound, sound, sound, self.category_map[stem]
        else:
            return (
                sound,
                self.augment(samples=sound, sample_rate=TARGET_SR),
                self.augment(samples=sound, sample_rate=TARGET_SR),
                self.category_map[stem],
            )


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
