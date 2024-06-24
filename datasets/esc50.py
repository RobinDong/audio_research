import torch
import librosa
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchaudio import transforms as audio_tran
from audiomentations import Compose, AddGaussianNoise, Shift, TimeStretch, PitchShift

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
        self.trans = torch.nn.Sequential(
            audio_tran.MFCC(
                sample_rate=TARGET_SR,
                n_mfcc=MFCC,
                melkwargs={
                    "n_fft": 2048,
                    "hop_length": 512,
                    "n_mels": NR_MELS,
                    "f_min": FMIN,
                    "f_max": FMAX,
                },
            ),
        )
        if not validation:
            self.augment = Compose(
                [
                    AddGaussianNoise(
                        min_amplitude=config.dataset.gaussian_min_amp,
                        max_amplitude=config.dataset.gaussian_max_amp,
                        p=0.2,
                    ),
                    TimeStretch(
                        min_rate=config.dataset.ts_min_rate,
                        max_rate=config.dataset.ts_max_rate,
                        p=0.2,
                    ),
                    PitchShift(
                        min_semitones=config.dataset.ps_min_semi,
                        max_semitones=config.dataset.ps_max_semi,
                        p=0.2,
                    ),
                    Shift(
                        min_shift=config.dataset.shift_min,
                        max_shift=config.dataset.shift_max,
                        p=0.2,
                    ),
                ]
            )

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
        if not self.validation:
            sound = self.augment(samples=sound, sample_rate=TARGET_SR)
        sound = torch.from_numpy(sound)

        sound = self.trans(sound)

        len_sound = sound.shape[-1]
        assert len_sound >= FRAMES_WIN, sound.shape

        if self.validation:
            start = 0
        else:
            start = np.random.randint(len_sound - FRAMES_WIN)

        # sound = sound[:, start : start + FRAMES_WIN]
        # sound = sound.astype(np.float32)

        return sound, self.category_map[Path(filename).stem]


if __name__ == "__main__":
    ds = ESC50Dataset(
        [
            "/data/audio/ESC-50-master/audio/1-100032-A-0.wav",
            "/data/audio/ESC-50-master/audio/2-126433-A-17.wav",
            "/data/audio/ESC-50-master/audio/5-253094-D-49.wav",
        ],
        "/data/audio/ESC-50-master/meta/",
    )
    sound, label = ds[2]
    print(sound.shape, label)
    sound, label = ds[0]
    print(sound.shape, label)
