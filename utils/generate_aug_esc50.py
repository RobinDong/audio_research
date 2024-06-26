import os
import glob
import json
import torch
import librosa
import functools
import numpy as np
import multiprocessing

from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
from torchaudio import transforms as audio_tran
from audiomentations import Compose, AddGaussianNoise, Shift, TimeStretch, PitchShift

MFCC = 64
NR_MELS = 128
FMIN = 20
FMAX = 20000
TARGET_SR = 32000

FRAMES_PER_SEC = 62.6
FRAMES_WIN = int(FRAMES_PER_SEC * 4)  # 4 seconds


def generate_aug(
    repeat_index: int,
    config,
    file_lst: str,
    meta_dir: str,
    output_dir: str,
    validation: bool = False,
):
    trans = torch.nn.Sequential(
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
        augment = Compose(
            [
                AddGaussianNoise(
                    min_amplitude=config.dataset.gaussian_min_amp,
                    max_amplitude=config.dataset.gaussian_max_amp,
                    p=1.0,
                ),
                TimeStretch(
                    min_rate=config.dataset.ts_min_rate,
                    max_rate=config.dataset.ts_max_rate,
                    p=1.0,
                ),
                PitchShift(
                    min_semitones=config.dataset.ps_min_semi,
                    max_semitones=config.dataset.ps_max_semi,
                    p=1.0,
                ),
                Shift(
                    min_shift=config.dataset.shift_min,
                    max_shift=config.dataset.shift_max,
                    p=1.0,
                ),
            ]
        )

    # create directories
    for index in range(1, 6):
        os.makedirs(f"{output_dir}/{index}", exist_ok=True)

    for filename in file_lst:
        data, sr = librosa.load(filename)
        sound = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        if repeat_index > 0 or not validation:
            sound = augment(samples=sound, sample_rate=TARGET_SR)
        sound = torch.from_numpy(sound)

        sound = trans(sound)

        len_sound = sound.shape[-1]
        assert len_sound >= FRAMES_WIN, sound.shape

        stem = Path(filename).stem
        prefix = stem.split("-")[0]
        new_name = stem + "_" + str(repeat_index) + ".npy"
        with open(f"{output_dir}/{prefix}/{new_name}", "wb") as fp:
            np.save(fp, sound)


if __name__ == "__main__":
    config = namedtuple("config", ["dataset"])
    with open("config.json", "r") as fp:
        config.dataset = json.load(fp, object_hook=lambda node: SimpleNamespace(**node))

    wave_lst = glob.glob("/data/audio/ESC-50-master/audio/*.wav")
    pool = multiprocessing.Pool(4)
    pool.map(
        functools.partial(
            generate_aug,
            config=config,
            file_lst=wave_lst,
            meta_dir="/data/audio/ESC-50-master/meta",
            output_dir="/data/audio/aug_gen",
        ),
        range(60),
    )
