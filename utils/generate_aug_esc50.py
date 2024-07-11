import os
import sys
import glob
import json
import torch
import librosa
import functools
import numpy as np
import multiprocessing
import audiomentations

from pathlib import Path
from collections import defaultdict

from torchaudio import transforms as audio_tran
from audiomentations import (
    Compose,
    Shift,
    PolarityInversion,
    BitCrush,
)

import warnings

warnings.filterwarnings("ignore")

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
    trans = audio_tran.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=MFCC,
        log_mels=True,
        melkwargs={
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": NR_MELS,
            "f_min": FMIN,
            "f_max": FMAX,
        },
    ).cuda()

    if not validation:
        augs = [Shift(), PolarityInversion(), BitCrush()]
        ops = defaultdict(dict)
        for key, value in config.items():
            op_name, attr_name = key.split(".")
            ops[op_name][attr_name] = value[0]

        for op_name, attrs in ops.items():
            attrs_string = ", ".join([f"{key}={val}" for key, val in attrs.items()])
            aug = eval(f"audiomentations.{op_name}({attrs_string})")
            augs.append(aug)
        augment = Compose(augs)

    # create directories
    for index in range(1, 6):
        os.makedirs(f"{output_dir}/{index}", exist_ok=True)

    for filename in file_lst:
        data, sr = librosa.load(filename)
        sound = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        if repeat_index > 0 or not validation:
            sound = augment(samples=sound, sample_rate=TARGET_SR)

        sound = torch.from_numpy(sound.copy()).cuda()
        sound = trans(sound).cpu()

        len_sound = sound.shape[-1]
        assert len_sound >= FRAMES_WIN, sound.shape

        stem = Path(filename).stem
        prefix = stem.split("-")[0]
        new_name = stem + "_" + str(repeat_index) + ".npy"
        with open(f"{output_dir}/{prefix}/{new_name}", "wb") as fp:
            np.save(fp, sound)


if __name__ == "__main__":
    with open(f"config_{sys.argv[1]}.json", "r") as fp:
        config = json.load(fp)

    wave_lst = glob.glob("/data/audio/ESC-50-master/audio/*.wav")
    pool = multiprocessing.Pool(8)
    pool.map(
        functools.partial(
            generate_aug,
            config=config,
            file_lst=wave_lst,
            meta_dir="/data/audio/ESC-50-master/meta",
            output_dir=f"/data/audio/aug_gen/ns{sys.argv[1]}",
        ),
        range(48),
    )
