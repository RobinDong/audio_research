import os
import sys
import copy
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
from audiomentations import (
    Compose,
    AddColorNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AirAbsorption,
    Aliasing,
    BandPassFilter,
    BandStopFilter,
    BitCrush,
    ClippingDistortion,
    HighPassFilter,
    HighShelfFilter,
    Limiter,
    LowPassFilter,
    LowShelfFilter,
    Mp3Compression,
    PeakingFilter,
    PitchShift,
    PolarityInversion,
    RepeatPart,
    Reverse,
    RoomSimulator,
    SevenBandParametricEQ,
    Shift,
    SpecChannelShuffle,
    SpecFrequencyMask,
    TanhDistortion,
    TimeMask,
    TimeStretch,
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
        augs = [
            AddColorNoise(min_snr_db=10, max_snr_db=20, min_f_decay=-1.0, max_f_decay=1.0),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01),
            AddGaussianSNR(min_snr_db=10, max_snr_db=20),
            AirAbsorption(min_humidity=50, max_humidity=70, min_distance=20, max_distance=50),
            BitCrush(min_bit_depth=12, max_bit_depth=20),
            ClippingDistortion(min_percentile_threshold=20, max_percentile_threshold=30),
            Limiter(min_threshold_db=-8.0, max_threshold_db=-4.0),
            Mp3Compression(min_bitrate=16, max_bitrate=32),
            PeakingFilter(min_center_freq=500.0, max_center_freq=750.0, min_gain_db=-6.0, max_gain_db=6.0, min_q=1.0, max_q=2.5),
            PitchShift(min_semitones=-1.0, max_semitones=1.0),
            PolarityInversion(),
            # RepeatPart,
            Reverse(),
            RoomSimulator(),
            SevenBandParametricEQ(min_gain_db=-3.0, max_gain_db=3.0),
            Shift(min_shift=-0.2, max_shift=0.2),
            # SpecChannelShuffle,
            # SpecFrequencyMask,
            TanhDistortion(min_distortion=0.4, max_distortion=0.6),
            TimeMask(min_band_part=0.4, max_band_part=0.6),
            TimeStretch(),
        ]
        sample_ops = np.random.choice(augs, config.dataset.N)
        arr = [copy.copy(op) for _ in range(config.dataset.M) for op in sample_ops]
        augment = Compose(arr)

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
    config = namedtuple("config", ["dataset"])
    with open("config.json", "r") as fp:
        config.dataset = json.load(fp, object_hook=lambda node: SimpleNamespace(**node))

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
        range(20),
    )
