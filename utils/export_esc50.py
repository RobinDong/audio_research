import os
import fire
import glob
import librosa
import numpy as np

from pathlib import Path
from tqdm import tqdm

NR_MELS = 128
FMIN = 20
FMAX = 20000
QUANT_RANGE = 4  # range (-4, 4)
QUANT_GRANU = 256  # 8bit
TARGET_SR = 32000


class ESC50Exporter:
    def export(self, esc50_dir: str, npy_dir: str):
        file_lst = glob.glob(f"{esc50_dir}/audio/*.wav")
        if not os.path.exists(npy_dir):
            os.mkdir(npy_dir)
        for filename in tqdm(file_lst):
            data, sr = librosa.load(filename)
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            # We use 20Hz to 20kHz as the range of human hearing
            spectro = librosa.feature.melspectrogram(
                y=data, sr=sr, n_mels=NR_MELS, fmin=FMIN, fmax=FMAX
            )
            s_db = librosa.power_to_db(spectro, ref=np.max)

            means = s_db.mean()
            stds = s_db.std()
            normalized = (s_db - means) / stds
            # range (-4, 4)
            normalized = (
                (normalized + QUANT_RANGE) * QUANT_GRANU * 2 / QUANT_RANGE
            ).astype(np.uint8)

            new_name = Path(filename).stem + ".npy"
            with open(f"{npy_dir}/{new_name}", "wb") as fp:
                np.save(fp, normalized)


if __name__ == "__main__":
    exporter = ESC50Exporter()
    fire.Fire(exporter.export)
