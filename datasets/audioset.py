import io
import glob
import torch
import pickle
import torchaudio
import audiomentations  # noqa: F401

import numpy as np

from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from audiomentations import (
    Compose,
    Shift,
    PolarityInversion,
    BitCrush,
)

RAW_LENGTH = 480000
TARGET_LENGTH = 998
AUDIO_RATIO = 0.8
AUDIO_WIN = int(TARGET_LENGTH * AUDIO_RATIO)
AUG_P = 0.5
NUM_CLASSES = 527
MIXUP_ALPHA = 10.0


class AudioSetDataset(Dataset):
    def __init__(self, config, data_dir: str, meta_file: str, validation: bool = False):
        self.parent_data_dir = Path(data_dir).parent

        idx_lst = glob.glob(f"{data_dir}")
        self.file_lst = []
        for idx_filename in idx_lst:
            with open(idx_filename, "rb") as fp:
                obj = pickle.load(fp)
                self.file_lst += obj

        print("file_lst:", len(self.file_lst))

        self.validation = validation
        self.name_to_labels = self.load_meta(meta_file)

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

    def load_meta(self, meta_file: str) -> dict[str, int]:
        name_to_labels = {}
        label_map = {}
        label_start = 0
        with open(meta_file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if line[0] == "#":
                    continue
                filename = line.split(",")[0]
                labels = line.split('"')[1].split(",")
                label_ids = set()
                for label in labels:
                    if label not in label_map:
                        label_map[label] = label_start
                        label_start += 1
                    label_ids.add(label_map[label])
                name_to_labels[filename] = tuple(label_ids)

        return name_to_labels

    def __len__(self):
        return len(self.file_lst)

    def index_to_sound(self, index):
        flac_filename, tar_filename, start, length = self.file_lst[index]

        with open(f"{tar_filename}", "rb") as tar_fp:
            tar_fp.seek(start)
            binary = tar_fp.read(length)
            try:
                sound, sr = torchaudio.load(io.BytesIO(binary))
                sound = sound - sound.mean()
            except Exception:
                return None, None

        if sound.size(0) >= 2:
            sound = sound.mean(dim=0)
        elif sound.size(0) == 1:
            sound = sound[0]

        diff = RAW_LENGTH - sound.size(0)
        if diff > 0:
            sound = torch.nn.ZeroPad1d((0, diff))(sound)
        elif diff < 0:
            sound = sound[:RAW_LENGTH]
        return sound.unsqueeze(0), sr

    def sound_to_fbank(self, sound, sr):
        fbank = torchaudio.compliance.kaldi.fbank(
            sound,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.shape[0]
        """diff = TARGET_LENGTH - n_frames
        if diff > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, diff))(fbank)
        elif diff < 0:
            fbank = fbank[:TARGET_LENGTH, :]"""

        diff = AUDIO_WIN - n_frames
        if diff > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, diff))(fbank)
        elif diff < 0:
            if self.validation:
                start = 0
            else:
                start = np.random.randint(0, -diff)
            fbank = fbank[start : start + AUDIO_WIN, :]
        return fbank

    def index_to_label(self, index):
        flac_filename, _, _, _ = self.file_lst[index]

        label_indices = np.zeros(NUM_CLASSES)
        label_ids = self.name_to_labels[flac_filename]
        for label_id in label_ids:
            label_indices[label_id] = 1.0
        return label_indices

    def __getitem__(self, index):
        sound, sr = self.index_to_sound(index)
        if sound is None:
            return None, None
        label = self.index_to_label(index)
        """if sound is not None and not self.validation:  # mixup
            mixup_idx = np.random.randint(0, len(self.file_lst))
            sound2, sr2 = self.index_to_sound(mixup_idx)
            if sound2 is not None:
                label2 = self.index_to_label(mixup_idx)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                sound = sound * lam + sound2 * (1 - lam)
                label = label * lam + label2 * (1 - lam)"""

        '''sound = sound - sound.mean()
        mean, std = torch.std_mean(sound, dim=1)
        sound = (sound - mean) / std'''
        fbank = self.sound_to_fbank(sound, sr)
        return fbank, label

    @staticmethod
    def my_collate(batch):
        batch = filter(lambda pair: pair[0] is not None, batch)
        return torch.utils.data.dataloader.default_collate(list(batch))


if __name__ == "__main__":
    ds = AudioSetDataset(
        {},
        "/data/audioset",
        "/data/audioset",
    )
    length = len(ds)
    total = 0
    for index in range(length):
        sound, labels = ds[index]
        if sound is not None:
            total += 1
    print("total:", total)
