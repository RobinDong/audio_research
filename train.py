import os
import sys
import json
import glob
import time
import math
import timm
import torchaudio
import contextlib
import multiprocessing
import numpy as np
import torch.nn as nn

from collections import defaultdict, OrderedDict
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision.transforms import v2
from config import TrainConfig
from datasets.esc50 import ESC50Dataset
from supcon_loss import SupConLoss

SEED = 20240605
CKPT_DIR = "out"
LABEL_SMOOTH_RATIO = 0.5

TARGET_SR = 32000


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device_type = "cuda" if torch.cuda.is_available() else "mps"
        self.dtype = torch.float32
        if self.device_type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.ctx = torch.amp.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
            self.ctx = contextlib.suppress()
        self.train_batch_iter = None
        self.train_loader = self.val_loader = None

        self.cutmix = v2.CutMix(num_classes=config.num_classes)
        self.mixup = v2.MixUp(num_classes=config.num_classes)
        self.cm = v2.RandomChoice([self.cutmix, self.mixup])

        self.trans = torchaudio.compliance.kaldi.fbank
        self.supcon_loss = SupConLoss()

    def criterion(self, outputs, targets):
        outputs = self.classifier(outputs)
        one_hot = torch.zeros(
            (targets.shape[0], self.config.num_classes),
            dtype=self.dtype,
            device=self.device_type,
        )
        one_hot.fill_((1 - LABEL_SMOOTH_RATIO) / self.config.num_classes)
        one_hot = one_hot + (targets * LABEL_SMOOTH_RATIO)
        return torch.sum(-one_hot * F.log_softmax(outputs, -1), -1).mean()

    def next_batch(self, loader, batch_iter):
        try:
            data_entry = next(batch_iter)
            if len(data_entry[0]) < self.config.batch_size:
                batch_iter = iter(loader)
                data_entry = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            data_entry = next(batch_iter)
        return data_entry

    def train_step(self, model, optimizer):
        data_entry = self.next_batch(self.train_loader, self.train_batch_iter)
        # sounds, _, _, labels = self.next_batch(self.train_loader, self.train_batch_iter)
        sounds = torch.cat(data_entry[:-1])
        sounds = sounds.to(self.device_type)
        sounds = torch.stack(
            [
                self.trans(
                    sound.unsqueeze(0),
                    htk_compat=True,
                    sample_frequency=TARGET_SR,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=128,
                    dither=0.0,
                    frame_shift=10,
                )
                for sound in sounds
            ]
        )
        sounds = sounds.unsqueeze(-1).permute(0, 3, 1, 2).to(self.dtype)
        labels = torch.cat((data_entry[-1], data_entry[-1], data_entry[-1]))
        labels = labels.to(self.device_type)
        sounds, labels = self.cm(sounds, labels)

        with self.ctx:
            out = model(sounds)
            loss = self.criterion(out, labels)
            # Add SupCon loss
            features = out.view(self.config.batch_size, 3, -1)
            sc_loss = self.supcon_loss(features, data_entry[-1])
            # loss += sc_loss * 0.01

        l2_penalty = sum([(p**2).sum() for p in self.classifier.parameters()])
        # loss = loss + l2_penalty * 0.1

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return out, labels, loss, sc_loss

    def get_lr(self, iteration):
        config = self.config
        # 1) linear warmup for warmup_iters steps
        if iteration < config.warmup_iters:
            return config.lr * iteration / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - config.warmup_iters) / (
            config.lr_decay_iters - config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.lr - config.min_lr)

    def get_accuracy(self, out, target):
        out = self.classifier(out)
        _, predict = torch.max(out, dim=-1)
        correct = predict == torch.max(target, dim=-1)[1]
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    def get_validation_metrics(self, data_entry, model):
        sounds, _, _, labels = data_entry
        sounds = sounds.to(self.device_type)
        sounds = torch.stack(
            [
                self.trans(
                    sound.unsqueeze(0),
                    htk_compat=True,
                    sample_frequency=TARGET_SR,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=128,
                    dither=0.0,
                    frame_shift=10,
                )
                for sound in sounds
            ]
        )
        sounds = sounds.unsqueeze(-1).permute(0, 3, 1, 2).to(self.dtype)
        labels = labels.to(self.device_type)
        labels = F.one_hot(labels, self.config.num_classes)
        # forward
        with self.ctx:
            out = model(sounds)
            loss = self.criterion(out, labels)
        # accuracy
        accuracy = self.get_accuracy(out, labels)
        return OrderedDict(
            [
                ("loss", loss.item()),
                ("accuracy", accuracy),
            ]
        )

    @torch.no_grad()
    def validate(self, cmodel):
        cmodel.eval()

        batch_iter = iter(self.val_loader)
        accumulator = defaultdict(float)
        length = len(self.val_loader)
        for _ in range(length):
            data_entry = next(batch_iter)
            metrics = self.get_validation_metrics(data_entry, cmodel)
            for key, val in metrics.items():
                accumulator[key] += val

        for key, val in accumulator.items():
            accumulator[key] /= length

        cmodel.train()
        return accumulator

    def load_dataset(self):
        file_list = glob.glob(f"{self.config.data_path}/*.wav")
        assert len(file_list) > 0
        if self.config.dataset_name == "ESC-50":
            file_set = set(file_list)
            val_set = set(
                glob.glob(f"{self.config.data_path}/{self.config.eval_prefix}*.wav")
            )
            train_set = file_set - val_set

            with open(f"config_{sys.argv[1]}.json", "r") as fp:
                config = json.load(fp)

            train_ds = ESC50Dataset(config, list(train_set), self.config.meta_dir)
            val_ds = ESC50Dataset(
                config, list(val_set), self.config.meta_dir, validation=True
            )
        else:
            point = int(len(file_list) * self.config.eval_ratio)
            train_lst, val_lst = file_list[point:], file_list[:point]
            np.random.seed(SEED)
            np.random.shuffle(file_list)

        self.train_loader = data.DataLoader(
            train_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        self.train_batch_iter = iter(self.train_loader)

        self.val_loader = data.DataLoader(
            val_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def init(self, resume: str):
        # create/load model
        model = (
            timm.create_model(
                "efficientnet_b0",
                in_chans=1,
                num_classes=self.config.num_classes,
                drop_rate=0.3,
                drop_path_rate=0.3,
                pretrained=True,
            )
            .to(self.dtype)
            .to(self.device_type)
        )
        self.classifier = model.classifier
        model.classifier = Identity()
        if resume:
            checkpoint = torch.load(resume, map_location=self.device_type)
            state_dict = checkpoint["model"]
            # self.config = TrainConfig(**checkpoint["train_config"])
            # iter_start = checkpoint["iteration"] + 1
            iter_start = 1
            self.config.lr = 1e-3
            self.config.min_lr = 1e-4
            self.config.batch_size = 16
            self.config.max_iters = self.config.lr_decay_iters = 4000 + 1
            model.load_state_dict(state_dict)
            print(f"Resume from {iter_start - 1} for training...")
        else:
            iter_start = 1

        self.load_dataset()
        return model, iter_start

    def train(self, resume="", learning_rate=None):
        model, iter_start = self.init(resume)
        if learning_rate:
            self.config.lr = learning_rate
            iter_start = 0
        cmodel = torch.compile(model)
        cmodel = model
        optimizer = torch.optim.SGD(
            cmodel.parameters(),
            lr=self.config.lr,
            momentum=0.9,
            nesterov=False,
        )

        best_accuracy = 0.0
        begin = time.time()

        for iteration in range(iter_start, self.config.max_iters):
            lr = self.get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            out, labels, loss, sc_loss = self.train_step(cmodel, optimizer)

            if iteration % self.config.log_iters == 0 and iteration > 0:
                metrics = OrderedDict(
                    [
                        ("loss", loss.item()),
                        ("sc_loss", sc_loss.item()),
                        ("accuracy", self.get_accuracy(out, labels)),
                    ],
                )
                epoch = iteration // len(self.train_loader)
                now = time.time()
                duration = now - begin
                begin = now
                messages = [f"[{epoch:03d}: {iteration:06d}]"]
                for name, val in metrics.items():
                    messages.append(f"{name}: {val:.3f}")
                messages.append(f"lr: {lr:.3e}")
                messages.append(f"time: {duration:.1f}")
                print(" ".join(messages), flush=True)
            if iteration % self.config.eval_iters == 0 and iteration > 0:
                accumulator = self.validate(cmodel)
                val_accuracy = accumulator["accuracy"]
                best_accuracy = max(best_accuracy, val_accuracy)
                checkpoint = {
                    "model": model.state_dict(),
                    "iteration": iteration,
                    "train_config": asdict(self.config),
                    "eval_accuracy": val_accuracy,
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        CKPT_DIR,
                        f"{iteration}.pt",
                    ),
                )
                messages = ["[Val]"]
                for name, val in accumulator.items():
                    messages.append(f"{name}: {val:.3f}")
                print(" ".join(messages), flush=True)

        print("Best validating accuracy:", best_accuracy)
        return best_accuracy


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    multiprocessing.set_start_method("spawn")

    config = TrainConfig()

    if config.dataset_name == "ESC-50":
        bests = []
        for prefix in ["1", "2", "3", "4", "5"]:
            config.eval_prefix = prefix
            trainer = Trainer(config)
            bests.append(trainer.train())
        print("Avg accuracy:", sum(bests) / len(bests))
