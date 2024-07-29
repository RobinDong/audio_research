import os
import sys
import json
import glob
import time
import math
import timm
import torchvision
import contextlib
import multiprocessing
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision.transforms import v2

from config import TrainConfig
from stats import calculate_stats
from supcon_loss import SupConLoss
from datasets.esc50 import ESC50Dataset
from datasets.audioset import AudioSetDataset

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

        self.supcon_loss = SupConLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.resizer = torchvision.transforms.Resize((384, 384))

    def criterion(self, outputs, targets):
        if self.config.dataset_name == "ESC-50":
            one_hot = torch.zeros(
                (targets.shape[0], self.config.num_classes),
                dtype=self.dtype,
                device=self.device_type,
            )
            one_hot.fill_((1 - LABEL_SMOOTH_RATIO) / self.config.num_classes)
            one_hot = one_hot + (targets * LABEL_SMOOTH_RATIO)
            return torch.sum(-one_hot * F.log_softmax(outputs, -1), -1).mean()
        else:
            return self.loss_fn(outputs, targets)

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

    def mixup_data(self, sounds, labels):
        lam = np.random.beta(10, 10)

        batch_size = sounds.size(0)
        index = torch.randperm(batch_size).cuda()

        mixed_sounds = lam * sounds + (1 - lam) * sounds[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
        return mixed_sounds, mixed_labels

    def train_step(self, model, optimizer):
        sounds, labels = self.next_batch(self.train_loader, self.train_batch_iter)
        # shape of 'sounds' for AudioSet [batch_size, 1, 1024, 128]
        sounds = sounds.unsqueeze(1).to(self.device_type).to(self.dtype)
        labels = labels.to(self.device_type)
        if self.config.dataset_name == "ESC-50":
            sounds, labels = self.cm(sounds, labels)
        elif self.config.dataset_name == "AudioSet":
            sounds = self.resizer(sounds)
            # sounds, labels = self.mixup_data(sounds, labels)

        with self.ctx:
            out = model(sounds)
            loss = self.criterion(out, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return out, labels, loss

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
        _, predict = torch.max(out, dim=-1)
        correct = predict == torch.max(target, dim=-1)[1]
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    def get_auc(self, out, target, sigmoid=False):
        if sigmoid:
            out = F.sigmoid(out)
        mAP, AUC = calculate_stats(
            out.cpu().detach().numpy(), target.cpu().detach().numpy()
        )
        return mAP, AUC

    @torch.no_grad()
    def validate(self, cmodel, iteration):
        cmodel.eval()

        accumu_out = []
        accumu_labels = []
        accumu_loss = 0
        for data_entry in self.val_loader:
            sounds, labels = data_entry
            sounds = sounds.unsqueeze(1).to(self.device_type).to(self.dtype)
            labels = labels.to(self.device_type)
            if self.config.dataset_name == "ESC-50":
                labels = F.one_hot(labels, self.config.num_classes)
            elif self.config.dataset_name == "AudioSet":
                sounds = self.resizer(sounds)
            # forward
            with self.ctx:
                out = cmodel(sounds)
                loss = self.criterion(out, labels)
                accumu_out.append(out.cpu().detach())
                accumu_labels.append(labels.cpu().detach())
                accumu_loss += loss

        # accuracy
        out = torch.cat(accumu_out)
        labels = torch.cat(accumu_labels)
        if self.config.dataset_name == "ESC-50":
            accuracy = self.get_accuracy(out, labels)
            res = OrderedDict(
                [
                    ("loss", accumu_loss / len(self.val_loader)),
                    ("accuracy", accuracy),
                ]
            )
        elif self.config.dataset_name == "AudioSet":
            mAP, AUC = self.get_auc(out, labels, sigmoid=True)
            res = OrderedDict(
                [
                    ("loss", accumu_loss / len(self.val_loader)),
                    ("mAP", mAP),
                    ("AUC", AUC),
                ]
            )

        cmodel.train()
        return res

    def load_dataset(self):
        file_list = glob.glob(f"{self.config.data_path}/*.wav")
        with open(f"config_{sys.argv[1]}.json", "r") as fp:
            config = json.load(fp)

        assert len(file_list) > 0
        if self.config.dataset_name == "ESC-50":
            file_set = set(file_list)
            val_set = set(
                glob.glob(f"{self.config.data_path}/{self.config.eval_prefix}*.wav")
            )
            train_set = file_set - val_set

            train_ds = ESC50Dataset(config, list(train_set), self.config.meta_dir)
            val_ds = ESC50Dataset(
                config, list(val_set), self.config.meta_dir, validation=True
            )
        elif self.config.dataset_name == "AudioSet":
            train_ds = AudioSetDataset(
                config,
                "/data/audioset/bal_train*.pkl",
                "/data/audioset/balanced_train_segments.csv",
            )
            val_ds = AudioSetDataset(
                config,
                "/data/audioset/eval00.pkl",
                "/data/audioset/eval_segments.csv",
                validation=True,
            )

        self.train_loader = data.DataLoader(
            train_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            collate_fn=AudioSetDataset.my_collate,
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
            collate_fn=AudioSetDataset.my_collate,
        )

    def init(self, resume: str):
        # create/load model
        model_name = (
            "efficientnet_b0"
            if self.config.dataset_name == "ESC-50"
            else "timm/deit_base_distilled_patch16_384.fb_in1k"
        )
        drop_rate = 0.3 if self.config.dataset_name == "ESC-50" else 0.0
        model = (
            timm.create_model(
                model_name,
                in_chans=1,
                num_classes=self.config.num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_rate,
                pretrained=True,
            )
            .to(self.dtype)
            .to(self.device_type)
        )
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
        if self.config.dataset_name == "ESC-50":
            optimizer = torch.optim.SGD(
                cmodel.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                nesterov=False,
            )
        elif self.config.dataset_name == "AudioSet":
            optimizer = torch.optim.Adam(cmodel.parameters(), self.config.lr)

        best_metric = 0.0
        begin = time.time()

        if self.config.dataset_name == "AudioSet":
            accumu_out = []
            accumu_labels = []

        for iteration in range(iter_start, self.config.max_iters):
            lr = self.get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            out, labels, loss = self.train_step(cmodel, optimizer)
            if self.config.dataset_name == "AudioSet":
                accumu_out.append(out.cpu().detach())
                accumu_labels.append(labels.cpu().detach())

            if iteration % self.config.log_iters == 0 and iteration > 0:
                metrics = OrderedDict(
                    [
                        ("loss", loss.item()),
                    ],
                )
                if self.config.dataset_name == "ESC-50":
                    metrics["accuracy"] = self.get_accuracy(out, labels)
                '''elif self.config.dataset_name == "AudioSet":
                    out = torch.cat(accumu_labels)
                    labels = torch.cat(accumu_labels)
                    metrics["mAP"], metrics["AUC"] = self.get_auc(out, labels)'''

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
                accumulator = self.validate(cmodel, iteration)
                main_metric = (
                    "accuracy" if self.config.dataset_name == "ESC-50" else "mAP"
                )
                val_metric = accumulator[main_metric]
                best_metric = max(best_metric, val_metric)
                checkpoint = {
                    "model": model.state_dict(),
                    "iteration": iteration,
                    "train_config": asdict(self.config),
                    "eval_metric": val_metric,
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

        print("Best validating metric:", best_metric)
        return best_metric


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    multiprocessing.set_start_method("spawn")

    config = TrainConfig()

    if sys.argv[2] == "ESC-50":
        bests = []
        for prefix in ["1", "2", "3", "4", "5"]:
            config.eval_prefix = prefix
            trainer = Trainer(config)
            bests.append(trainer.train())
        print("Avg accuracy:", sum(bests) / len(bests))
    elif sys.argv[2] == "AudioSet":
        config.dataset_name = "AudioSet"
        config.num_classes = 527
        config.batch_size = 16
        config.num_workers = 16
        config.lr = 1e-4
        config.min_lr = 5e-6
        config.lr_decay_iters = 12000 + 1
        config.max_iters = config.lr_decay_iters
        trainer = Trainer(config)
        trainer.train()
