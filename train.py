import os
import sys
import json
import glob
import time
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
        # self.resizer = torchvision.transforms.Resize((384, 384))

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

    def mixup_data(self, sounds, labels):
        lam = np.random.beta(10, 10)

        batch_size = sounds.size(0)
        index = torch.randperm(batch_size).cuda()

        mixed_sounds = lam * sounds + (1 - lam) * sounds[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
        return mixed_sounds, mixed_labels

    def train_step(self, model, optimizer, batch):
        sounds, labels = batch
        # shape of 'sounds' for AudioSet [batch_size, 1, 1024, 128]
        sounds = sounds.unsqueeze(1).to(self.device_type).to(self.dtype)
        labels = labels.to(self.device_type)
        if self.config.dataset_name == "ESC-50":
            sounds, labels = self.cm(sounds, labels)
        elif self.config.dataset_name == "AudioSet":
            # sounds = self.resizer(sounds)
            sounds = sounds.transpose(2, 3)
            sounds, labels = self.mixup_data(sounds, labels)

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
    def validate(self, cmodel):
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
                # sounds = self.resizer(sounds)
                sounds = sounds.transpose(2, 3)
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
                "/data/audioset/eval*.pkl",
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
            self.config.batch_size * 8,
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
        drop_rate = 0.3 if self.config.dataset_name == "ESC-50" else 0.1
        model = (
            timm.create_model(
                model_name,
                img_size = (128, 998),
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
            self.config.lr = 1e-3
            self.config.batch_size = 16
            model.load_state_dict(state_dict)
            print("Resume training...")

        self.load_dataset()
        return model

    def train(self, resume="", learning_rate=None):
        model = self.init(resume)
        if learning_rate:
            self.config.lr = learning_rate
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

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, list(range(1, self.config.epochs, 5)), gamma=0.5
        )

        log_iters = len(self.train_loader) // 5

        for epoch in range(self.config.epochs):
            for iteration, batch in enumerate(self.train_loader):
                out, labels, loss = self.train_step(cmodel, optimizer, batch)
                if self.config.dataset_name == "AudioSet":
                    accumu_out.append(out.cpu().detach())
                    accumu_labels.append(labels.cpu().detach())

                if iteration % log_iters == 0 and iteration > 0:
                    metrics = OrderedDict(
                        [
                            ("loss", loss.item()),
                        ],
                    )
                    if self.config.dataset_name == "ESC-50":
                        metrics["accuracy"] = self.get_accuracy(out, labels)
                    elif self.config.dataset_name == "AudioSet":
                        out = torch.cat(accumu_out)
                        labels = torch.cat(accumu_labels)
                        # metrics["mAP"], metrics["AUC"] = self.get_auc(out, labels)
                        accumu_out = []
                        accumu_labels = []

                    now = time.time()
                    duration = now - begin
                    begin = now
                    messages = [f"[{epoch:03d}: {iteration:06d}]"]
                    for name, val in metrics.items():
                        messages.append(f"{name}: {val:.3f}")
                    lr = optimizer.param_groups[0]["lr"]
                    messages.append(f"lr: {lr:.3e}")
                    messages.append(f"time: {duration:.1f}")
                    print(" ".join(messages), flush=True)

            accumulator = self.validate(cmodel)
            main_metric = "accuracy" if self.config.dataset_name == "ESC-50" else "mAP"
            val_metric = accumulator[main_metric]
            best_metric = max(best_metric, val_metric)
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
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
            scheduler.step()

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
        config.lr = 2e-4
        trainer = Trainer(config)
        trainer.train()
