import os
import glob
import time
import math
import fire
import timm
import contextlib
import numpy as np

from collections import defaultdict, OrderedDict
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.utils import data
from config import TrainConfig
from datasets.esc50 import ESC50Dataset

SEED = 20240605
CKPT_DIR = "out"
LABEL_SMOOTH_RATIO = 0.5


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

    def criterion(self, outputs, targets):
        one_hot = torch.zeros((targets.shape[0], self.config.num_classes), dtype=self.dtype, device=self.device_type)
        one_hot.fill_((1 - LABEL_SMOOTH_RATIO) / self.config.num_classes)
        one_hot.scatter_(1, targets.unsqueeze(-1), LABEL_SMOOTH_RATIO)
        return torch.sum(-one_hot * F.log_softmax(outputs, -1), -1).mean()

    def train_step(self, model, optimizer):
        try:
            data_entry = next(self.train_batch_iter)
            if len(data_entry[0]) < self.config.batch_size:
                self.train_batch_iter = iter(self.train_loader)
                data_entry = next(self.train_batch_iter)
        except StopIteration:
            self.train_batch_iter = iter(self.train_loader)
            data_entry = next(self.train_batch_iter)

        sounds, label = data_entry
        sounds = sounds.unsqueeze(-1).permute(0, 3, 1, 2).to(self.dtype).to(self.device_type)
        label = label.to(self.device_type)

        with self.ctx:
            out = model(sounds)
            loss = self.criterion(out, label)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return out, loss

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

    @torch.no_grad()
    def validate(self, cmodel):
        cmodel.eval()

        batch_iter = iter(self.val_loader)
        accumulator = defaultdict(float)
        length = len(self.val_loader)
        for _ in range(length - 1):
            data_entry = next(batch_iter)
            metrics = self.train_provider.get_validation_metrics(
                data_entry, cmodel, self.ctx, self.device_type
            )
            for key, val in metrics.items():
                accumulator[key] += val

        for key, val in accumulator.items():
            accumulator[key] /= length

        cmodel.train()
        return accumulator

    def init(self, resume: str):
        # create/load model
        model = timm.create_model(
            "convnextv2_atto",
            in_chans=1,
            num_classes=config.num_classes,
            drop_rate=0,
            drop_path_rate=0,
        ).to(self.dtype).to(self.device_type)
        if resume:
            checkpoint = torch.load(resume, map_location=self.device_type)
            state_dict = checkpoint["model"]
            self.config = TrainConfig(**checkpoint["train_config"])
            iter_start = checkpoint["iteration"] + 1
            model.load_state_dict(state_dict)
            print(f"Resume from {iter_start - 1} for training...")
        else:
            iter_start = 1

        # load dataset
        file_list = glob.glob(f"{self.config.data_path}/*.npy")
        assert len(file_list) > 0
        np.random.seed(SEED)
        np.random.shuffle(file_list)
        point = int(len(file_list) * self.config.eval_ratio)
        train_lst, val_lst = file_list[:point], file_list[point:]
        train_ds = ESC50Dataset(train_lst, self.config.meta_dir)
        val_ds = ESC50Dataset(val_lst, self.config.meta_dir, validation=True)
        self.train_loader = data.DataLoader(
            train_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.train_batch_iter = iter(self.train_loader)

        self.val_loader = data.DataLoader(
            val_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return model, iter_start

    def train(self, resume="", learning_rate=None):
        model, iter_start = self.init(resume)
        if learning_rate:
            self.config.lr = learning_rate
            iter_start = 0
        # cmodel = torch.compile(model)
        cmodel = model
        optimizer = torch.optim.AdamW(
            cmodel.parameters(),
            lr=self.config.lr,
            weight_decay=0.0,
            amsgrad=True,
        )

        begin = time.time()

        for iteration in range(iter_start, self.config.max_iters):
            lr = self.get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            out, loss = self.train_step(cmodel, optimizer)

            if iteration % self.config.log_iters == 0 and iteration > 0:
                metrics = OrderedDict(
                    [
                        ("loss", loss.item()),
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
                avg_accuracy = accumulator["accuracy"]
                checkpoint = {
                    "model": model.state_dict(),
                    "iteration": iteration,
                    "train_config": asdict(self.config),
                    "eval_accuracy": avg_accuracy,
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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    config = TrainConfig()
    trainer = Trainer(config)

    fire.Fire(trainer.train)
