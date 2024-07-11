from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str = "/data/audio/ESC-50-master/audio"
    meta_dir: str = "/data/audio/ESC-50-master/meta"
    eval_ratio: float = 0.01
    num_workers: int = 4
    lr: float = 1e-1
    batch_size: int = 16
    min_lr: float = 3e-3
    grad_clip: float = 10.0
    log_iters: int = 100
    eval_iters: int = 500
    warmup_iters: int = 1 * eval_iters
    lr_decay_iters: int = 12000 + 1
    max_iters: int = lr_decay_iters
    num_classes: int = 50
    dataset_name: str = "ESC-50"
    eval_prefix: str = "1"  # only for ESC-50
