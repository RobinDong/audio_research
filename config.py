from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_path: str = "/data/audio/ESC-50-master/audio"
    meta_dir: str = "/data/audio/ESC-50-master/meta"
    eval_ratio: float = 0.01
    num_workers: int = 32
    lr: float = 1e-1
    batch_size: int = 48
    min_lr: float = 3e-3
    grad_clip: float = 10.0
    log_iters: int = 100
    eval_iters: int = 500
    epochs: int = 25
    num_classes: int = 50
    dataset_name: str = "ESC-50"
    eval_prefix: str = "1"  # only for ESC-50
