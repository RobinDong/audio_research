from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str = "/Users/robin/Downloads/ESC-50-master/npy"
    meta_dir: str = "/Users/robin/Downloads/ESC-50-master/meta"
    eval_ratio: float = 0.1
    num_workers: int = 4
    lr: float = 1e-2
    batch_size: int = 64
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    log_iters: int = 20
    warmup_iters: int = 40
    eval_iters: int = 200
    lr_decay_iters: int = 51200
    max_iters: int = 100000
    num_classes: int = 50
