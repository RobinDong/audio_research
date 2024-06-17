from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str = "/data/audio/esc_npy"
    meta_dir: str = "/data/audio/ESC-50-master/meta"
    eval_ratio: float = 0.01
    num_workers: int = 4
    lr: float = 1e-2
    batch_size: int = 64
    min_lr: float = 5e-3
    grad_clip: float = 10.0
    log_iters: int = 100
    eval_iters: int = 500
    warmup_iters: int = 2 * eval_iters
    lr_decay_iters: int = 6000
    max_iters: int = lr_decay_iters
    num_classes: int = 50
