
"""Stores hyperparameters for training a model (for a single experiment)."""

from dataclasses import dataclass

@dataclass(frozen=False)
class TrainingConfig:
    """Model training parameters."""
    lr_annealing_strategy: str
    base_lr: float
    weight_decay: float
    num_ce_classes: int
    print_every: int
    poly_lr_power: float
    optimizer_algo: str
    num_layers: int
    pretrained: bool
    dataparallel: bool
    resize_h: int
    resize_w: int
    train_h: int
    train_w: int
    apply_photometric_augmentation: bool
    modalities: ["layout"]

    cfg_stem: str
    num_epochs: int
    workers: int
    batch_size: int

    data_root: str
    layout_data_root: str
    model_save_dirpath: str
    gpu_ids: str = None

    # TODO: make a rendering config with resolution, spatial extent stored in it.
