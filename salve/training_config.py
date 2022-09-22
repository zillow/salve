"""Stores hyperparameters for training a model (for a single experiment)."""

from typing import List

from dataclasses import dataclass


@dataclass(frozen=False)
class TrainingConfig:
    """Model training hyperparameters.

    Attributes:
        lr_annealing_strategy: learning rate annealing strategy.
        base_lr: base learning rate.
        weight_decay: weight decay.
        num_ce_classes: number of classes for cross entropy loss.
        print_every: 
        poly_lr_power: polynomial learning rate power.
        optimizer_algo: optimizer algorithm (adam vs. sgd).
        num_layers: number of ResNet layers.
        pretrained: whether to initialize ResNet from ImageNet pretrained weights.
        dataparallel: bool
        resize_h: image height (in pixels) to resize input image to, before cropping.
        resize_w: image width (in pixels) to resize input image to, before cropping.
        train_h: image crop height for training.
        train_w: image crop width for training.
        apply_photometric_augmentation: whether to apply photometric augmentations
            during training time.
        modalities:
        cfg_stem:
        num_epochs: number of epochs to train for.
        workers: how many subprocesses to use for data loading.
        batch_size: batch size (how many samples per batch to load).
        data_root: 
        layout_data_root: 
        model_save_dirpath: 
        gpu_ids: 
    """

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
    modalities = ["layout"]

    cfg_stem: str
    num_epochs: int
    workers: int
    batch_size: int

    data_root: str
    layout_data_root: str
    model_save_dirpath: str
    gpu_ids: str = None

    # TODO: make a rendering config with resolution, spatial extent stored in it.
