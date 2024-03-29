"""Shared utilities for model training/testing."""

import logging
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import Tensor, nn

import salve.utils.normalization_utils as normalization_utils
import salve.utils.transform as transform
from salve.dataset.zind_data import ZindData
from salve.models.early_fusion import EarlyFusionCEResnet
from salve.training_config import TrainingConfig
from salve.utils.avg_meter import AverageMeter


def cross_entropy_forward(
    model: nn.Module,
    split: str,
    x1: Tensor,
    x2: Tensor,
    x3: Tensor,
    x4: Tensor,
    x5: Tensor,
    x6: Tensor,
    is_match: Tensor,
) -> Tuple[Tensor, Tensor]:
    """ """
    if split == "train":
        logits = model(x1, x2, x3, x4, x5, x6)
        probs = torch.nn.functional.softmax(logits.clone(), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, is_match.squeeze())

    else:
        with torch.no_grad():
            logits = model(x1, x2, x3, x4, x5, x6)
            probs = torch.nn.functional.softmax(logits.clone(), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, is_match.squeeze())

    return probs, loss


def print_time_remaining(batch_time: AverageMeter, current_iter: int, max_iter: int) -> None:
    """Use a running average of time to run a single batch through the network to estimate training time remaining.

    Note: this estimate may include both forward prop and optionally backprop time.
    """
    remain_iter = max_iter - current_iter
    remain_time = remain_iter * batch_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
    logging.info(f"\tRemain {remain_time}")


def poly_learning_rate(base_lr: float, curr_iter: int, max_iter: int, power: float = 0.9) -> float:
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def get_train_transform(args: TrainingConfig) -> Callable:
    """Get data transforms for train split.

    For a single modality, we will get 1 rendering for each panorama in the image pair (2 images in total).
    For two modalities, we will get 2 renderings for each panorama in the image pair (4 images in total).
    For three modalities, we will get 3 renderings for each panorama in the image pair (6 images in total).

    Args:
        args: training hyperparameters.

    Returns:
        A callable object with sequentially chained data transformations.
    """
    if len(args.modalities) == 1:
        Resize = transform.ResizePair
        Crop = transform.CropPair
        ToTensor = transform.ToTensorPair
        Normalize = transform.NormalizePair
        RandomHorizontalFlip = transform.RandomHorizontalFlipPair
        RandomVerticalFlip = transform.RandomVerticalFlipPair
        Compose = transform.ComposePair

    elif len(args.modalities) == 2:
        Resize = transform.ResizeQuadruplet
        Crop = transform.CropQuadruplet
        ToTensor = transform.ToTensorQuadruplet
        Normalize = transform.NormalizeQuadruplet
        RandomHorizontalFlip = transform.RandomHorizontalFlipQuadruplet
        RandomVerticalFlip = transform.RandomVerticalFlipQuadruplet
        Compose = transform.ComposeQuadruplet

    elif len(args.modalities) == 3:
        Resize = transform.ResizeSextuplet
        Crop = transform.CropSextuplet
        ToTensor = transform.ToTensorSextuplet
        Normalize = transform.NormalizeSextuplet
        RandomHorizontalFlip = transform.RandomHorizontalFlipSextuplet
        RandomVerticalFlip = transform.RandomVerticalFlipSextuplet
        Compose = transform.ComposeSextuplet

    mean, std = normalization_utils.get_imagenet_mean_std()

    # TODO: check if cropping helps. currently prevent using the exact same 224x224 square every time
    # We use random crops to prevent memorization.

    transform_list = [Resize(size=(args.resize_h, args.resize_w))]

    if args.apply_photometric_augmentation:
        transform_list += [transform.PhotometricShift(jitter_types=["brightness", "contrast", "saturation", "hue"])]

    transform_list.extend(
        [
            Crop(size=(args.train_h, args.train_w), crop_type="rand", padding=mean),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )
    logging.info("Train transform_list: " + str(transform_list))
    return Compose(transform_list)


def get_val_test_transform(args: TrainingConfig) -> Callable:
    """Get data transforms for val or test split."""
    if len(args.modalities) == 1:
        Resize = transform.ResizePair
        Crop = transform.CropPair
        ToTensor = transform.ToTensorPair
        Normalize = transform.NormalizePair
        Compose = transform.ComposePair

    elif len(args.modalities) == 2:
        Resize = transform.ResizeQuadruplet
        Crop = transform.CropQuadruplet
        ToTensor = transform.ToTensorQuadruplet
        Normalize = transform.NormalizeQuadruplet
        Compose = transform.ComposeQuadruplet

    elif len(args.modalities) == 3:
        Resize = transform.ResizeSextuplet
        Crop = transform.CropSextuplet
        ToTensor = transform.ToTensorSextuplet
        Normalize = transform.NormalizeSextuplet
        Compose = transform.ComposeSextuplet

    mean, std = normalization_utils.get_imagenet_mean_std()

    # Uses deterministic center crops instead of random crops.
    transform_list = [
        Resize((args.resize_h, args.resize_w)),
        Crop(size=(args.train_h, args.train_w), crop_type="center", padding=mean),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]

    return Compose(transform_list)


def get_img_transform_list(args: TrainingConfig, split: str):
    """Get data transform for specified dataset split."""
    if split == "train":
        split_transform = get_train_transform(args)

    elif split in ["val", "test"]:
        split_transform = get_val_test_transform(args)

    return split_transform


def get_optimizer(args: TrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    """ """
    if args.optimizer_algo == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Unknown optimizer")

    return optimizer


def get_dataloader(args: TrainingConfig, split: str) -> torch.utils.data.DataLoader:
    """ """
    data_transform = get_img_transform_list(args, split=split)

    split_data = ZindData(split=split, transform=data_transform, args=args)

    drop_last = True if split == "train" else False
    sampler = None
    shuffle = True if split == "train" else False

    split_loader = torch.utils.data.DataLoader(
        split_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=sampler,
    )
    return split_loader


def get_model(args: TrainingConfig) -> nn.Module:
    """ """
    model = EarlyFusionCEResnet(args.num_layers, args.pretrained, args.num_ce_classes, args)

    logging.info(model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.dataparallel:
        model = torch.nn.DataParallel(model)

    return model


def unnormalize_img(input: Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
    """Undo the normalization operation on a normalized tensor.

    Note: we pass in by reference a Pytorch tensor.
    """
    for t, m, s in zip(input, mean, std):
        t.mul_(s).add_(m)


def load_model_checkpoint(ckpt_fpath: str, model: nn.Module, args: TrainingConfig) -> nn.Module:
    """Load serialized weights from a Pytorch checkpoint file."""
    if not Path(ckpt_fpath).exists():
        raise RuntimeError(f"=> no checkpoint found at {ckpt_fpath}")

    # Alternatively, get device from print(next(model.parameters()).device)

    # map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=> loading checkpoint '{ckpt_fpath}'")
    checkpoint = torch.load(ckpt_fpath)  # , map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=True)  # False)
    print(f"=> loaded checkpoint '{ckpt_fpath}'")

    return model
