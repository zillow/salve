

from typing import Callable, List, Tuple

import torch
from mseg_semantic.utils.avg_meter import AverageMeter
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from torch import nn, Tensor

import tbv_transform
from early_fusion import EarlyFusionCEResnet
from logger_utils import get_logger

from zind_data import ZindData

logger = get_logger()



def cross_entropy_forward(
    model: nn.Module,
    args,
    split: str,
    x1: Tensor,
    x2: Tensor,
    x3: Tensor,
    x4: Tensor,
    gt_is_match: Tensor
    ) -> Tuple[Tensor, Tensor]:
    """ """
    if split == "train":
        logits = model(x1, x2, x3, x4)
        probs = torch.nn.functional.softmax(logits.clone(), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, is_match.squeeze())

    else:
        with torch.no_grad():
            logits = model(x1, x2, x3, x4)
            probs = torch.nn.functional.softmax(logits.clone(), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, is_match.squeeze())

    return probs, loss


def print_time_remaining(batch_time: AverageMeter, current_iter: int, max_iter: int) -> None:
    """ """
    remain_iter = max_iter - current_iter
    remain_time = remain_iter * batch_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    logger.info(f"\tRemain {remain_time}")


def poly_learning_rate(base_lr: float, curr_iter: int, max_iter: int, power: float = 0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr



def get_train_transform_list(args) -> List[Callable]:
    """ """
    mean, std = get_imagenet_mean_std()

    # TODO: check if cropping helps. currently prevent using the exact same 224x224 square every time
    # random crops to prevent memorization

    transform_list = [
        tbv_transform.ResizeQuadruplet((args.resize_h, args.resize_w)),
        tbv_transform.CropQuadruplet(size=(args.train_h, args.train_w), crop_type="rand", padding=mean),
        tbv_transform.RandomHorizontalFlipQuadruplet(),
        tbv_transform.RandomVerticalFlipQuadruplet(),
        tbv_transform.ToTensorQuadruplet(),
        tbv_transform.NormalizeQuadruplet(mean=mean, std=std)
    ]
    return transform_list



def get_val_test_transform_list(args) -> List[Callable]:
    """Get data transforms for val or test split"""
    mean, std = get_imagenet_mean_std()

    transform_list = [
        tbv_transform.ResizeQuadruplet((args.resize_h, args.resize_w)),
        tbv_transform.CropQuadruplet(size=(args.train_h, args.train_w), crop_type="center", padding=mean),
        tbv_transform.ToTensorQuadruplet(),
        tbv_transform.NormalizeQuadruplet(mean=mean, std=std)
    ]

    return transform_list


def get_img_transform_list(args, split: str):
    """ """
    if split == "train":
        transform_list = get_train_transform_list(args)

    elif split in ["val", "test"]:
        transform_list = get_val_test_transform_list(args)

    return tbv_transform.ComposeQuadruplet(transform_list)


def get_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    """ """
    if args.optimizer_algo == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError("Unknown optimizer")


    return optimizer



def get_dataloader(args, split: str) -> torch.utils.data.DataLoader:
    """ """
    data_transform = get_img_transform_list(args, split=split)

    split_data = ZindData(
        split=split,
        transform=data_transform,
        args=args
    )

    drop_last = True if split=="train" else False
    sampler = None
    shuffle = True if split == "train" else False

    split_loader = torch.utils.data.DataLoader(
        split_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=sampler
    )
    return split_loader


def get_model(args) -> nn.Module:
    """ """
    model = EarlyFusionCEResnet(args.num_layers, args.pretrained, args.num_ce_classes, args)

    logger.info(model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.dataparallel:
        model = torch.nn.DataParallel(model)

    return model


def unnormalize_img(input: Tensor, mean: Tuple[float,float,float], std: Tuple[float,float,float]) -> None:
    """Pass in by reference a Pytorch tensor.
    """

    for t,m,s in zip(input, mean, std):
        t.mul_(s).add_(m)




