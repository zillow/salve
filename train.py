
import logging
import os
import random
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from argoverse.utils.datetime_utils import generate_datetime_string
from argoverse.utils.json_utils import save_json_dict

from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter

from train_utils import (
    poly_learning_rate,
    get_optimizer,
    get_dataloader,
    get_model,
    cross_entropy_forward,
    print_time_remaining,
)
from logger_utils import get_logger

logger = get_logger()


def check_mkdir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)


def main(args) -> None:
    """ """
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True

    logger.info(str(args))

    train_loader = get_dataloader(args, split="train")
    val_loader = get_dataloader(args, split="val")

    model = get_model(args)
    optimizer = get_optimizer(args, model)

    cfg_stem = args.cfg_stem
    exp_start_time = generate_datetime_string()

    results_dict = defaultdict(list)

    # TODO: only available in `config` class
    # results_dict["args"] = [{k: v} for k, v in args.items()]

    for epoch in range(args.num_epochs):

        logger.info(f"On epoch {epoch}")
        train_metrics_dict = run_epoch(args, epoch, model, train_loader, optimizer, split="train")

        for k, v in train_metrics_dict.items():
            results_dict[f"train_{k}"] += [v]

        val_metrics_dict = run_epoch(args, epoch, model, val_loader, optimizer, split="val")

        for k, v in vall_metrics_dict.items():
            results_dict[f"val_{k}"] += [v]

        crit_acc_stat = "val_mAcc"

        if epoch > 0:
            curr_stat = results_dict[crit_acc_stat][-1]
            prev_best = max(results_dict[crit_acc_stat][:-1])
            is_best = curr_stat > prev_best

        # if the best model, save to disk
        if epoch == 0 or is_best:

            results_dir = f"{args.model_save_dirpath}/{exp_start_time}"
            check_mkdir(results_dir)
            ckpt_fpath = f"{results_dir}/train_ckpt.pth"
            logger.info(f"Saving checkpoint to: {ckpt_fpath}")

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "max_epochs": args.num_epochs,
                },
                ckpt_fpath,
            )

        results_json_fpath = f"{results_dir}/results-{exp_start_time}-{cfg.stem}.json"
        save_json_dict(results_json_fpath, results_dict)

        logger.info("Results on crit stat: " + str([f"v:.3f" for v in results_dict[crit_acc_stat]]))


def run_epoch(args, epoch: int, model, data_loader, optimizer, split: str) -> Dict[str, float]:
    """Run all data belonging to a particular split through the network."""
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    sam = SegmentationAverageMeter()

    if split == "train":
        model.train()

    else:
        model.eval()

    end = time.time()

    for iter, example in enumerate(data_loader):

        if iter % args.print_every == 0:
            logger.info(f"\tOn iter {iter}")

        # assume cross entropy loss only currently
        x1, x2, x3, x4, is_match = example

        n = x1.size(0)

        #"""
        for k in range(n):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,5))
            mean, std = get_imagenet_mean_std()
            from train_utils import unnormalize_img

            unnormalize_img(x1[k].cpu(), mean, std)

            plt.subplot(2,2,1)
            plt.imshow(x1[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,2)
            plt.imshow(x2[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,3)
            plt.imshow(x3[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,4)
            plt.imshow(x4[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.title("Is match" + str(is_match[k]).numpy().item())

            plt.show()
            plt.close("all")
        #"""

        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)
        x3 = x3.cuda(non_blocking=True)
        x4 = x4.cuda(non_blocking=True)

        gt_is_match = is_match.cuda(non_blocking=True)

        

        is_match_probs, loss = cross_entropy_forward(model, args, split, x1, x2, x3, x4, gt_is_match)

        sam.update_metrics_cpu(
            pred=torch.argmax(is_match_probs, dim=1).cpu().numpy(),
            target=gt_is_match.squeeze().cpu().numpy(),
            num_classes=args.num_ce_classes,
        )

        max_iter = args.num_epochs * len(data_loader)
        current_iter = epoch * len(data_loader) + iter + 1

        if split == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if split == "train" and args.lr_annealing_strategy == "poly":
                # decay learning rate only during training
                current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.poly_lr_power)

                if iter % args.print_every == 0:
                    logger.info(
                        f"\tLR:{current_lr:.5f}, base_lr: {args.base_lr:.3f}, current_iter:{current_iter}, max_iter:{max_iter}, power:{args.poly_lr_power}"
                    )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            loss_meter.update(loss.item(), n)

            if iter > 0:
                # ignore the first iter, while GPU warms up
                batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_every == 0:
                _, accs, _, avg_mAcc, _ = sam.get_metrics()
                logger.info(f"\t{args.num_ce_classes}-Cls Accuracies" + str([float(f"{acc:.2f}") for acc in accs]))
                logger.info(
                    f"\t{split} result at iter [{iter+1}/{len(data_loader)}] {args.num_ce_classes}-CE mAcc {avg.mAcc:.4f}"
                )
                print_time_remaining(batch_time, current_iter, max_iter)

    _, accs, _, avg_mAcc, _ = sam.get_metrics()
    logger.info(f"{split} result at epoch [{epoch+1}/{args.num_epochs}]: mAcc{avg_mAcc:.4f}")
    logger.info("Cls Accuracies: " + str([float(f"acc:.2f") for acc in accs]))

    metrics_dict = {"avg_loss": loss_meter.avg, "mAcc": avg_mAcc}

    return metrics_dict


if __name__ == "__main__":

    args = SimpleNamespace(
        **{
            "cfg_stem": "rgb_only_4tuple",
            "num_epochs": 10,
            "lr_annealing_strategy": "poly",
            "base_lr": 0.001,
            "weight_decay": 0.0001,
            "num_ce_classes": 2,
            "model_save_dirpath": "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25",
            "print_every": 10,
            "poly_lr_power": 0.9,
            "optimizer_algo": "adam",
            "batch_size": 256,
            "workers": 8,
            "num_layers": 18,
            "pretrained": True,
            "dataparallel": True,
            "resize_h": 224,
            "resize_w": 224,
            "data_root": "/Users/johnlam/Downloads/DGX-rendering-2021_06_25/ZinD_BEV_RGB_only_2021_06_25"
        }
    )
    main(args)
