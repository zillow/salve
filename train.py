
import argparse
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from argoverse.utils.datetime_utils import generate_datetime_string
from argoverse.utils.json_utils import save_json_dict
from hydra.utils import instantiate
from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter

from train_utils import (
    poly_learning_rate,
    get_optimizer,
    get_dataloader,
    get_model,
    cross_entropy_forward,
    print_time_remaining,
)
from logger_utils import get_logger, setup_file_logger


#logger = get_logger()

home_dir = "/Users/johnlam/Downloads"
#home_dir = "/mnt/data/johnlam"
setup_file_logger(home_dir, program_name="training")


def check_mkdir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)


def main(args) -> None:
    """ """
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True

    logging.info(str(args))

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

        logging.info(f"On epoch {epoch}")
        train_metrics_dict = run_epoch(args, epoch, model, train_loader, optimizer, split="train")

        for k, v in train_metrics_dict.items():
            results_dict[f"train_{k}"] += [v]

        val_metrics_dict = run_epoch(args, epoch, model, val_loader, optimizer, split="val")

        for k, v in val_metrics_dict.items():
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
            logging.info(f"Saving checkpoint to: {ckpt_fpath}")

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "max_epochs": args.num_epochs,
                },
                ckpt_fpath,
            )

        results_json_fpath = f"{results_dir}/results-{exp_start_time}-{cfg_stem}.json"
        save_json_dict(results_json_fpath, results_dict)

        logging.info("Results on crit stat: " + str([f"{v:.3f}" for v in results_dict[crit_acc_stat]]))


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
            logging.info(f"\tOn iter {iter}")

        # assume cross entropy loss only currently
        x1, x2, x3, x4, is_match, fp0, fp1, fp2, fp3 = example

        n = x1.size(0)

        """
        # debug routines
        for k in range(n):
            import matplotlib.pyplot as plt
            from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
            from train_utils import unnormalize_img

            plt.figure(figsize=(10,5))
            mean, std = get_imagenet_mean_std()

            unnormalize_img(x1[k].cpu(), mean, std)
            unnormalize_img(x2[k].cpu(), mean, std)
            unnormalize_img(x3[k].cpu(), mean, std)
            unnormalize_img(x4[k].cpu(), mean, std)

            plt.subplot(2,2,1)
            plt.imshow(x1[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,2)
            plt.imshow(x2[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,3)
            plt.imshow(x3[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.subplot(2,2,4)
            plt.imshow(x4[k].numpy().transpose(1,2,0).astype(np.uint8))

            plt.title("Is match" + str(is_match[k].numpy().item()))

            print(fp0[k])
            print(fp1[k])
            print(fp2[k])
            print(fp3[k])
            print()

            plt.show()
            plt.close("all")
        """
        
        if torch.cuda.is_available():
            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            x3 = x3.cuda(non_blocking=True)
            x4 = x4.cuda(non_blocking=True)

            gt_is_match = is_match.cuda(non_blocking=True)
        else:
            gt_is_match = is_match

        #import pdb; pdb.set_trace()
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
                    logging.info(
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
                logging.info(f"\t{args.num_ce_classes}-Cls Accuracies" + str([float(f"{acc:.2f}") for acc in accs]))
                logging.info(
                    f"\t{split} result at iter [{iter+1}/{len(data_loader)}] {args.num_ce_classes}-CE mAcc {avg_mAcc:.4f}"
                )
                print_time_remaining(batch_time, current_iter, max_iter)

    _, accs, _, avg_mAcc, _ = sam.get_metrics()
    logging.info(f"{split} result at epoch [{epoch+1}/{args.num_epochs}]: mAcc{avg_mAcc:.4f}")
    logging.info("Cls Accuracies: " + str([float(f"{acc:.2f}") for acc in accs]))

    metrics_dict = {"avg_loss": loss_meter.avg, "mAcc": avg_mAcc}

    return metrics_dict


@dataclass(frozen=False)
class TrainingConfig:
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

    cfg_stem: str
    num_epochs: int
    workers: int
    batch_size: int

    data_root: str
    model_save_dirpath: str
    gpu_ids: str = None



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, required=True, help="GPU device IDs to use for training.")
    opts = parser.parse_args()

    config_name = "2021_06_26_08_38_09__resnet18_floor_ceiling_rgbonly.yaml"

    with hydra.initialize_config_module(config_module="afp.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # always take from the command line
    args.gpu_ids = opts.gpu_ids

    print("Using GPUs ", args.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    main(args)
