"""Script to train CNN models. Pytorch DataParallel is employed.

Configure the `home_dir` directory below for the location where log files will be saved.
"""

import argparse
import logging
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from hydra.utils import instantiate

import salve.utils.datetime_utils as datetime_utils
import salve.utils.io as io_utils
import salve.utils.logger_utils as logger_utils
import salve.train_utils as train_utils
from salve.utils.avg_meter import AverageMeter, SegmentationAverageMeter

# logger = logger_utils.get_logger()

# home_dir = "/Users/johnlam/Downloads"
# home_dir = "/mnt/data/johnlam"
home_dir = "/data/johnlam"
logger_utils.setup_file_logger(home_dir, program_name="training")


def check_mkdir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)


def main(args) -> None:
    """Execute training loop for `num_epochs` epochs."""
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True

    logging.info(str(args))

    train_loader = train_utils.get_dataloader(args, split="train")
    val_loader = train_utils.get_dataloader(args, split="val")

    model = train_utils.get_model(args)
    optimizer = train_utils.get_optimizer(args, model)

    # model = train_utils.load_model_checkpoint(
    #     ckpt_fpath="/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26/train_ckpt.pth",
    #     model=model,
    #     args=args
    # )

    cfg_stem = args.cfg_stem
    exp_start_time = datetime_utils.generate_datetime_string()

    results_dict = defaultdict(list)

    # TODO: only available in `config` class
    # results_dict["args"] = [{k: v} for k, v in args.items()]

    for epoch in range(args.num_epochs):

        logging.info(f"On epoch {epoch}")
        train_metrics_dict = run_epoch(args, epoch, model, train_loader, optimizer, split="train")

        for k, v in train_metrics_dict.items():
            results_dict[f"train_{k}"] += [v]

        with torch.no_grad():
            val_metrics_dict = run_epoch(args, epoch, model, val_loader, optimizer, split="val")

        for k, v in val_metrics_dict.items():
            results_dict[f"val_{k}"] += [v]

        crit_acc_stat = "val_mAcc"

        if epoch > 0:
            curr_stat = results_dict[crit_acc_stat][-1]
            prev_best = max(results_dict[crit_acc_stat][:-1])
            is_best = curr_stat > prev_best

        # If current model parameters represent the best model thus far, save to disk.
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
                    f"curr_{crit_acc_stat}": results_dict[crit_acc_stat][-1],
                    f"best_so_far_{crit_acc_stat}": max(results_dict[crit_acc_stat]),
                },
                ckpt_fpath,
            )

        results_json_fpath = f"{results_dir}/results-{exp_start_time}-{cfg_stem}.json"
        io_utils.save_json_file(json_fpath=results_json_fpath, data=results_dict)
        shutil.copyfile(f"salve/configs/{args.cfg_stem}.yaml", f"{results_dir}/{args.cfg_stem}.yaml")

        logging.info("Results on crit stat: " + str([f"{v:.3f}" for v in results_dict[crit_acc_stat]]))


def visualize_unnormalized_examples(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    is_match: torch.Tensor,
    fp0: List[str],
    fp1: List[str],
) -> None:
    """Visualize training examples right before the forward pass through the network.

    Note: Purely for debugging input to the network
    """

    # TODO(johnwlambert): verify the type of fp0 and fp1
    import matplotlib.pyplot as plt
    import salve.utils.normalization_utils as normalization_utils

    n = x1.shape[0]

    for k in range(n):
        plt.figure(figsize=(10, 5))
        mean, std = normalization_utils.get_imagenet_mean_std()

        train_utils.unnormalize_img(x1[k].cpu(), mean, std)
        train_utils.unnormalize_img(x2[k].cpu(), mean, std)

        plt.subplot(2, 2, 1)
        plt.imshow(x1[k].numpy().transpose(1, 2, 0).astype(np.uint8))

        plt.subplot(2, 2, 2)
        plt.imshow(x2[k].numpy().transpose(1, 2, 0).astype(np.uint8))

        if x3 is not None:
            train_utils.unnormalize_img(x3[k].cpu(), mean, std)
            train_utils.unnormalize_img(x4[k].cpu(), mean, std)

            plt.subplot(2, 2, 3)
            plt.imshow(x3[k].numpy().transpose(1, 2, 0).astype(np.uint8))

            plt.subplot(2, 2, 4)
            plt.imshow(x4[k].numpy().transpose(1, 2, 0).astype(np.uint8))

        plt.title("Is match" + str(is_match[k].numpy().item()))

        print(fp0[k])
        print(fp1[k])
        print()

        plt.show()
        plt.close("all")


def run_epoch(
    args, epoch: int, model, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, split: str
) -> Dict[str, float]:
    """Run all samples belonging to a particular data split through the network (single epoch)."""
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

        # Assume cross entropy loss only currently.
        if (
            args.modalities == ["layout"]
            or args.modalities == ["ceiling_rgb_texture"]
            or args.modalities == ["floor_rgb_texture"]
        ):
            x1, x2, is_match, fp0, fp1 = example
            x3, x4, x5, x6 = None, None, None, None

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture"]):
            x1, x2, x3, x4, is_match, fp0, fp1 = example
            x5, x6 = None, None

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture", "layout"]):
            x1, x2, x3, x4, x5, x6, is_match, fp0, fp1 = example

        n = x1.size(0)

        debug = False
        if debug:
            visualize_unnormalized_examples(x1, x2, x3, x4, is_match, fp0, fp1)

        if torch.cuda.is_available():
            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            x3 = x3.cuda(non_blocking=True) if x3 is not None else None
            x4 = x4.cuda(non_blocking=True) if x4 is not None else None
            x5 = x5.cuda(non_blocking=True) if x5 is not None else None
            x6 = x6.cuda(non_blocking=True) if x6 is not None else None

            gt_is_match = is_match.cuda(non_blocking=True)
        else:
            gt_is_match = is_match

        is_match_probs, loss = train_utils.cross_entropy_forward(
            model, split, x1, x2, x3, x4, x5, x6, gt_is_match
        )

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
                current_lr = train_utils.poly_learning_rate(
                    args.base_lr, current_iter, max_iter, power=args.poly_lr_power
                )

                if iter % args.print_every == 0:
                    logging.info(
                        f"\tLR:{current_lr:.5f}, base_lr: {args.base_lr:.3f}, "
                        f"current_iter:{current_iter}, max_iter:{max_iter}, power:{args.poly_lr_power}"
                    )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            loss_meter.update(loss.item(), n)

            if iter > 0:
                # When computing timing running stats, ignore the first iter, while GPU warms up.
                batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_every == 0:
                _, accs, _, avg_mAcc, _ = sam.get_metrics()
                logging.info(f"\t{args.num_ce_classes}-Cls Accuracies" + str([float(f"{acc:.2f}") for acc in accs]))
                logging.info(
                    f"\t{split} result at iter [{iter+1}/{len(data_loader)}] "
                    f"{args.num_ce_classes}-CE mAcc {avg_mAcc:.4f}"
                )
                train_utils.print_time_remaining(batch_time=batch_time, current_iter=current_iter, max_iter=max_iter)

    _, accs, _, avg_mAcc, _ = sam.get_metrics()
    logging.info(f"{split} result at epoch [{epoch+1}/{args.num_epochs}]: mAcc{avg_mAcc:.4f}")
    logging.info("Cls Accuracies: " + str([float(f"{acc:.2f}") for acc in accs]))

    metrics_dict = {"avg_loss": loss_meter.avg, "mAcc": avg_mAcc}

    return metrics_dict


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, required=True, help="GPU device IDs to use for training.")
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="File name of config file under `salve/configs/*` (not file path!). Should end in .yaml",
    )
    opts = parser.parse_args()

    with hydra.initialize_config_module(config_module="salve.configs"):
        # config is relative to the `salve` module
        cfg = hydra.compose(config_name=opts.config_name)
        args = instantiate(cfg.TrainingConfig)

    # Always take GPU ids from the command line, not the config.
    args.gpu_ids = opts.gpu_ids
    if not args.cfg_stem:
        args.cfg_stem = Path(opts.config_name).stem

    print("Using GPUs ", args.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    main(args)
