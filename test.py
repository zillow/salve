"""
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import argoverse.utils.json_utils as json_utils
import hydra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from hydra.utils import instantiate
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from mseg_semantic.utils.avg_meter import SegmentationAverageMeter
from torch import nn

import afp.utils.pr_utils as pr_utils
import train_utils as train_utils
from afp.training_config import TrainingConfig
from afp.utils.logger_utils import get_logger


logger = get_logger()


def run_test_epoch(
    serialization_save_dir: str, ckpt_fpath: str, model: nn.Module, data_loader, split: str, save_viz: bool
) -> Dict[str, Any]:
    """ """
    pr_meter = PrecisionRecallMeter()
    sam = SegmentationAverageMeter()

    for i, test_example in enumerate(data_loader):

        # assume cross entropy loss only currently
        if (
            args.modalities == ["layout"]
            or args.modalities == ["ceiling_rgb_texture"]
            or args.modalities == ["floor_rgb_texture"]
        ):
            x1, x2, is_match, fp0, fp1 = test_example
            x3, x4, x5, x6 = None, None, None, None

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture"]):
            x1, x2, x3, x4, is_match, fp0, fp1 = test_example
            x5, x6 = None, None

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture", "layout"]):
            x1, x2, x3, x4, x5, x6, is_match, fp0, fp1 = test_example

        # assume cross entropy loss only currently
        n = x1.size(0)

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
            model, args, split, x1, x2, x3, x4, x5, x6, gt_is_match
        )

        y_hat = torch.argmax(is_match_probs, dim=1)

        sam.update_metrics_cpu(
            pred=torch.argmax(is_match_probs, dim=1).cpu().numpy(),
            target=gt_is_match.squeeze().cpu().numpy(),
            num_classes=args.num_ce_classes,
        )

        pr_meter.update(
            y_true=gt_is_match.squeeze().cpu().numpy(), y_hat=torch.argmax(is_match_probs, dim=1).cpu().numpy()
        )

        if save_viz:
            visualize_examples(
                ckpt_fpath=ckpt_fpath,
                batch_idx=i,
                split=split,
                args=args,
                x1=x1,
                x2=x2,
                x3=x3,
                x4=x4,
                y_hat=y_hat,
                y_true=gt_is_match,
                probs=is_match_probs,
                fp0=fp0,
                fp1=fp1,
            )

        serialize_predictions = True
        if serialize_predictions:
            save_edge_classifications_to_disk(
                serialization_save_dir,
                batch_idx=i,
                y_hat=y_hat,
                y_true=gt_is_match,
                probs=is_match_probs,
                fp0=fp0,
                fp1=fp1,
            )

        _, accs, _, avg_mAcc, _ = sam.get_metrics()
        print(f"{split} result: mAcc{avg_mAcc:.4f}", "Cls Accuracies:", [float(f"{acc:.2f}") for acc in accs])

        # check recall and precision
        # treat correctly aligned as a `positive`
        prec, rec, mAcc = pr_meter.get_metrics()
        print(f"Iter {i}/{len(data_loader)} Prec {prec:.2f}, Rec {rec:.2f}, mAcc {mAcc:.2f}")

    metrics_dict = {}
    return metrics_dict


class PrecisionRecallMeter:
    def __init__(self) -> None:
        """ """
        self.all_y_true = np.zeros((0, 1))
        self.all_y_hat = np.zeros((0, 1))

    def update(self, y_true: np.ndarray, y_hat: np.ndarray) -> None:
        """ """
        y_true = y_true.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)

        self.all_y_true = np.vstack([self.all_y_true, y_true])
        self.all_y_hat = np.vstack([self.all_y_hat, y_hat])

    def get_metrics(self) -> Tuple[float, float, float]:
        """ """
        prec, rec, mAcc = pr_utils.compute_precision_recall(y_true=self.all_y_true, y_pred=self.all_y_hat)
        return prec, rec, mAcc


def save_edge_classifications_to_disk(
    serialization_save_dir: str,
    batch_idx: int,
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    probs: torch.Tensor,
    fp0: torch.Tensor,
    fp1: torch.Tensor,
) -> None:
    """ """
    n = y_hat.shape[0]
    save_dict = {
        "y_hat": y_hat.cpu().numpy().tolist(),
        "y_true": y_true.cpu().numpy().tolist(),
        "y_hat_probs": probs[torch.arange(n), y_hat].cpu().numpy().tolist(),
        "fp0": fp0,
        "fp1": fp1,
    }
    os.makedirs(serialization_save_dir, exist_ok=True)
    json_utils.save_json_dict(f"{serialization_save_dir}/batch_{batch_idx}.json", save_dict)


def visualize_examples(
    ckpt_fpath: str,
    batch_idx: int,
    split: str,
    args,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    **kwargs,
) -> None:
    """ """
    vis_save_dir = f"{Path(ckpt_fpath).parent}/{split}_examples_2021_07_28"

    y_hat = kwargs["y_hat"]
    y_true = kwargs["y_true"]
    probs = kwargs["probs"]

    fp0 = kwargs["fp0"]
    fp1 = kwargs["fp1"]

    n, _, h, w = x1.shape

    for j in range(n):

        pred_label_idx = y_hat[j].cpu().numpy().item()
        true_label_idx = y_true[j].cpu().numpy().item()

        save_fps_only = True
        if save_fps_only:
            is_fp = pred_label_idx == 1 and true_label_idx == 0
            if not is_fp:
                plt.close("all")
                continue

        fig = plt.figure(figsize=(10, 5))
        # gs1 = gridspec.GridSpec(ncols=2, nrows=2)
        # gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
        mean, std = get_imagenet_mean_std()

        for i, x in zip(range(4), [x1, x2, x3, x4]):
            train_utils.unnormalize_img(x[j], mean, std)

            # ax = plt.subplot(gs1[i])
            plt.subplot(2, 2, i + 1)
            plt.imshow(x[j].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))

        # plt.title("Is match" + str(y_true[j].cpu().numpy().item()))

        # print(fp0[j])
        # print(fp1[j])
        # print()

        title = f"Pred class {pred_label_idx}, GT Class (is_match?): {true_label_idx}"
        title += f"w/ prob {probs[j, pred_label_idx].cpu().numpy():.2f}"

        plt.suptitle(title)
        fig.tight_layout()

        building_id = Path(fp0[j]).parent.stem

        check_mkdir(vis_save_dir)
        plt.savefig(f"{vis_save_dir}/building_{building_id}__{Path(fp0[j]).stem}.jpg", dpi=500)

        # plt.show()
        plt.close("all")


def check_mkdir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)


def evaluate_model(
    serialization_save_dir: str, ckpt_fpath: str, args: TrainingConfig, split: str, save_viz: bool
) -> None:
    """ """
    cudnn.benchmark = True

    data_loader = train_utils.get_dataloader(args, split=split)
    model = train_utils.get_model(args)
    model = train_utils.load_model_checkpoint(ckpt_fpath, model, args)

    model.eval()

    with torch.no_grad():
        metrics_dict = run_test_epoch(serialization_save_dir, ckpt_fpath, model, data_loader, split, save_viz)


def plot_metrics(json_fpath: str) -> None:
    """ """
    json_data = json_utils.read_json_file(json_fpath)

    fig = plt.figure(dpi=200, facecolor="white")
    plt.style.use("ggplot")
    sns.set_style({"font_famiily": "Times New Roman"})

    num_rows = 1
    metrics = ["avg_loss", "mAcc"]
    num_cols = 2

    color_dict = {"train": "r", "val": "g"}

    for i, metric_name in enumerate(metrics):

        subplot_id = int(f"{num_rows}{num_cols}{i+1}")
        fig.add_subplot(subplot_id)

        for split in ["train", "val"]:
            color = color_dict[split]

            metric_vals = json_data[f"{split}_{metric_name}"]
            plt.plot(range(len(metric_vals)), metric_vals, color, label=split)

        plt.ylabel(metric_name)
        plt.xlabel("epoch")

    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, required=True, help="GPU device IDs to use for training.")
    opts = parser.parse_args()

    print("Using GPUs ", opts.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    # ResNet-18, floor and ceiling, RGB-only
    # model_results_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"
    # model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"

    # ResNet-50, floor and ceiling, RGB-only
    # model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26"
    # config_fpath = "afp/configs/2021_07_15_resnet50_ceiling_floor_rgbonly_test_set_inference.yaml"
    # serialization_save_dir = "2021_07_15_serialized_edge_classifications_v2"

    # ResNet-50, floor and ceiling, RGB-only, more tours (GT WDO)
    # model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_07_24/2021_07_26_14_42_49"
    # config_fpath = "afp/configs/2021_07_24_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"
    # serialization_save_dir = "2021_07_28_serialized_edge_classifications"

    # # ResNet-50, floor and ceiling, RGB only, 186 tours, but low-res (predicted WDO)
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20"
    # config_fpath = "afp/configs/2021_10_22_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"
    # serialization_save_dir = "/data/johnlam/2021_10_22___ResNet50_186tours_serialized_edge_classifications_test2021_11_02"

    # # ResNet-50, floor and ceiling, RGB only, 373 tours, but (predicted WDO), low-res
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_25_14_31_23"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_25_14_31_23/2021_10_22_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"
    # serialization_save_dir = "/data/johnlam/2021_10_26__ResNet50_373tours_serialized_edge_classifications_test2021_11_02"

    # # # # ResNet-152, floor and ceiling, RGB only, 435 tours, (predicted WDO), low-res
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16/2021_10_26_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"
    # # serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"
    # serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16"

    # # ResNet-152, ceiling only, RGB only, 587 tours, (predicted-WDO), low-res
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06/2021_11_04_resnet152_ceilingonly_rgbonly_no_photometric_augment.yaml"
    # # # serialization_save_dir = (
    # # #     "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_08"
    # # # )
    # # # serialization_save_dir = (
    # # #     "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_12"
    # # # )
    # serialization_save_dir = (
    #     "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"
    # )

    # # # ResNet-152, floor only, RGB only, 587 tours, (predicted-WDO), low-res
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48/2021_11_04_resnet152_flooronly_rgbonly_no_photometric_augment.yaml"
    # #serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_09" # tested at 93.9 acc.
    # #serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_11" # tested at 94.7 acc.
    # serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"

    # ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (FAILED AT 10 epochs)
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07/2021_11_10_resnet152_layoutonly.yaml"
    # serialization_save_dir = (
    #     "/data/johnlam/2021_11_10__ResNet152layoutonly__877tours_serialized_edge_classifications_test2021_11_15"
    # )

    # # ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (TRAINED 15/50 epochs so far, will go to completion0
    # model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10"
    # config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10/2021_11_10_resnet152_layoutonly.yaml"
    # # serialization_save_dir = (
    # #     "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test2021_11_15"
    # # )
    # serialization_save_dir = (
    #     "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test109buildings_2021_11_16"
    # )

    # # # ResNet-152, floor and ceiling, RGB only, 587 tours, (predicted WDO), low-res, with equal data now as floor-only or ceiling only
    model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11"
    config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11/2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.yaml"
    serialization_save_dir = "/data/johnlam/2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23"

    # model_results_dir should have only these 3 files within it
    # config_fpath = glob.glob(f"{model_results_dir}/*.yaml")[0]
    ckpt_fpath = glob.glob(f"{model_results_dir}/*.pth")[0]
    train_results_fpath = glob.glob(f"{model_results_dir}/*.json")[0]

    # plot_metrics(json_fpath=train_results_fpath)

    config_name = Path(config_fpath).name

    with hydra.initialize_config_module(config_module="afp.configs"):
        # config is relative to the afp module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # # use single-GPU for inference?
    # args.dataparallel = False

    args.batch_size = 64  # 128
    args.workers = 10

    split = "test"
    save_viz = True
    evaluate_model(serialization_save_dir, ckpt_fpath, args, split, save_viz)

    train_results_json = json_utils.read_json_file(train_results_fpath)
    val_mAccs = train_results_json["val_mAcc"]
    print("Val accs: ", val_mAccs)
    print("Num epochs trained", len(val_mAccs))
    print("Max val mAcc", max(val_mAccs))
