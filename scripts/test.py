"""Run inference with a pretrained SALVe model."""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import hydra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from hydra.utils import instantiate
from torch import nn

import salve.utils.io as io_utils
import salve.utils.normalization_utils as normalization_utils
import salve.utils.pr_utils as pr_utils
import salve.train_utils as train_utils
from salve.training_config import TrainingConfig
from salve.utils.avg_meter import SegmentationAverageMeter
from salve.utils.logger_utils import get_logger


logger = get_logger()
SALVE_CONFIGS_ROOT = Path(__file__).resolve().parent / "salve" / "configs"


def run_test_epoch(
    args: TrainingConfig,
    serialization_save_dir: str,
    ckpt_fpath: str,
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    split: str,
    save_viz: bool,
) -> Dict[str, Any]:
    """Run all samples from test split through a pretrained network."""
    pr_meter = PrecisionRecallMeter()
    sam = SegmentationAverageMeter()

    for i, test_example in enumerate(data_loader):

        # Assume cross entropy loss only currently.
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

        # Assume cross entropy loss only currently.
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
            model, split, x1, x2, x3, x4, x5, x6, gt_is_match
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

        # Check recall and precision.
        # Treat correctly aligned as a `positive`.
        prec, rec, mAcc = pr_meter.get_metrics()
        print(f"Iter {i}/{len(data_loader)} Prec {prec:.2f}, Rec {rec:.2f}, mAcc {mAcc:.2f}")

    metrics_dict = {}
    return metrics_dict


class PrecisionRecallMeter:
    """Data structure to compute precision, recall, and mean accuracy from streaming samples."""

    def __init__(self) -> None:
        """Initialize empty arrays for predictions & ground truth categories."""
        self.all_y_true = np.zeros((0, 1))
        self.all_y_hat = np.zeros((0, 1))

    def update(self, y_true: np.ndarray, y_hat: np.ndarray) -> None:
        """Append predictions & ground truth for new batch samples."""
        y_true = y_true.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)

        self.all_y_true = np.vstack([self.all_y_true, y_true])
        self.all_y_hat = np.vstack([self.all_y_hat, y_hat])

    def get_metrics(self) -> Tuple[float, float, float]:
        """Computes precision, recall, and mean accuracy from all samples seen thus far."""
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
    """Serialize model predictions for each alignment hypothesis. Each batch will be saved to its own JSON file.

    Args:
        serialization_save_dir: Directory where serialized predictions should be saved to.
        batch_idx: index of batch, among all batches (e.g. 3rd batch of 2000 batches).
        y_hat: tensor of shape (B,) representing model's predicted most likely class index for each example.
        y_true: tensor of shape (B,) representing ground truth class index for each example.
        probs:
        fp0: list of length (B,) representing file path to panorama 1 (of image pair).
        fp1: list of length (B,) representing file path to panorama 2 (of image pair).
    """
    n = y_hat.shape[0]
    save_dict = {
        "y_hat": y_hat.cpu().numpy().tolist(),
        "y_true": y_true.cpu().numpy().tolist(),
        "y_hat_probs": probs[torch.arange(n), y_hat].cpu().numpy().tolist(),
        "fp0": fp0,
        "fp1": fp1,
    }
    os.makedirs(serialization_save_dir, exist_ok=True)
    io_utils.save_json_file(json_fpath=f"{serialization_save_dir}/batch_{batch_idx}.json", data=save_dict)


def visualize_examples(
    ckpt_fpath: str,
    batch_idx: int,
    split: str,
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
        mean, std = normalization_utils.get_imagenet_mean_std()

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
    """Evaluate pretrained model on a dataset split of ZInD."""
    cudnn.benchmark = True

    data_loader = train_utils.get_dataloader(args, split=split)
    model = train_utils.get_model(args)
    model = train_utils.load_model_checkpoint(ckpt_fpath, model, args)

    model.eval()

    with torch.no_grad():
        metrics_dict = run_test_epoch(args, serialization_save_dir, ckpt_fpath, model, data_loader, split, save_viz)


def plot_metrics(json_fpath: str) -> None:
    """Plot train/val accuracy vs. epochs, from training job log (JSON file)."""
    json_data = io_utils.read_json_file(json_fpath)

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


@click.command(help="Script to run inference with pretrained SALVe model.")
@click.option(
    "--gpu_ids",
    type=List[int],
    required=True,
    help="List of GPU device IDs to use for training.",
)
@click.option(
    "--model_results_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory where trained model and training logs are saved.",
)
@click.option(
    "--config_name",
    type=str,
    required=True,
    help="File name of config file under `salve/configs/*` (not file path!). Should end in .yaml",
)
@click.option(
    "--serialization_save_dir",
    type=str,
    required=True,
    help="Directory where serialized predictions should be saved to.",
)
@click.option(
    "--use_dataparallel",
    type=bool,
    default=True,
    help=".",
)
def run_evaluate_model(
    gpu_ids: List[int], model_results_dir: str, config_name: str, serialization_save_dir: str, use_dataparallel: bool
) -> None:
    """Click entry point for SALVe pretrained model inference."""

    print("Using GPUs ", gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # model_results_dir should have only these 3 files within it
    # config_fpath = glob.glob(f"{model_results_dir}/*.yaml")[0]

    ckpt_fpaths = glob.glob(f"{model_results_dir}/*.pth")
    if len(ckpt_fpaths) != 1:
        raise FileNotFoundError("Model results dir was invalid (no checkpoint found).")

    ckpt_fpath = ckpt_fpaths[0]
    train_results_fpath = glob.glob(f"{model_results_dir}/*.json")[0]

    # plot_metrics(json_fpath=train_results_fpath)

    if not Path(f"{SALVE_CONFIGS_ROOT}/{config_name}").exists():
        raise FileNotFoundError("")

    with hydra.initialize_config_module(config_module="salve.configs"):
        # config is relative to the `salve` module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # # use single-GPU for inference?
    args.dataparallel = use_dataparallel

    args.batch_size = 64  # 128
    args.workers = 10

    split = "test"
    save_viz = False
    evaluate_model(
        serialization_save_dir=serialization_save_dir, ckpt_fpath=ckpt_fpath, args=args, split=split, save_viz=save_viz
    )

    train_results_json = io_utils.read_json_file(train_results_fpath)
    val_mAccs = train_results_json["val_mAcc"]
    print("Val accs: ", val_mAccs)
    print("Num epochs trained", len(val_mAccs))
    print("Max val mAcc", max(val_mAccs))


if __name__ == "__main__":

    run_evaluate_model()
