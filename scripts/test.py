"""Run inference with a pretrained SALVe model."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import hydra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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
SALVE_CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "salve" / "configs"


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


@torch.no_grad()
def run_test_epoch(
    args: TrainingConfig,
    serialization_save_dir: str,
    ckpt_fpath: str,
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    split: str,
    save_viz: bool,
    serialize_predictions: bool = True,
) -> Dict[str, Any]:
    """Run all samples from test split through a pretrained network.

    Args:
        args: model/dataset/inference hyperparameters.
        serialization_save_dir: Directory where serialized predictions should be saved to.
        ckpt_fpath: Path to where trained pytorch model checkpoint file is saved.
        model: Pytorch SALVe model instance.
        data_loader: Data loader.
        split: ZInD data split to evalaute on.
        save_viz: Whether to salve visualizations of false positives to disk.
        serialize_predictions: Whether to save serialized predictions per alignment hypothesis.

    Returns:
        Dictionary of (key, value) pairs representing (metric name, metric value) summaries.
    """
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

        is_match_probs, loss = train_utils.cross_entropy_forward(model, split, x1, x2, x3, x4, x5, x6, gt_is_match)

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

    _, accs, _, avg_mAcc, _ = sam.get_metrics()
    prec, rec, mAcc = pr_meter.get_metrics()
    metrics_dict = {
        "split": split,
        "checkpoint_file_path": ckpt_fpath,
        "average_accuracy": avg_mAcc,
        "class_accuracies": accs,
        "precision": prec,
        "recall": rec,
        "mean_accuracy": mAcc,
    }
    return metrics_dict


def evaluate_model(
    serialization_save_dir: str, ckpt_fpath: str, args: TrainingConfig, split: str, save_viz: bool
) -> None:
    """Evaluate pretrained model on a dataset split of ZInD."""
    cudnn.benchmark = True

    data_loader = train_utils.get_dataloader(args, split=split)
    model = train_utils.get_model(args)
    model = train_utils.load_model_checkpoint(ckpt_fpath, model, args)

    model.eval()
    metrics_dict = run_test_epoch(
        args=args,
        serialization_save_dir=serialization_save_dir,
        ckpt_fpath=ckpt_fpath,
        model=model,
        data_loader=data_loader,
        split=split,
        save_viz=save_viz,
    )

    results_summary_fpath = f"{Path(ckpt_fpath).stem}.json"
    print(f"Saving {split} split accuracy summary to {results_summary_fpath}")
    io_utils.save_json_file(json_fpath=results_summary_fpath, data=metrics_dict)


@click.command(help="Script to run inference with pretrained SALVe model.")
@click.option(
    "--gpu_ids",
    type=str,
    required=True,
    help="String representing comma-separated list of GPU device IDs to use for training.",
)
@click.option(
    "--model_ckpt_fpath",
    type=click.Path(exists=True),
    required=True,
    help="Path to where trained pytorch model checkpoint file is saved.",
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
    help="Whether to use Pytorch's DataParallel for distributed inference (True) or single GPU (False).",
)
@click.option("--num_workers", type=int, default=10, help="Number of Pytorch dataloader workers.")
@click.option("--test_batch_size", type=int, default=64, help="Batch size to use during inference.")  # 128
@click.option(
    "--split",
    type=click.Choice(
        [
            "train",
            "val",
            "test",
        ]
    ),
    default="test",
    help="ZInD data split to evalaute on.",
)
def run_evaluate_model(
    gpu_ids: List[int],
    model_ckpt_fpath: str,
    config_name: str,
    serialization_save_dir: str,
    use_dataparallel: bool,
    num_workers: int,
    test_batch_size: int,
    split: str,
) -> None:
    """Click entry point for SALVe pretrained model inference."""

    print("Using GPUs ", gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if Path(model_ckpt_fpath).suffix != ".pth":
        raise FileNotFoundError("Model checkpoint file path should end in .pth.")

    if not Path(f"{SALVE_CONFIGS_ROOT}/{config_name}").exists():
        raise FileNotFoundError(f"No config found at `{SALVE_CONFIGS_ROOT}/{config_name}`.")

    with hydra.initialize_config_module(config_module="salve.configs"):
        # Note: config is relative to the `salve` module.
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # Apply command-line overrides to config.
    args.dataparallel = use_dataparallel
    args.batch_size = test_batch_size
    args.workers = num_workers

    save_viz = False
    evaluate_model(
        serialization_save_dir=serialization_save_dir,
        ckpt_fpath=model_ckpt_fpath,
        args=args,
        split=split,
        save_viz=save_viz,
    )


if __name__ == "__main__":

    run_evaluate_model()
