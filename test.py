
import argparse
import glob
import os
from typing import Any, Dict, Tuple

import hydra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from argoverse.utils.json_utils import read_json_file
from hydra.utils import instantiate
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from mseg_semantic.utils.avg_meter import SegmentationAverageMeter
from pathlib import Path
from torch import nn

from logger_utils import get_logger
from train_utils import (
    get_dataloader,
    get_model,
    cross_entropy_forward,
    unnormalize_img,
    load_model_checkpoint
)

logger = get_logger()


def run_test_epoch(ckpt_fpath: str, model: nn.Module, data_loader, split: str, save_viz: bool) -> Dict[str,Any]:
    """ """
    pr_meter = PrecisionRecallMeter()

    sam = SegmentationAverageMeter()

    for i, test_example in enumerate(data_loader):

        # assume cross entropy loss only currently
        x1, x2, x3, x4, is_match, fp0, fp1, fp2, fp3 = test_example

        n = x1.size(0)

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

        y_hat = torch.argmax(is_match_probs, dim=1)

        sam.update_metrics_cpu(
            pred=torch.argmax(is_match_probs, dim=1).cpu().numpy(),
            target=gt_is_match.squeeze().cpu().numpy(),
            num_classes=args.num_ce_classes,
        )

        pr_meter.update(
            y_true=gt_is_match.squeeze().cpu().numpy(),
            y_hat=torch.argmax(is_match_probs, dim=1).cpu().numpy()
        )

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
            fp0 = fp0,
            fp1 = fp1,
            fp2 = fp2,
            fp3 = fp3
        )

        _, accs, _, avg_mAcc, _ = sam.get_metrics()
        print(f"{split} result: mAcc{avg_mAcc:.4f}", "Cls Accuracies:", [ float(f"{acc:.2f}") for acc in accs ])

        # check recall and precision
        # treat correctly aligned as a `positive`
        prec, rec, mAcc = pr_meter.get_metrics()
        print(f"Prec {prec:.2f}, Rec {rec:.2f}, mAcc {mAcc:.2f}")

    metrics_dict = {}
    return metrics_dict


class PrecisionRecallMeter:

    def __init__(self):
        """ """
        self.all_y_true = np.zeros((0,1))
        self.all_y_hat = np.zeros((0,1))

    def update(self, y_true: np.ndarray, y_hat: np.ndarray) -> None:
        """ """
        y_true = y_true.reshape(-1,1)
        y_hat = y_hat.reshape(-1,1)

        self.all_y_true = np.vstack([self.all_y_true, y_true])
        self.all_y_hat = np.vstack([self.all_y_hat, y_hat])

    def get_metrics(self) -> Tuple[float,float,float]:
        """ """
        prec, rec, mAcc = compute_precision_recall(y_true=self.all_y_true, y_pred=self.all_y_hat)
        return prec, rec, mAcc


def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    """ Define 1 as the target class (positive)

    In confusion matrix, `actual` are along rows, `predicted` are columns:

              Predicted
          \\ P  N
    Actual P TP FN
           N FP TN
    """
    EPS = 1e-7

    TP = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    FP = np.logical_and(y_true != y_pred, y_pred == 1).sum()

    FN = np.logical_and(y_true != y_pred, y_pred == 0).sum()
    TN = np.logical_and(y_true == y_pred, y_pred == 0).sum()

    # form a confusion matrix
    C = np.zeros((2,2))
    C[0,0] = TP
    C[0,1] = FN

    C[1,0] = FP
    C[1,1] = TN

    # Normalize the confusion matrix
    C[0] /= (C[0].sum() + EPS)
    C[1] /= (C[1].sum() + EPS)

    mAcc = np.mean(np.diag(C))

    prec = TP / (TP + FP + EPS)
    rec = TP / (TP + FN + EPS)

    if (TP + FN) == 0:
        # there were no positive GT elements
        raise Warning("Recall undefined...")

    #import sklearn.metrics
    #prec, rec, _, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    return prec, rec, mAcc


def test_compute_precision_recall_1() -> None:
    """All incorrect predictions."""
    y_pred = np.array([0,0,1])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    assert prec == 0.0
    assert rec == 0.0
    assert mAcc == 0.0
    

def test_compute_precision_recall_2() -> None:
    """All correct predictions."""
    y_pred = np.array([1,1,0])
    y_true = np.array([1,1,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    assert np.isclose(prec, 1.0)
    assert np.isclose(rec, 1.0)
    assert np.isclose(mAcc, 1.0)
    

# def test_compute_precision_recall_3() -> None:
#     """All correct predictions."""
#     y_pred = np.array([0,0,0])
#     y_true = np.array([1,1,0])

#     import pdb; pdb.set_trace()
#     prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    
#     # no false positives, but low recall
#     assert np.isclose(prec, 1.0)
#     assert np.isclose(rec, 1/3)


def test_compute_precision_recall_4() -> None:
    """All correct predictions."""
    y_pred = np.array([0,1,0,1])
    y_true = np.array([1,1,0,0])

    prec, rec, mAcc = compute_precision_recall(y_true, y_pred)

    # no false positives, but low recall
    assert np.isclose(prec, 0.5)
    assert np.isclose(rec, 0.5)
    assert np.isclose(mAcc, 2.0)


def visualize_examples(ckpt_fpath: str, batch_idx: int, split: str, args, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, **kwargs) -> None:
    """ """
    vis_save_dir = f"{Path(ckpt_fpath).parent}/{split}_examples"

    y_hat = kwargs["y_hat"]
    y_true = kwargs["y_true"]
    probs = kwargs["probs"]

    fp0 = kwargs["fp0"]
    fp1 = kwargs["fp1"]
    fp2 = kwargs["fp2"]
    fp3 = kwargs["fp3"]
    
    n, _, h, w = x1.shape

    for j in range(n):

        mean, std = get_imagenet_mean_std()

        plt.figure(figsize=(10,5))
        gs1 = gridspec.GridSpec(ncols=4, nrows=4)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        mean, std = get_imagenet_mean_std()

        for i, x in zip(range(4), [x1,x2,x4,x4]):
            unnormalize_img(x[j], mean, std)

            ax = plt.subplot(gs1[i])
            ax.imshow(x[j].cpu().numpy().transpose(1,2,0).astype(np.uint8))

        #plt.title("Is match" + str(y_true[j].cpu().numpy().item()))

        print(fp0[j])
        print(fp1[j])
        print(fp2[j])
        print(fp3[j])
        print()

        pred_label_idx = y_hat[j].cpu().numpy().item()
        true_label_idx = y_true[j].cpu().numpy().item()
        title = f"Pred class {pred_label_idx}, GT Class (is_match?): {true_label_idx}"
        title += f"w/ prob {probs[j, pred_label_idx].cpu().numpy():.2f}"

        plt.suptitle(title)

        import pdb; pdb.set_trace()
        building_id = Path(fp0[j]).parent.stem

        check_mkdir(vis_save_dir)
        plt.savefig(f"{vis_save_dir}/building_{building_id}__{Path(fp0).stem}.jpg", dpi=500)

        #plt.show()
        plt.close("all")


def check_mkdir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)


def evaluate_model(ckpt_fpath: str, args, split: str, save_viz: bool) -> None:
    """ """
    cudnn.benchmark = True

    data_loader = get_dataloader(args, split=split)
    model = get_model(args)
    model = load_model_checkpoint(ckpt_fpath, model, args)

    model.eval()

    with torch.no_grad():
        metrics_dict = run_test_epoch(ckpt_fpath, model, data_loader, split, save_viz)


def plot_metrics(json_fpath: str) -> None:
    """ """
    json_data = read_json_file(json_fpath)

    fig = plt.figure(dpi=200, facecolor="white")
    plt.style.use('ggplot')
    sns.set_style({'font_famiily': 'Times New Roman'})

    num_rows = 1
    metrics = ['avg_loss', 'mAcc']
    num_cols = 2

    color_dict = {'train': 'r', 'val': 'g'}

    for i, metric_name in enumerate(metrics):

        subplot_id = int(f'{num_rows}{num_cols}{i+1}')
        fig.add_subplot(subplot_id)

        for split in ['train', 'val']:
            color = color_dict[split]

            metric_vals = json_data[f'{split}_{metric_name}']
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
    #model_results_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"
    #model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"

    # ResNet-50, floor and ceiling, RGB-only
    model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26"

    # model_results_dir should have only these 3 files within it
    config_fpath = glob.glob(f"{model_results_dir}/*.yaml")[0]
    ckpt_fpath = glob.glob(f"{model_results_dir}/*.pth")[0]
    train_results_fpath = glob.glob(f"{model_results_dir}/*.json")[0]

    #plot_metrics(json_fpath=train_results_fpath)

    config_name = Path(config_fpath).name

    with hydra.initialize_config_module(config_module="afp.configs"):
        # config is relative to the afp module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # # use single-GPU for inference?
    # args.dataparallel = False

    args.batch_size = 128
    args.workers = 16

    split = "val"
    save_viz = True
    evaluate_model(ckpt_fpath, args, split, save_viz)

    train_results_json = read_json_file(train_results_fpath)
    val_mAccs = train_results_json['val_mAcc']
    print("Val accs: ", val_mAccs)
    print("Num epochs trained", len(val_mAccs))
    print("Max val mAcc", max(val_mAccs))



    # test_compute_precision_recall_1()
    # test_compute_precision_recall_2()
    # #test_compute_precision_recall_3()
    # test_compute_precision_recall_4()
    # quit()


