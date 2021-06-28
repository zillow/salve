


import glob
from typing import Any, Dict

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from argoverse.utils.json_utils import read_json_file
from hydra.utils import instantiate
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from mseg_semantic.utils.avg_meter import SegmentationAverageMeter
from pathlib import Path
from torch import nn

from afp.training_config import TrainingConfig
from train_utils import (
    get_dataloader,
    get_model,
    cross_entropy_forward,
    unnormalize_img
)


def load_model_checkpoint(ckpt_fpath: str, model: nn.Module, args: TrainingConfig) -> nn.Module:
    """ """
    if not Path(ckpt_fpath).exists():
        raise RuntimeError(f"=> no checkpoint found at {ckpt_fpath}")

    map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=> loading checkpoint '{ckpt_fpath}'")
    checkpoint = torch.load(ckpt_fpath, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"=> loaded checkpoiint '{ckpt_fpath}'")

    return model

def run_test_epoch(model: nn.Module, data_loader, split: str, save_viz: bool) -> Dict[str,Any]:
    """ """

    sam = SegmentationAverageMeter()

    for i, test_example in enumerate(data_loader):

        # assume cross entropy loss only currently
        x1, x2, x3, x4, is_match, fp0, fp1, fp2, fp3 = example

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

        visualize_examples(
            batch_idx=i,
            split=split,
            args=args,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            y_hat=y_hat,
            y_true=gt_is_match,
            probs=probs,
            fp0 = fp0,
            fp1 = fp1,
            fp2 = fp2,
            fp3 = fp3
        )

        _, accs, _, avg_mAcc, _ = sam.get_metrics()
        print(f"{split} result: mAcc{avg_mAcc:.4f}", "Cls Accuracies:", [ float(f"{acc:.2f}") for acc in accs ])

        metrics_dict = {}

        return metrics_dict


def visualize_examples(batch_idx: int, split: str, args, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, **kwargs) -> None:
    """ """
    y_hat = kwargs["y_hat"]
    y_true = kwargs["y_true"]
    probs = kwargs["probs"]

    fp0 = kwargs["fp0"]
    fp1 = kwargs["fp1"]
    fp2 = kwargs["fp2"]
    fp3 = kwargs["fp3"]

    n, _, h, w = x.shape

    for j in range(n):

        mean, std = get_imagenet_mean_std()

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

        pred_label_idx = y_hat[j].cpu().numpy().item()
        true_label_idx = y_true[j].cpu().numpy().item()
        title = f"Pred class {pred_label_idx}, GT Class: {true_label_idx}"
        title += f"w/ prob {probs[j, pred_label_idx].cpu().numpy():.2f}"

        plt.suptitle("title")

        check_mkdir(vis_save_dir)
        plt.savefig(f"{vis_save_dir}/batch{batch_idx}_example{j}.jpg")
        plt.close("all")

        plt.show()



def evaluate_model(ckpt_fpath: str, args, split: str, save_viz: bool) -> None:
    """ """
    cudnn.benchmark = True
    
    data_loader = get_dataloader(args, split=split)
    model = get_model(args)
    model = load_model_checkpoint(ckpt_fpath, model, args)

    with torch.no_grad():
        metrics_dict = run_test_epoch(model, data_loader, split, save_viz)



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

    import pdb; pdb.set_trace()

    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":

    model_results_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"

    # model_results_dir should have only these 3 files within it
    config_fpath = glob.glob(f"{model_results_dir}/*.yaml")[0]
    ckpt_fpath = glob.glob(f"{model_results_dir}/*.pth")[0]
    train_results_fpath = glob.glob(f"{model_results_dir}/*.json")[0]

    #plot_metrics(json_fpath=train_results_fpath)

    config_name = Path(config_fpath).name

    with hydra.initialize_config_module(config_module="afp.configs"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    # always use single-GPU for inference
    args.dataparallel = False

    split = "val"
    save_viz = True
    evaluate_model(ckpt_fpath, args, split, save_viz)

    train_results_json['val_mAcc'] = read_json_file(train_results_fpath)
    print("Num epochs trained", len(val_mAccs))
    print("Max val mAcc", max(val_mAccs))

