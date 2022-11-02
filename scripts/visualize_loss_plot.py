"""Script to visualize loss & accuracy plots over time, as training progresses."""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns

import salve.utils.io as io_utils


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


@click.command(help="Script to visualize loss plot, given training logs.")
@click.option(
    "--train_results_fpath",
    type=click.Path(exists=True),
    required=True,
    help="Path to where JSON file containing training log is stored.",
)
def run_visualize_loss_plot(train_results_fpath: str):

    if not Path(train_results_fpath).suffix == ".json":
        raise ValueError("Must provide .json file.")

    train_results_json = io_utils.read_json_file(train_results_fpath)
    val_mAccs = train_results_json["val_mAcc"]
    print("Val accs: ", val_mAccs)
    print("Num epochs trained", len(val_mAccs))
    print("Max val mAcc", max(val_mAccs))

    plot_metrics(json_fpath=train_results_fpath)


if __name__ == "__main__":

    run_visualize_loss_plot()
