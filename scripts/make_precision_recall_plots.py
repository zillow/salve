
"""Make precision-recall curves for different models.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import salve.common.edge_classification as edge_classification
import salve.utils.pr_utils as pr_utils


# # ResNet-152 for all 3 models here.
model_dict = {
    "ceiling + floor": "/Users/johnlam/Downloads/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02",
    "floor-only": "/Users/johnlam/Downloads/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_09",
    "ceiling-only": "/Users/johnlam/Downloads/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_12",
    "layout-only": "/Users/johnlam/Downloads/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test2021_11_15"
}
# ResNet-152 for all 4 models here.
# model_dict = {
#     "ceiling + floor": "/Users/johnlam/Downloads/2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23",
#     "floor-only": "/Users/johnlam/Downloads/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test109buildings_2021_11_16",
#     "ceiling-only": "/Users/johnlam/Downloads/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test109buildings_2021_11_16",
#     "layout-only": "/Users/johnlam/Downloads/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test109buildings_2021_11_16"
# }


def compare_precision_recall_across_models() -> None:
    """Make precision-recall curves on a single plot for different trained models."""
    plt.style.use("ggplot")
    #sns.set_style({"font.family": "Times New Roman"})
    
    palette = np.array(sns.color_palette("hls", 5))
    color_dict = {
        "ceiling + floor": palette[0], # "r",
        "floor-only": palette[1], # "b",
        "ceiling-only": palette[2], # "g"
        "layout-only": palette[3]
    }

    for model_name, serialized_preds_json_dir in model_dict.items():

        floor_edgeclassifications_dict = edge_classification.get_edge_classifications_from_serialized_preds(
            serialized_preds_json_dir
        )
        all_model_measurements = []
        for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

            # stack predictions together across tours
            all_model_measurements.extend(measurements)

        #plt.title(model_name)
        prec, recall, _ = pr_utils.plot_precision_recall_curve_sklearn(all_model_measurements)
        plt.plot(recall, prec, color=color_dict[model_name], label=model_name)
    
    plt.legend(fontsize='x-large')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig("2021_11_16b_precision_recall.pdf", dpi=500)
    plt.show()


def make_thresholds_plot() -> None:
    """"Compare how confidence thresholds affect performance."""
    plt.style.use("ggplot")
    #sns.set_style({"font.family": "Times New Roman"})

    # this is the current operating point -- 93% confidence and above.
    op_conf_thresh = 0.93

    palette = np.array(sns.color_palette("hls", 5))
    color_dict = {
        "ceiling + floor": palette[0], # "r",
        "floor-only": palette[1], # "b",
        "ceiling-only": palette[2], # "g"
        "layout-only": palette[3]
    }

    for model_name, serialized_preds_json_dir in model_dict.items():

        floor_edgeclassifications_dict = edge_classification.get_edge_classifications_from_serialized_preds(
            serialized_preds_json_dir
        )
        all_model_measurements = []
        for (building_id, floor_id), measurements in floor_edgeclassifications_dict.items():

            # stack predictions together across tours
            all_model_measurements.extend(measurements)

        #plt.title(model_name)
        prec, recall, thresholds = pr_utils.plot_precision_recall_curve_sklearn(all_model_measurements)

        n = thresholds.shape[0]
        plt.plot(thresholds[:n-1], prec[:n-1], color=color_dict[model_name], label=f"{model_name}: precision", linestyle="solid")
        plt.plot(thresholds[:n-1], recall[:n-1], color=color_dict[model_name], label=f"{model_name}: recall", linestyle="dashed")
    
    # show the operating point
    plt.plot([op_conf_thresh, op_conf_thresh], [0,1], color=palette[4])

    plt.xlabel("CNN probability for 'positive' class")
    plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig("2021_11_16b_precision_recall_vs_confthresholds_v3.pdf", dpi=500)
    plt.show()


if __name__ == "__main__":

    compare_precision_recall_across_models()
    make_thresholds_plot()
