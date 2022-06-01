
"""Make bar charts showing SALVe results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



results_dict = {
    "OpenSfM": {
        "loc_pct_mean": 27.62,
        "loc_pct_med": 22.22,
        "avg_rot_err_mean": 9.52,
        "avg_rot_err_med": 0.36,
        "avg_trans_err_mean": 1.88,
        "avg_trans_err_med": 0.12,
        "floorplan_iou_mean": 0.29,
        "floorplan_iou_med": 0.26
    },
    "OpenMVG": {
        "loc_pct_mean":13.94,
        "loc_pct_med": 8.70,
        "avg_rot_err_mean": 3.84,
        "avg_rot_err_med": 0.37,
        "avg_trans_err_mean": 0.41,
        "avg_trans_err_med": 0.10,
        "floorplan_iou_mean": 0.16,
        "floorplan_iou_med": 0.07
    },
    "Ours (ST + AA)": {
        "loc_pct_mean": 60.70,
        "loc_pct_med": 57.10,
        "avg_rot_err_mean": 3.69,
        "avg_rot_err_med": 0.03,
        "avg_trans_err_mean": 0.81,
        "avg_trans_err_med": 0.26,
        "floorplan_iou_mean": 0.56,
        "floorplan_iou_med": 0.52,
    },
    "Ours (PGO + AA)": {
        "loc_pct_mean": 60.70,
        "loc_pct_med": 57.10,
        "avg_rot_err_mean": 3.73,
        "avg_rot_err_med": 0.17,
        "avg_trans_err_mean": 0.80,
        "avg_trans_err_med": 0.25,
        "floorplan_iou_mean": 0.56,
        "floorplan_iou_med": 0.53
    },
    "Oracle (GT WDO + GT Layout)": {
        "loc_pct_mean": 88.58,
        "loc_pct_med": 93.44,
        "avg_rot_err_mean": 5.02,
        "avg_rot_err_med": 0.21,
        "avg_trans_err_mean": 0.98,
        "avg_trans_err_med": 0.22,
        "floorplan_iou_mean": 0.78,
        "floorplan_iou_med": 0.86
    }
}


def main() -> None:
    """ """
    #sns.set_style({'font.family': 'monospace'}) 

    b_names = []
    loc_pct_mean = []
    loc_pct_med = []
    avg_rot_err_mean = []
    avg_rot_err_med = []
    avg_trans_err_mean = []
    avg_trans_err_med = []

    floorplan_iou_mean = []
    floorplan_iou_med = []

    fig, ax = plt.subplots()
    # for each baseline
    for b_name, b_results in results_dict.items():
        loc_pct_mean.append(b_results["loc_pct_mean"])
        loc_pct_med.append(b_results["loc_pct_med"])
        avg_rot_err_mean.append(b_results["avg_rot_err_mean"])
        avg_rot_err_med.append(b_results["avg_rot_err_med"])
        avg_trans_err_mean.append(b_results["avg_trans_err_mean"])
        avg_trans_err_med.append(b_results["avg_trans_err_med"])

        floorplan_iou_mean.append(b_results["floorplan_iou_mean"])
        floorplan_iou_med.append(b_results["floorplan_iou_med"])
        b_names.append(b_name)

    x = np.arange(len(b_names))  # the label locations

    """
    width = 0.8/2.5  # the width of the bars
    ax.bar(x=x - width, width=width, height=loc_pct_mean,label=r"Mean")
    ax.bar(x=x, width=width, height=loc_pct_med,label=r"Median")
    plt.ylabel(r"% Panoramas Localized")
    ax.set_xticks(x - width/2)
    """

    """
    width = 0.8/2.5  # the width of the bars
    ax.bar(x=x - width, width=width, height=avg_rot_err_mean,label=r"Mean")
    ax.bar(x=x, width=width, height=avg_rot_err_med,label=r"Median")
    ax.set_xticks(x - width/2)
    plt.ylabel(r"Avg. Rotation Error per Pano (deg.)")
    """

    """
    width = 0.8/2.5  # the width of the bars
    ax.bar(x=x - width, width=width, height=avg_trans_err_mean,label=r"Mean")
    ax.bar(x=x, width=width, height=avg_trans_err_med,label=r"Median")
    ax.set_xticks(x - width/2)
    plt.ylabel(r"Avg. Translation Error per Pano (meters.)")
    """

    width = 0.8/2.5  # the width of the bars
    ax.bar(x=x - width, width=width, height=floorplan_iou_mean,label=r"Mean")
    ax.bar(x=x, width=width, height=floorplan_iou_med,label=r"Median")
    ax.set_xticks(x - width/2)
    plt.ylabel(r"Average Floorplan IoU")

    ha = 'right' 
    ax.set_xticklabels(b_names, rotation=45, ha=ha)
    mpl.style.use('ggplot')

    ax.legend(
        #loc='lower center', 
        #loc='center left', 
        loc='upper center',
        # bbox_to_anchor=(1, 0.5)
        bbox_to_anchor=(0.5, 1.3)
        # bbox_to_anchor=(0.5, 1.0), 
        # shadow=True, 
        # ncol=2, 
        #prop={'size': 8}
    )
    fig.tight_layout(pad=2)
    plt.show()




if __name__ == "__main__":
    main()