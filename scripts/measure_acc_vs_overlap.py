"""
Measure how the amount of visual overlap affects the trained verifier model's accuracy.
"""

import glob
import os
from pathlib import Path

import argoverse.utils.json_utils as json_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from argoverse.utils.sim2 import Sim2

import afp.common.edge_classification as edge_classification
import afp.common.posegraph2d as posegraph2d
import afp.utils.iou_utils as iou_utils
import afp.utils.rotation_utils as rotation_utils
from afp.common.edge_classification import EdgeClassification


def measure_acc_vs_visual_overlap(
    serialized_preds_json_dir: str, hypotheses_save_root: str, raw_dataset_dir: str
) -> None:
    """
    Count separately for negative and positive examples.
    """

    # fig = plt.figure(dpi=200, facecolor='white')
    plt.style.use("ggplot")
    sns.set_style({"font.family": "Times New Roman"})

    tuples = []
    gt_floor_pg_dict = {}

    # maybe interesting to also check histograms at different confidence thresholds
    confidence_threshold = 0.0

    gt_class = 0
    classname_str = "positives_only" if gt_class == 1 else "negatives_only"

    json_fpaths = glob.glob(f"{serialized_preds_json_dir}/batch*.json")
    # import random
    # random.shuffle(json_fpaths)

    for json_idx, json_fpath in enumerate(json_fpaths):
        print(f"On {json_idx}/{len(json_fpaths)}")

        # if json_idx > 30:
        #     continue

        json_data = json_utils.read_json_file(json_fpath)
        y_hat_list = json_data["y_hat"]
        y_true_list = json_data["y_true"]
        y_hat_prob_list = json_data["y_hat_probs"]
        fp0_list = json_data["fp0"]
        fp1_list = json_data["fp1"]

        

        # for each GT positive
        for y_hat, y_true, y_hat_prob, fp0, fp1 in zip(y_hat_list, y_true_list, y_hat_prob_list, fp0_list, fp1_list):

            if y_true != gt_class:
                continue

            # if y_true != 0:
            #     continue


            if not (
                fp0 == '/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_221___opening_0_0_identity_floor_rgb_floor_01_partial_room_04_pano_74.jpg' \
                and fp1 =='/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_221___opening_0_0_identity_floor_rgb_floor_01_partial_room_10_pano_11.jpg'
            ):
                continue
        
            import pdb;pdb.set_trace()

            if y_hat_prob < confidence_threshold:
                continue

            if "/data/johnlam" in fp0:
                fp0 = fp0.replace("/data/johnlam", "/home/johnlam")
                fp1 = fp1.replace("/data/johnlam", "/home/johnlam")

            f1 = imageio.imread(fp0)
            f2 = imageio.imread(fp1)
            floor_iou = iou_utils.texture_map_iou(f1, f2)

            i1 = int(Path(fp0).stem.split("_")[-1])
            i2 = int(Path(fp1).stem.split("_")[-1])

            if i1 >= i2:
                temp_i2 = i1
                temp_i1 = i2

                i1 = temp_i1
                i2 = temp_i2

            building_id = Path(fp0).parent.stem

            s = Path(fp0).stem.find("floor_0")
            e = Path(fp0).stem.find("_partial")
            floor_id = Path(fp0).stem[s:e]

            pair_idx = Path(fp0).stem.split("_")[1]

            is_identity = "identity" in Path(fp0).stem
            configuration = "identity" if is_identity else "rotated"

            # Rip out the WDO indices (`wdo_pair_uuid`), given `pair_3905___door_3_0_identity_floor_rgb_floor_01_partial_room_02_pano_38.jpg`
            k = Path(fp0).stem.split("___")[1].find(f"_{configuration}")
            assert k != -1
            wdo_pair_uuid = Path(fp0).stem.split("___")[1][:k]
            assert any([wdo_type in wdo_pair_uuid for wdo_type in ["door", "window", "opening"]])

            m = EdgeClassification(
                i1=i1,
                i2=i2,
                prob=y_hat_prob,
                y_hat=y_hat,
                y_true=y_true,
                pair_idx=pair_idx,
                wdo_pair_uuid=wdo_pair_uuid,
                configuration=configuration,
            )

            if (building_id, floor_id) not in gt_floor_pg_dict:
                gt_floor_pg_dict[(building_id, floor_id)] = posegraph2d.get_gt_pose_graph(
                    building_id, floor_id, raw_dataset_dir
                )

            gt_floor_pg = gt_floor_pg_dict[(building_id, floor_id)]

            wTi1_gt = gt_floor_pg.nodes[i1].global_Sim2_local
            wTi2_gt = gt_floor_pg.nodes[i2].global_Sim2_local
            i2Ti1_gt = wTi2_gt.inverse().compose(wTi1_gt)

            # # technically it is i2Si1, but scale will always be 1 with inferred WDO.
            i2Ti1 = edge_classification.get_alignment_hypothesis_for_measurement(
                m, hypotheses_save_root, building_id, floor_id
            )

            theta_deg_est = i2Ti1.theta_deg
            theta_deg_gt = i2Ti1_gt.theta_deg

            # need to wrap around at 360
            rot_err = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
            trans_err = np.linalg.norm(i2Ti1_gt.translation - i2Ti1.translation)
            tuples += [(floor_iou, y_hat, y_true, rot_err, trans_err)]

    bin_edges = np.linspace(0, 1, 11)
    counts = np.zeros(10)
    acc_bins = np.zeros(10)
    rot_err_bins = np.zeros(10)
    trans_err_bins = np.zeros(10)

    # running computation of the mean.
    for (iou, y_pred, y_true, rot_err, trans_err) in tuples:

        bin_idx = np.digitize(iou, bins=bin_edges)
        # digitize puts it into `bins[i-1] <= x < bins[i]` so we have to subtract 1
        acc = y_pred == y_true
        acc_bins[bin_idx - 1] += acc
        rot_err_bins[bin_idx - 1] += rot_err
        trans_err_bins[bin_idx - 1] += trans_err
        counts[bin_idx - 1] += 1

    counts = counts.astype(np.float32)
    mean_acc_bins = np.divide(acc_bins.astype(np.float32), counts) * 100

    avg_rot_err_bins = np.divide(rot_err_bins, counts)
    avg_trans_err_bins = np.divide(trans_err_bins, counts)

    def format_bar_chart() -> None:
        """ """
        xtick_labels = []
        for i in range(len(bin_edges) - 1):
            xtick_labels += [f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})"]
        plt.xticks(ticks=np.arange(10), labels=xtick_labels, rotation=20)
        plt.tight_layout()

    # bar chart.
    plt.bar(np.arange(10), mean_acc_bins)
    plt.xlabel("Floor-Floor Texture Map IoU")
    plt.ylabel("Mean Accuracy (%)")
    format_bar_chart()
    plt.savefig(
        f"{Path(serialized_preds_json_dir).stem}___bar_chart_iou_{classname_str}__confthresh{confidence_threshold}.pdf",
        dpi=500,
    )
    plt.close("all")
    # plt.savefig(f"{Path(serialized_preds_json_dir).stem}___bar_chart_iou_allexamples__confthresh{confidence_threshold}.jpg", dpi=500)

    plt.bar(np.arange(10), avg_rot_err_bins)
    plt.xlabel("Floor-Floor Texture Map IoU")
    plt.ylabel("Rotation Error (degrees)")
    format_bar_chart()
    plt.savefig(
        f"{Path(serialized_preds_json_dir).stem}___bar_chart_rot_error_iou_positives_only__confthresh{confidence_threshold}.pdf",
        dpi=500,
    )
    plt.close("all")

    plt.bar(np.arange(10), avg_trans_err_bins)
    plt.xlabel("Floor-Floor Texture Map IoU")
    plt.ylabel("Translation Error")
    format_bar_chart()
    plt.savefig(
        f"{Path(serialized_preds_json_dir).stem}___bar_chart_trans_error_iou_positives_only__confthresh{confidence_threshold}.pdf",
        dpi=500,
    )
    plt.close("all")

    # TODO: write a unit test for this function.


def test_measure_acc_vs_visual_overlap() -> None:
    """ """
    serialized_preds_json_dir = ""
    hypotheses_save_root = ""
    raw_dataset_dir = ""

    # first 3 entries from '/home/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02/batch_126.json'
    # Example 0 -- IoU 0.709, Rot Error 1.43 deg, Trans Error 0.01
    # Example 1 -- IoU 0.528, Rot Error 0.71 deg, Trans Error 0.055
    # Example 2 -- IoU 0.153, Rot Error 6.19 deg, Trans Error 0.2964
    batch126a_dict = {
        "y_hat": [1, 1, 1],
        "y_true": [1, 1, 1],
        "y_hat_probs": [0.9995661377906799, 0.9970742464065552, 0.6184548735618591],
        "fp0": [
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_54___door_0_0_identity_floor_rgb_floor_02_partial_room_01_pano_48.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_102___door_1_1_rotated_floor_rgb_floor_02_partial_room_02_pano_60.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_48___door_0_1_rotated_floor_rgb_floor_02_partial_room_03_pano_47.jpg",
        ],
        "fp1": [
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_54___door_0_0_identity_floor_rgb_floor_02_partial_room_01_pano_51.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_102___door_1_1_rotated_floor_rgb_floor_02_partial_room_08_pano_57.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_48___door_0_1_rotated_floor_rgb_floor_02_partial_room_06_pano_54.jpg",
        ],
    }

    # last 4 entries from batch 126
    # Example 0 -- IoU 0.381, Rot Error 2.325, Trans Error 0.057
    # Example 1 -- IoU 0.355, Rot Error 1.590, Trans Error 0.0665
    # Example 2 -- IoU 0.244, Rot Error 1.465, Trans Error 0.122
    # Example 3 -- IoU 0.422, Rot Error 2.7, Trans Error 0.141
    batch126b_dict = {
        "y_hat": [0, 1, 0, 1],
        "y_true": [1, 1, 1, 1],
        "y_hat_probs": [0.8956580758094788, 0.7106943726539612, 0.573124885559082, 0.7701843976974487],
        "fp0": [
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_95___opening_0_0_rotated_floor_rgb_floor_02_partial_room_05_pano_55.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_31___door_1_1_rotated_floor_rgb_floor_02_partial_room_02_pano_45.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_122___door_0_0_identity_floor_rgb_floor_02_partial_room_07_pano_65.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_46___door_0_1_rotated_floor_rgb_floor_02_partial_room_03_pano_47.jpg",
        ],
        "fp1": [
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_95___opening_0_0_rotated_floor_rgb_floor_02_partial_room_08_pano_58.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_31___door_1_1_rotated_floor_rgb_floor_02_partial_room_08_pano_57.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_122___door_0_0_identity_floor_rgb_floor_02_partial_room_07_pano_68.jpg",
            "/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0668/pair_46___door_0_1_rotated_floor_rgb_floor_02_partial_room_06_pano_52.jpg",
        ],
    }

    # '/home/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02/batch_2451.json'
    # choose just 2 pairs with known poses.
    # Example 0 -- IoU 0.817, Rot Error 1.214, Trans Error 0.621
    # Example 1 -- IoU
    batch2451_dict = {
        "y_hat": [1, 0],
        "y_true": [0, 0],
        "y_hat_probs": [0.8538389205932617, 0.9999430179595947],
        "fp0": [
            '/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_3412___opening_1_0_rotated_floor_rgb_floor_01_partial_room_01_pano_4.jpg',
            '/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_221___opening_0_0_identity_floor_rgb_floor_01_partial_room_04_pano_74.jpg'
        ],
        "fp1": [
            '/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_3412___opening_1_0_rotated_floor_rgb_floor_01_partial_room_10_pano_10.jpg',
            '/data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres/incorrect_alignment/1398/pair_221___opening_0_0_identity_floor_rgb_floor_01_partial_room_10_pano_11.jpg'
        ]
    }

    measure_acc_vs_visual_overlap(serialized_preds_json_dir, hypotheses_save_root, raw_dataset_dir)
    assert False


if __name__ == "__main__":

    serialized_preds_json_dir = (
        "/home/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"
    )
    # serialized_preds_json_dir = "/home/johnlam/2021_10_22___ResNet50_186tours_serialized_edge_classifications_test2021_11_02"
    # serialized_preds_json_dir = "/home/johnlam/2021_10_26__ResNet50_373tours_serialized_edge_classifications_test2021_11_02"
    raw_dataset_dir = "/home/johnlam/zind_bridgeapi_2021_10_05"
    hypotheses_save_root = (
        "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
    )
    measure_acc_vs_visual_overlap(serialized_preds_json_dir, hypotheses_save_root, raw_dataset_dir)
