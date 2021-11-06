from afp.common.floor_reconstruction_report import FloorReconstructionReport

"""
# aligned

On building 1210, floor_02
WDO Type Distribution:  defaultdict(<class 'float'>, {'door': 0.8918918918918912, 'window': 0.10810810810810811})
Max rot error: 6.2, Max trans error: 0.3
Mean relative rot. error: 1.6. trans. error: 0.1. Over  19 of 19 GT panos, estimated 37 edges
most confident was correct 1.00
	Class imbalance ratio 12.59
On average, over all tours, WDO types used were:
For door, 63.4%
For opening, 23.0%
For window, 19.3%
	Unfiltered Edge Acc = 1.00
Before any filtering, the largest CC contains 13 / 19 panos .
Localized 68.42% of panos: 13 / 19
[2021-11-06 00:02:06,694 INFO geometry_comparisons.py line 130 77982] Sim(3) Rotation `aRb`: rz=-0.15 deg., ry=0.00 deg., rx=0.00 deg.
[2021-11-06 00:02:06,694 INFO geometry_comparisons.py line 131 77982] Sim(3) Translation `atb`: [tx,ty,tz]=[ 0.02 -0.    0.  ]
[2021-11-06 00:02:06,694 INFO geometry_comparisons.py line 132 77982] Sim(3) Scale `asb`: 1.07
[2021-11-06 00:02:06,694 INFO geometry_comparisons.py line 140 77982] Pose graph Sim(3) alignment complete.
Mean translation error: 0.1, Mean rotation error: 1.3
	Avg translation error: 0.11



# unaligned

On building 1210, floor_02
WDO Type Distribution:  defaultdict(<class 'float'>, {'door': 0.8918918918918912, 'window': 0.10810810810810811})
Max rot error: 6.2, Max trans error: 0.3
Mean relative rot. error: 1.6. trans. error: 0.1. Over  19 of 19 GT panos, estimated 37 edges
most confident was correct 1.00
	Class imbalance ratio 12.59
On average, over all tours, WDO types used were:
For door, 63.4%
For opening, 23.0%
For window, 19.3%
	Unfiltered Edge Acc = 1.00
Before any filtering, the largest CC contains 13 / 19 panos .
Localized 68.42% of panos: 13 / 19
[2021-11-05 23:48:12,717 INFO geometry_comparisons.py line 130 77617] Sim(3) Rotation `aRb`: rz=-0.15 deg., ry=0.00 deg., rx=0.00 deg.
[2021-11-05 23:48:12,717 INFO geometry_comparisons.py line 131 77617] Sim(3) Translation `atb`: [tx,ty,tz]=[ 0.02 -0.    0.  ]
[2021-11-05 23:48:12,717 INFO geometry_comparisons.py line 132 77617] Sim(3) Scale `asb`: 1.07
[2021-11-05 23:48:12,718 INFO geometry_comparisons.py line 140 77617] Pose graph Sim(3) alignment complete.
Mean translation error: 0.1, Mean rotation error: 1.3
	Avg translation error: 0.05

"""


def test_from_wSi_list() -> None:
    """Ensure we can correctly create a FloorReconstructionReport from pose graph optimization output for Building 1210, Floor 02 of ZinD."""

    wSi_list = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Sim2(
            R=np.array([[1.0000000e00, 1.4511669e-13], [-1.4511669e-13, 1.0000000e00]], dtype=np.float32),
            t=np.array([3.1663807e-13, 4.0534674e-13], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9963625, 0.08521605], [-0.08521605, -0.9963625]], dtype=np.float32),
            t=np.array([-0.05208764, -0.657844], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.8538526, 0.5205148], [-0.5205148, -0.8538526]], dtype=np.float32),
            t=np.array([0.77260476, -1.6241723], dtype=np.float32),
            s=1.0,
        ),
        None,
        Sim2(
            R=np.array([[0.007844, -0.99996924], [0.99996924, 0.007844]], dtype=np.float32),
            t=np.array([-0.743632, 0.03829836], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.8644665, -0.50269043], [0.50269043, -0.8644665]], dtype=np.float32),
            t=np.array([-1.3128754, -0.0555355], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9977786, -0.06661703], [0.06661703, -0.9977786]], dtype=np.float32),
            t=np.array([-2.2001665, -1.263223], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.9995646, -0.02950616], [0.02950616, -0.9995646]], dtype=np.float32),
            t=np.array([-0.79566294, -0.76166594], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.00257046, -0.9999967], [0.9999967, -0.00257046]], dtype=np.float32),
            t=np.array([-0.6911983, 0.80846286], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[0.00632679, -0.99998], [0.99998, 0.00632679]], dtype=np.float32),
            t=np.array([-1.3925239, 0.91490793], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.01266379, -0.99991983], [0.99991983, -0.01266379]], dtype=np.float32),
            t=np.array([-2.4355152, 1.7160583], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.01020425, -0.9999479], [0.9999479, -0.01020425]], dtype=np.float32),
            t=np.array([-2.3332891, 0.30607823], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[-0.10058811, 0.9949282], [-0.9949282, -0.10058811]], dtype=np.float32),
            t=np.array([-1.3064604, 2.2962294], dtype=np.float32),
            s=1.0,
        ),
        Sim2(
            R=np.array([[0.02900542, 0.99957925], [-0.99957925, 0.02900542]], dtype=np.float32),
            t=np.array([-0.8010526, 2.38649], dtype=np.float32),
            s=1.0,
        ),
        None,
        None,
        None,
        None,
        None,
    ]

    gt_floor_pg = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
    plot_save_dir = "dummy_dir"

    report = FloorReconstructionReport.from_wSi_list(wSi_list, gt_floor_pg, plot_save_dir=plot_save_dir)

    assert np.isclose(avg_abs_rot_err, 9999)
    assert np.isclose(avg_abs_trans_err, 9999)
    assert np.isclose(percent_panos_localized, 9999)
    assert False

