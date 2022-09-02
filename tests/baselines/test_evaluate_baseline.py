
""" """

import salve.baselines.evaluate_baseline as eval_utils


def test_count_panos_on_floor() -> None:
    """Ensure that the number of panoramas on a floor is counted correctly."""

    raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"
    building_id = "0982"

    num_floor0_panos = eval_utils.count_panos_on_floor(raw_dataset_dir, building_id, floor_id="floor_00")
    assert num_floor0_panos == 8

    num_floor1_panos = eval_utils.count_panos_on_floor(raw_dataset_dir, building_id, floor_id="floor_01")
    assert num_floor1_panos == 13
