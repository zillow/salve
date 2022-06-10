""" TODO: ADD DOCSTRING """

import math
from copy import deepcopy
from typing import Any


def convert_floor_map_to_localization_cluster(floor_map_object: Any) -> Any:
    """TODO

    Args:
        floor_map_object: TODO

    Returns:
       clusters_all: TODO
    """
    clusters_all = []
    for fsid, floor_shape in floor_map_object.data["floor_shapes"].items():
        clusters = {}
        panoids = floor_map_object.get_panoids_with_floor_id(fsid)
        for panoid in panoids:
            pose = floor_map_object.get_pano_global_pose(panoid)
            item = {
                "pose": {
                    "rotation": pose.rotation,
                    "x": pose.position.x,
                    "y": pose.position.y,
                }
            }
            clusters[panoid] = item
        clusters_all.append(clusters)
    return clusters_all


def align_pred_poses_with_gt(floor_map_object: Any, cluster: Any) -> Any:
    """We align two pose graphs by SE(2), not Sim(2), setting first pose of each to the same pose??

    Args:
        floor_map_object: TODO
        cluster: TODO

    Returns:
        new_cluster:
    """
    cluster_gt = {}
    for panoid in cluster["panos"]:
        pose_gt = floor_map_object.get_pano_global_pose(panoid)
        if pose_gt:
            cluster_gt[panoid] = pose_gt

    new_cluster = deepcopy(cluster)
    new_cluster["panos"] = {}

    start_panoid = cluster["start_panoid"]
    pose_gt = cluster_gt[start_panoid]
    pose_pred = cluster["panos"][start_panoid]["pose"]
    translation1 = [-pose_pred["x"], -pose_pred["y"]]
    rotation2 = -(pose_gt.rotation - (pose_pred["rotation"])) * math.pi / 180
    translation3 = [pose_gt.position.x, pose_gt.position.y]
    print(translation1, rotation2, translation3)

    new_cluster["panos"] = {}
    for panoid_1 in cluster["panos"]:
        pose1 = cluster["panos"][panoid_1]["pose"]
        x1 = pose1["x"] + translation1[0]
        y1 = pose1["y"] + translation1[1]

        x2 = math.cos(rotation2) * x1 - math.sin(rotation2) * y1
        y2 = math.sin(rotation2) * x1 + math.cos(rotation2) * y1

        x3 = x2 + translation3[0]
        y3 = y2 + translation3[1]
        rotation3 = pose1["rotation"] + (pose_gt.rotation - (pose_pred["rotation"]))

        new_cluster["panos"][panoid_1] = {
            "pose": {
                "x": x3,
                "y": y3,
                "rotation": rotation3,
            }
        }

    return new_cluster


def test_align_pred_poses_with_gt() -> None:
    """ """
    floor_map_object = None
    cluster = None
    new_cluster = align_pred_poses_with_gt(floor_map_object=floor_map_object, cluster=cluster)
    assert False

