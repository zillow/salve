

"""
TODO: maybe only need one landmark for each WDO, since they will act as the same? locked down?
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from gtsfm.common.keypoints import Keypoints
#from gtsfm.common.sfm_track import SfmTrack2d

from afp.common.sfm_track import SfmTrack2d
from afp.common.edge_classification import EdgeClassification
from afp.common.pano_data import PanoData


def get_kpt_idx(wdo_idx: int, wdo_object_type: str, pano_data: PanoData) -> int:
    """Get index for the start vertex. Grouped by twos for (start vertex, end vertex).

    Args:
        wdo_idx: for this particular object type, index within the list.
    """
    num_openings = len(pano_data.openings)
    num_windows = len(pano_data.windows)
    num_doors = len(pano_data.doors)

    if wdo_object_type == "opening":
        kpt_idx = wdo_idx * 2

    elif wdo_object_type == "window":
        kpt_idx = (num_openings + wdo_idx) * 2

    elif wdo_object_type == "door":
        kpt_idx = (num_openings + num_windows + wdo_idx) * 2

    return kpt_idx


def test_get_kpt_idx() -> None:
    """For 1 opening and 3 windows, and no doors, get keypoint index.

    [s] - openings
    [e]
    [s] - windows
    [e]
    [s]
    [e]
    [s]
    [e]
    """
    from types import SimpleNamespace

    # fill with dummy objects
    pano_data = SimpleNamespace(
        openings=[None],
        windows=[None,None,None],
        doors=[]
    )
    wdo_idx = 1
    wdo_object_type = "window"
    kpt_idx = get_kpt_idx(wdo_idx, wdo_object_type, pano_data)
    assert kpt_idx == 4


def perform_data_association(measurements: List[EdgeClassification], pano_dict_inferred: Dict[int, PanoData]) -> List[SfmTrack2d]:
    """Perform union-find data association.

    Args:
        measurements:
        pano_dict_inferred:

    Returns:
        tracks_2d: feature/landmark tracks for landmark-based SLAM.
    """
    num_panos = max(pano_dict_inferred.keys()) + 1

    EMPTY_KEYPOINTS = Keypoints(coordinates=np.zeros((0,2)))
    # we can think of 2d landmarks as keypoints.
    keypoints_list = [ EMPTY_KEYPOINTS ] * num_panos
    for i, pano_data in pano_dict_inferred.items():
        keypoints = []
        # order as openings, windows, doors
        for obj_type in ["openings", "windows", "doors"]:
            wdos = getattr(pano_data, obj_type)
            for wdo in wdos:
                s, e = wdo.pt1, wdo.pt2
                keypoints.append(s)
                keypoints.append(e)

        # must be 2-dimensional array for SfmTrack's union-find to not reject it.
        keypoints_list[i] = Keypoints(coordinates=np.array(keypoints).reshape(-1,2)) 

    matches_dict = defaultdict(list)

    for m in measurements:
        i1, i2 = m.i1, m.i2
        alignment_object, i, j = m.wdo_pair_uuid.split("_")
        i, j = int(i), int(j)

        s_1 = get_kpt_idx(wdo_idx=i, wdo_object_type=alignment_object, pano_data=pano_dict_inferred[i1])
        s_2 = get_kpt_idx(wdo_idx=j, wdo_object_type=alignment_object, pano_data=pano_dict_inferred[i2])

        e_1 = s_1 + 1
        e_2 = s_2 + 1
        
        if m.configuration == "rotated":
            matches_dict[(i1, i2)] += [(s_1, s_2)]
            matches_dict[(i1, i2)] += [(e_1, e_2)]
        else:
            matches_dict[(i1, i2)] += [(s_1, e_2)]
            matches_dict[(i1, i2)] += [(e_1, s_2)]

        # i1_s, i1_e = get_ith_wdo_room_endpoints_from_pano(
        #     pano_data=pano_dict_inferred[m.i1], i=i, alignment_object=alignment_object, use_rotated=
        # )
        # i2_s, i2_e = get_ith_wdo_room_endpoints_from_pano(
        #     pano_data=pano_dict_inferred[m.i2], i=j, alignment_object=alignment_object, use_rotated=m.configuration == "rotated"
        # )

    matches_dict = {k: np.array(v) for k,v in matches_dict.items()}
    #tracks_2d = SlamFeatureTrack2d.generate_tracks_from_pairwise_matches(matches_dict)
    
    tracks_2d = SfmTrack2d.generate_tracks_from_pairwise_matches(matches_dict, keypoints_list)
    return tracks_2d


def get_ith_wdo_room_endpoints_from_pano(pano_data: PanoData, i: int, alignment_object: str, use_rotated: bool) -> Tuple[float,float]:
    """
    Args:
        pano_data: 
        i: 
        alignment_object: 
        use_rotated:

    Returns:
        s: start vertex of WDO (on 2d plane), in room coordinate system.
        e: end vertex of WDO (on 2d plane), in room coordinate system.
    """
    wdo_attr_key = alignment_object + "s"
    wdos = getattr(pano_data, wdo_attr_key)
    wdo = wdos[i]

    if use_rotated:
        wdo = wdo.get_rotated_version()

    s, e = wdo.pt1, wdo.pt2
    return s, e
