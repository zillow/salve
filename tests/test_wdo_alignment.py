"""Unit tests on W/D/O relative pose alignment hypotheses generation."""

import numpy as np

import salve.utils.wdo_alignment as wdo_alignment_utils
from salve.common.pano_data import PanoData, WDO
from salve.common.sim2 import Sim2
from salve.utils.wdo_alignment import AlignTransformType


def test_align_rooms_by_wd() -> None:
    """Test the generation relative pose alignment hypotheses between two panoramas using their W/D/O's.

    Panos are numbered as panos 5 and 8, and the only type of W/D/O's provided are window detections.
    """
    wTi5 = Sim2(
        R=np.array([[0.999897, -0.01435102], [0.01435102, 0.999897]], dtype=np.float32),
        t=np.array([0.7860708, -1.57248], dtype=np.float32),
        s=0.4042260417272217,
    )
    wTi8 = Sim2(
        R=np.array([[0.02998102, -0.99955046], [0.99955046, 0.02998102]], dtype=np.float32),
        t=np.array([0.91035557, -3.2141], dtype=np.float32),
        s=0.4042260417272217,
    )

    # fmt: off
    pano1_obj = PanoData(
        id=5,
        global_Sim2_local=wTi5,
        room_vertices_local_2d=np.array(
            [
                [ 1.46363621, -2.43808616],
                [ 1.3643741 ,  0.5424695 ],
                [ 0.73380685,  0.52146958],
                [ 0.7149462 ,  1.08780075],
                [ 0.4670652 ,  1.07954551],
                [ 0.46914653,  1.01704912],
                [-1.2252865 ,  0.96061904],
                [-1.10924507, -2.5237714 ]
            ]),
        image_path='panos/floor_01_partial_room_05_pano_5.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_Sim2_local=wTi5,
                pt1=[-1.0367953294361147, -2.5213585867749635],
                pt2=[-0.4661345615720372, -2.5023537435761822],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_Sim2_local=wTi5,
                pt1=[0.823799786466513, -2.45939477144822],
                pt2=[1.404932996095547, -2.4400411621788427],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )


    pano2_obj = PanoData(
        id=8,
        global_Sim2_local=wTi8,
        room_vertices_local_2d=np.array(
            [
                [-0.7336625 , -1.3968136 ],
                [ 2.23956454, -1.16554334],
                [ 2.19063694, -0.53652654],
                [ 2.75557561, -0.4925832 ],
                [ 2.73634178, -0.2453117 ],
                [ 2.67399906, -0.25016098],
                [ 2.54252291,  1.44010577],
                [-0.93330008,  1.16974146]
            ]),
        image_path='panos/floor_01_partial_room_05_pano_8.jpg',
        label='living room',
        doors=[],
        windows=[
            WDO(
                global_Sim2_local=wTi8,
                pt1=[-0.9276784906829552, 1.0974698581331057],
                pt2=[-0.8833992085857922, 0.5282122352406332],
                bottom_z=-0.5746298535133153,
                top_z=0.38684337323286566,
                type='windows'
            ),
            WDO(
                global_Sim2_local=wTi8,
                pt1=[-0.7833093301499523, -0.758550412558342],
                pt2=[-0.7382174598580689, -1.338254727497497],
                bottom_z=-0.5885416433689703,
                top_z=0.3591070365687572,
                type='windows'
            )
        ],
        openings=[]
    )

    # We assume these are "GT" W/D/O detections, meaning we are not using inferred W/D/Os and layout.
    # (See `use_inferred_wdos_layout` flag below).

    # fmt: on
    possible_alignment_info, num_invalid_configurations = wdo_alignment_utils.align_rooms_by_wd(
        pano1_obj,
        pano2_obj,
        transform_type=AlignTransformType.SE2,
        use_inferred_wdos_layout=False,
        visualize=False,
        verbose=True,
    )
    # Of 4 possible alignment hypotheses, only 2 satisfy freespace constraints (all satisfy relative width constraints).
    assert len(possible_alignment_info) == 2


if __name__ == "__main__":
    test_align_rooms_by_wd()
