import numpy as np
from argoverse.utils.sim2 import Sim2
from gtsam import Rot3, Similarity3

import afp.common.posegraph2d as posegraph2d


def test_convert_Sim3_to_Sim2() -> None:
    """Ensure that Similarity(3) to Similarity(2) conversion works by projection (x,y,z) to (x,y)."""
    a_Sim3_b = Similarity3(
        R=Rot3(np.array([[0.999997, 0.00256117, 0], [-0.00256117, 0.999997, 0], [0, 0, 1]])),
        t=np.array([0.02309136, -0.00173048, 0.0]),
        s=1.0653604360576439,
    )

    a_Sim2_b = posegraph2d.convert_Sim3_to_Sim2(a_Sim3_b)

    expected_aRb = np.array([[0.999997, 0.00256117], [-0.00256117, 0.999997]], dtype=np.float32)
    expected_atb = np.array([0.02309136, -0.00173048], dtype=np.float32)
    assert np.allclose(a_Sim2_b.rotation, expected_aRb)
    assert np.allclose(a_Sim2_b.translation, expected_atb)
    assert np.isclose(a_Sim2_b.scale, 1.0653604360576439)


"""

(Pdb) p bTi_list_est[16]
R: [
	1, 1.45117e-13, 0;
	-1.45117e-13, 1, 0;
	0, 0, 1
]
t: 3.16638e-13 4.05347e-13           0



(Pdb) p aligned_bTi_list_est[16]
R: [
	0.999997, 0.00256117, 0;
	-0.00256117, 0.999997, 0;
	0, 0, 1
]
t:   0.0246006 -0.00184358           0





(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.rotation
array([[ 0.9999967 ,  0.00256117],
       [-0.00256117,  0.9999967 ]], dtype=float32)
(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.translation
array([ 0.02309136, -0.00173048], dtype=float32)
(Pdb) p aligned_est_pose_graph.nodes[16].global_Sim2_local.scale
1.0

"""
