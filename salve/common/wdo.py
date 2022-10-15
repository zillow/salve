"""Data structure that parameterizes a window, door, or opening in 3D."""

import copy
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from salve.common.sim2 import Sim2


@dataclass(frozen=False)
class WDO:
    """Data structure that defines either a single door, single window, or single opening.

    Note: we define windows/doors/openings by their left and right boundaries.

    Attributes:
        global_Sim2_local: pose of W/D/O in ...TODO, as Similarity(2) transformation.
        pt1: start vertex of W/D/O.
        pt2: end vertex of W/D/O.
        bottom_z: z-coordinate of W/D/O base.
        top_z: z-coordinate of W/D/O top.
        type: category.
    """

    global_Sim2_local: Sim2
    pt1: Tuple[float, float]  # (x1,y1)
    pt2: Tuple[float, float]  # (x2,y2)
    bottom_z: float
    top_z: float
    type: str

    @property
    def centroid(self) -> np.ndarray:
        """Compute centroid of W/D/O 2d line segment."""
        return np.array([self.pt1, self.pt2]).mean(axis=0)

    @property
    def width(self) -> float:
        """Determine the width of the W/D/O.

        We define this as length of the line segment from start to end vertices.
        """
        return np.linalg.norm(np.array(self.pt1) - np.array(self.pt2))

    @property
    def vertices_local_2d(self) -> np.ndarray:
        """Returns 2d vertices in local coordinate frame."""
        return np.array([self.pt1, self.pt2])

    # TODO: come up with better name for vertices in BEV, vs. all 4 vertices
    @property
    def vertices_global_2d(self) -> np.ndarray:
        """Returns 2d vertices in global coordinate frame."""
        return self.global_Sim2_local.transform_from(self.vertices_local_2d)

    @property
    def vertices_local_3d(self) -> np.ndarray:
        """Returns 3d vertices in local coordinate frame."""
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return np.array([[x1, y1, self.bottom_z], [x2, y2, self.top_z]])

    @property
    def vertices_global_3d(self) -> np.ndarray:
        """Returns 3d vertices in global coordinate frame."""
        return self.global_Sim2_local.transform_from(self.vertices_local_3d)

    def get_wd_normal_2d(self) -> np.ndarray:
        """Returns 2-vector describing normal to line segment (rotate CCW from vector linking pt1->pt2)."""
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        vx = x2 - x1
        vy = y2 - y1
        n = np.array([-vy, vx])
        # normalize to unit length
        return n / np.linalg.norm(n)

    @property
    def polygon_vertices_local_3d(self) -> np.ndarray:
        """Return 3d vertices of W/D/O's polygon.

        Note: first vertex is repeated as last vertex
        """
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return np.array(
            [
                [x1, y1, self.bottom_z],
                [x1, y1, self.top_z],
                [x2, y2, self.top_z],
                [x2, y2, self.bottom_z],
                [x1, y1, self.bottom_z],
            ]
        )

    @classmethod
    def from_object_array(cls, wdo_data: Any, global_Sim2_local: Sim2, type: str) -> "WDO":
        """Create W/D/O object from .... TODO

        Args:
            wdo_data: array of shape (3,2)
            global_Sim2_local
            type: type of WDO, e.g.
        """
        pt1 = wdo_data[0].tolist()
        pt2 = wdo_data[1].tolist()
        bottom_z, top_z = wdo_data[2]
        # Multiply x-coordinate of ZInD W/D/O vertices by -1 to convert to right-handed World cartesian system.
        # Accounts for reflection over y-axis.
        pt1[0] *= -1
        pt2[0] *= -1
        return cls(global_Sim2_local=global_Sim2_local, pt1=pt1, pt2=pt2, bottom_z=bottom_z, top_z=top_z, type=type)

    def get_rotated_version(self) -> "WDO":
        """Rotate W/D/O by 180 degrees, as if seen from other side of doorway."""
        self_rotated = WDO(
            global_Sim2_local=self.global_Sim2_local,
            pt1=self.pt2,
            pt2=self.pt1,
            bottom_z=self.bottom_z,
            top_z=self.top_z,
            type=self.type,
        )

        return self_rotated

    def transform_from(self, i2Ti1: Sim2) -> "WDO":
        """If this W/D/O is in i1's frame, this will transfer the W/D/O into i2's frame."""
        pt1_ = tuple(i2Ti1.transform_from(np.array(self.pt1).reshape(1, 2)).squeeze().tolist())
        pt2_ = tuple(i2Ti1.transform_from(np.array(self.pt2).reshape(1, 2)).squeeze().tolist())

        # global_Sim2_local represented wTi1, so wTi1 * i1Ti2 = wTi2
        i1Ti2 = i2Ti1.inverse()
        self_transformed = WDO(
            global_Sim2_local=self.global_Sim2_local.compose(i1Ti2),  # TODO: update this as well by multiply with i1Ti2
            pt1=pt1_,
            pt2=pt2_,
            bottom_z=self.bottom_z,
            top_z=self.top_z,
            type=self.type,
        )
        return self_transformed

    def apply_Sim2(self, a_Sim2_b: Sim2, gt_scale: float) -> "WDO":
        """Convert the WDO's pose to a new global reference frame `a` for Sim(3) alignment.
        Previous was in global frame `b`.

        Consider this WDO to be the j'th W/D/O in some list/set.
        """
        aligned_self = copy.deepcopy(self)

        b_Sim2_j = self.global_Sim2_local
        a_Sim2_j = a_Sim2_b.compose(b_Sim2_j)
        # Equivalent of `transformFrom()` on Pose2 object.
        aligned_self.global_Sim2_local = Sim2(R=a_Sim2_j.rotation, t=a_Sim2_j.translation * a_Sim2_j.scale, s=gt_scale)
        return aligned_self
