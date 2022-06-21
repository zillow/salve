""" TODO: ADD DOCSTRING """
from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np

# TODO: Use shapely Point2d instead.
class Point2d:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance(self, other: Point2d) -> float:
        """Return the distance between two Point2d objects."""
        if not isinstance(other, Point2d):
            raise ValueError("Both arguments to `distance()` must be Point2d objects.")
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Point3d:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class Pose:
    def __init__(self, position: Point2d, rotation: float) -> None:
        self.position = position
        self.rotation = rotation


ORIGIN_POSE = Pose(position=Point2d(x=0, y=0), rotation=0)



def test_point2d_distance() -> None:
    """Test 2d distance method using a 3-4-5 Pythagorean triangle."""
    p1 = Point2d(x=0, y=4)
    p2 = Point2d(x=3, y=0)
    assert np.isclose(p1.distance(p2), 5.0)