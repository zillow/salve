from typing import Any, List, Optional

# TODO: Use shapely Point2d instead.
class Point2d:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Point3d:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Pose:
    def __init__(self, position: Point2d, rotation: float):
        self.position = position
        self.rotation = rotation

ORIGIN_POSE = Pose(position=Point2d(x=0, y=0), rotation=0)
