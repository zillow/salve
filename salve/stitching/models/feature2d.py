""" TODO: ADD DOCSTRING """

from salve.stitching.transform import rotate_xys_clockwise, xy_to_depth, xy_to_u, xy_to_uv
from salve.stitching.models.locations import Point2d, Pose


class Feature2dU:
    # 2D feature can be represented in xy coordinates or uv coordinates.
    # When we have a prediction in xy space, for example room shape prediction,
    # or DWO prediction after ray casting on 2D room shape, a feature can be stored in
    # class Feature2dXy. However, when we have a DWO prediction in panorama space only,
    # without raycasting, we only have the u coordiante for DWO bounding box. In this case,
    # we use Feature2dU to represent the boundary positions of DWO prediction.
    def __init__(self, u: float, feature_type: str):
        self.u = u
        self.feature_type = feature_type


class Feature2dXy(Feature2dU):
    def __init__(self, u: float, feature_type: str, xy: Point2d, depth: float):
        super(Feature2dXy, self).__init__(u, feature_type)
        self.xy = xy
        self.depth = depth

    @staticmethod
    def fromPoint2d(coord: Point2d, feature_type: str) -> "Feature2dXy":
        return Feature2dXy(u=xy_to_u(coord), feature_type=feature_type, xy=coord, depth=xy_to_depth(coord))

    def _rotate_clockwise(self, rotation_deg: float) -> "Feature2dXy":
        xy_rotated = rotate_xys_clockwise([self.xy], rotation_deg)[0]
        return Feature2dXy.fromPoint2d(xy_rotated, self.feature_type)

    def _translate(self, translation_x: float, translation_y: float) -> "Feature2dXy":
        xy = Point2d(x=self.xy.x + translation_x, y=self.xy.y + translation_y)
        return Feature2dXy.fromPoint2d(xy, self.feature_type)

    def project_to_camera_cartesian_by_camera_pose(self, pose: Pose) -> "Feature2dXy":
        # Convert coordinate xy from world CS to camera CS
        # with input translation and rotation.
        # The transformation is in order: 1) Translation, 2) Rotation
        return self._translate(-pose.position.x, -pose.position.y)._rotate_clockwise(-pose.rotation)

    def apply_camera_pose_to_camera_cartesian(self, pose: Pose) -> "Feature2dXy":
        # Apply clockwise rotate and translate to coordinate xy from camera CS.
        # The transformation is in order: 1) Rotation, 2) Translation
        return self._rotate_clockwise(pose.rotation)._translate(pose.position.x, pose.position.y)

    def uv(self, height: float) -> Point2d:
        return xy_to_uv(self.xy, height)
