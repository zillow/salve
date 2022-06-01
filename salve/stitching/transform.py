""" TODO: ADD DOCSTRING """

import math
from typing import List, Union

import numpy as np
from scipy import interpolate
from shapely.geometry import LineString, Point, Polygon

from salve.stitching.models.locations import Point2d, Point3d, Pose


def rotate_xys_clockwise(xys: List[Point2d], rotation_deg: float) -> List[Point2d]:
    """
    Rotate a list of xy coordinates around origin clockwise, by rotation_deg.
    Note that FMA room shape CS is clockwise.
    @param xys: list of xy coordinates
    @param rotation_deg: degree
    @return: list of rotated xy coordinates
    """
    xys_array = np.array([[xy.x, xy.y] for xy in xys])
    rotation_rad = math.radians(rotation_deg)
    rot_matrix = np.array(
        [[math.cos(-rotation_rad), -math.sin(-rotation_rad)], [math.sin(-rotation_rad), math.cos(-rotation_rad)]]
    )
    xys_transformed_array = rot_matrix.dot(xys_array.transpose()).transpose()
    xys_transformed = [Point2d(x=xy[0], y=xy[1]) for xy in xys_transformed_array]
    return xys_transformed


def uv_to_xyz(uv: Point2d) -> Point3d:
    """
    Compute coordinate xyz from texture coordinate uv, assuming length of
    vector xyz is 1. Z axis points up. Horizontal rotation goes clockwise.
    @param uv:
    @return xyz:
    """
    theta = math.pi - uv.y * math.pi
    # u = 0 is the left edge of panorama canvas. u = 0 happens when atan(x, y) = pi.
    phi = ((uv.x + 0.5) % 1.0) * math.pi * 2.0
    x = math.sin(theta) * math.sin(phi)
    y = math.sin(theta) * math.cos(phi)
    z = -math.cos(theta)
    return Point3d(x=x, y=y, z=z)


def u_to_xy(u: float) -> Point2d:
    # Compute xy coordinate from texture coordinate u, assuming length of
    # vector xy is 1.
    # u = 0 is the left edge of panorama canvas. u = 0 happens when atan(x, y) = pi.
    phi = ((u + 0.5) % 1.0) * math.pi * 2.0
    return Point2d(x=math.sin(phi), y=math.cos(phi))


def uv_to_xy(uv: Point2d, height: float) -> Point2d:
    # Compute coordinate xy from texture coordinate uv, given camera height.
    xyz = uv_to_xyz(uv)
    scale = -height / xyz.z
    x = xyz.x * scale
    y = xyz.y * scale
    return Point2d(x=x, y=y)


def uv_to_xy_batch(uvs, height: float):
    uvs_arr = np.array(uvs)
    theta_arr = (math.pi - uvs_arr[:, 1] * math.pi).reshape(len(uvs), 1)
    phi_arr = (((uvs_arr[:, 0] + 0.5) % 1.0) * math.pi * 2.0).reshape(len(uvs), 1)
    x_arr = np.sin(theta_arr) * np.sin(phi_arr)
    y_arr = np.sin(theta_arr) * np.cos(phi_arr)
    z_arr = -np.cos(theta_arr)
    scale_arr = -height / z_arr
    x_normalized_arr = x_arr * scale_arr
    y_normalized_arr = y_arr * scale_arr
    xys = [[x_normalized_arr[i, 0], y_normalized_arr[i, 0]] for i in range(len(uvs))]
    return xys


def xy_to_uv(xy: Point2d, height: float) -> Point2d:
    # Compute texture coordinate uv from Cartesian coordinate xy,
    # given camera height.
    u = xy_to_u(xy)
    depth = np.linalg.norm((xy.x, xy.y))
    v = 1.0 - math.atan(depth / height) / math.pi
    return Point2d(x=u, y=v)


def transform_xy_by_pose(xy: Point2d, pose: Pose) -> Point2d:
    # Rotate clockwise around origin [0, 0] and translate by pose.position.
    rot_rad = math.radians(-pose.rotation)
    x_rot = xy.x * math.cos(rot_rad) - xy.y * math.sin(rot_rad)
    y_rot = xy.x * math.sin(rot_rad) + xy.y * math.cos(rot_rad)
    x_final = x_rot + pose.position.x
    y_final = y_rot + pose.position.y
    return Point2d(x=x_final, y=y_final)


def project_xy_by_pose(xy: Point2d, pose: Pose) -> Point2d:
    x_translated = xy.x - pose.position.x
    y_translated = xy.y - pose.position.y
    rot_rad = math.radians(pose.rotation)
    x_final = x_translated * math.cos(rot_rad) - y_translated * math.sin(rot_rad)
    y_final = x_translated * math.sin(rot_rad) + y_translated * math.cos(rot_rad)
    return Point2d(x=x_final, y=y_final)


def xy_to_depth(xy: Point2d) -> float:
    """
    Compute a xy point depth from origin.
    @param xy: Input coordinate
    @return: depth number value
    """
    x, y = xy.x, xy.y
    return math.sqrt(x * x + y * y)


def xy_to_u(xy: Point2d) -> float:
    # Compute coordinate u from Cartesian coordinate xy.
    # u = 0 is the left edge of panorama canvas. u = 0 happens when atan(x, y) = pi.
    return (math.atan2(xy.x, xy.y) / math.pi + 1.0) / 2.0


def ray_cast_by_u(u: float, shape: Polygon) -> Union[Point2d, None]:
    """
    Ray cast from origin of the shape CS. The orientation of the ray is defined by u.
    Returns the closest intersection of the ray.
    @param u: U coordinate to define the ray orientation
    @param shape: Shapely polygon.
    @return: If no hit, return None. If there's a hit, return an xy coordinate.
    """
    xy = u_to_xy(u)
    vector_from = Point([0, 0])
    vector_to = Point([xy.x * 10000, xy.y * 10000])
    vector_line = LineString([vector_from, vector_to])

    xys_shape = shape.boundary.xy
    intersects = []
    for i in range(len(xys_shape[0]) - 1):
        pt = None
        edge_from = [xys_shape[0][i], xys_shape[1][i]]
        edge_to = [xys_shape[0][i + 1], xys_shape[1][i + 1]]
        edge_line = LineString([edge_from, edge_to])

        pt = line_segment_intersection(vector_line, edge_line)
        if pt:
            intersects.append([pt.distance(vector_from), pt])
    if intersects:
        i_closest_intersect = min(enumerate(intersects), key=lambda x: x[1][0])[0]
        intersection = intersects[i_closest_intersect][1]
        return Point2d(x=intersection.x, y=intersection.y)
    else:
        return None


def is_point_between_line_endpoints(point: Point, line: LineString, buffer_size: float = 1e-4) -> bool:
    # Check if a point is between the 2 endpoints of the input line segment.
    # The checking includes a buffer zone to avoid floating error.
    return line.distance(point) < buffer_size


def line_segment_intersection(line1: LineString, line2: LineString, buffer_size: float = 1e-4) -> Point:
    # TODO: Refactor this function, to not use line_intersection_infinite. Refactor to
    # extend line segment with buffer, then compute line segment intersection.

    # Compute line segment intersection. This function generates intersection,
    # that considers a buffer extension of the line segment.
    intersect = line_intersection_infinite(line1, line2)
    if (
        intersect
        and is_point_between_line_endpoints(intersect, line1, buffer_size)
        and is_point_between_line_endpoints(intersect, line2, buffer_size)
    ):
        return intersect
    return None


def line_intersection_infinite(line1_orig: LineString, line2_orig: LineString) -> Point:
    # Extending the both walls
    MAX_POSSIBLE_LENGTH = 10000000
    p11 = Point(
        (MAX_POSSIBLE_LENGTH + 1) * line1_orig.coords[1][0] - MAX_POSSIBLE_LENGTH * line1_orig.coords[0][0],
        (MAX_POSSIBLE_LENGTH + 1) * line1_orig.coords[1][1] - MAX_POSSIBLE_LENGTH * line1_orig.coords[0][1],
    )
    p12 = Point(
        (MAX_POSSIBLE_LENGTH + 1) * line1_orig.coords[0][0] - MAX_POSSIBLE_LENGTH * line1_orig.coords[1][0],
        (MAX_POSSIBLE_LENGTH + 1) * line1_orig.coords[0][1] - MAX_POSSIBLE_LENGTH * line1_orig.coords[1][1],
    )
    line1 = LineString([p11, p12])

    p21 = Point(
        (MAX_POSSIBLE_LENGTH + 1) * line2_orig.coords[1][0] - MAX_POSSIBLE_LENGTH * line2_orig.coords[0][0],
        (MAX_POSSIBLE_LENGTH + 1) * line2_orig.coords[1][1] - MAX_POSSIBLE_LENGTH * line2_orig.coords[0][1],
    )
    p22 = Point(
        (MAX_POSSIBLE_LENGTH + 1) * line2_orig.coords[0][0] - MAX_POSSIBLE_LENGTH * line2_orig.coords[1][0],
        (MAX_POSSIBLE_LENGTH + 1) * line2_orig.coords[0][1] - MAX_POSSIBLE_LENGTH * line2_orig.coords[1][1],
    )
    line2 = LineString([p21, p22])
    intersect = line1.intersection(line2)
    if isinstance(intersect, Point):
        return intersect
    return None


def get_global_coords_2d_from_room_cs(pano_xy, x, y, rotation, scale=1):
    """
    Generate 2D coordinates (xz) of room vertices in global CS.
    Since y axis points up, we have parameter names xzs in this function.

    @param mat_global_from_floor - 3 x 3 numpy array of 2D homogeneous transformation matrix.
    @return xzs_global - [[x1, z1], [x2, z2], ...]
    """
    # TODO: Change docstring style to https://google.github.io/styleguide/pyguide.html#383-functions-and-methods
    mat_floor_from_room = generate_2d_tranformation_matrix_from_room_to_floor(x, y, rotation, scale)

    xzs = [[pano_xy[0], pano_xy[1]]]

    xzs_floor = transform_xz(mat_floor_from_room, xzs)
    # xzs_global = transform_xz(mat_global_from_floor, xzs_floor)
    return xzs_floor


def gen_homogeneous_transformation_matrix_for_2d(shift, rot_rad: float, scale: float):
    """
    Compute the transformation matrix of applying scale -> rotation -> translation
    on 2D homogeneous coordinate.

    @param shift: list of 2 elements. Tranlsation in xz direction.
    @return mat_transform_2d: 3 x 3 numpy array
    """

    mat_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    mat_rot = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0], [np.sin(rot_rad), np.cos(rot_rad), 0], [0, 0, 1]])
    mat_translate = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    mat_transform_2d = np.matmul(mat_translate, np.matmul(mat_rot, mat_scale))
    return mat_transform_2d


def transform_xz(mat_transform_2d: np.array, xzs):
    """
    Transform 2D coordinates using input transformation matrix mat_transform_2d.

    @param mat_transform_2d - 3 x 3 numpy array of 2D homogeneous transformation matrix.
    @param xzs - list of [x, y].
    @return xzs_tranformed - list of [x, y].
    """
    xz_array = np.ones((len(xzs), 3))
    for i, xz in enumerate(xzs):
        xz_array[i, 0] = xz[0]
        xz_array[i, 1] = xz[1]

    xz_transformed = np.matmul(xz_array, np.transpose(mat_transform_2d))

    xzs_final = [[xz[0], xz[1]] for xz in xz_transformed]
    return xzs_final


def generate_2d_tranformation_matrix_from_room_to_floor(x: float, y: float, rotation: float, scale: float = 1.0):
    """
    Generate homogeneous transformation matrix to convert 2D homogeneous
    coordinate from room_shape CS to floor_shape CS. Note that room shape is a
    left-handed CS, room shape transformation is in a right-handed CS. This function
    will have to revert the handedness.

    @return mat_floor_from_room - np.array 3 x 3.
    """
    mat_floor_from_room = gen_homogeneous_transformation_matrix_for_2d(
        [-x, y],
        np.deg2rad(-rotation),
        scale,
    )
    return mat_floor_from_room


def reproject_uvs_to(uvs1_projected, wall_conf1, panoid, start_id):
    RES = 512
    us_projected = [uv.x for uv in uvs1_projected]

    us_projected1 = us_projected.copy()
    us_projected1 = np.array(([0] + us_projected1)[:-1])
    us_projected = np.array(us_projected)
    direction = (us_projected - us_projected1) > 0
    start = 0
    changes = []
    for j in range(512):
        if direction[j] != direction[j + 1]:
            changes.append([start, j])
            start = j + 1
    if changes[-1][1] != 511:
        changes.append([start, 511])
    if direction[0] != direction[1]:
        changes = changes[1:]
        changes[0][0] = 0

    sections = [changes[0]]
    for j, change in enumerate(changes[1:]):
        if change[1] - change[0] < 2:
            # sections[-1][1] = change[1]
            continue
        else:
            sections.append(change)

    original_us = np.arange(0.5 / RES, (RES + 0.5) / RES, 1.0 / RES)
    final_vs = np.zeros(RES)
    final_cs = np.zeros(RES)
    for i_section, section in enumerate(sections):
        us = [uv.x for uv in uvs1_projected[section[0] : section[1] + 1]]
        vs = [uv.y for uv in uvs1_projected[section[0] : section[1] + 1]]
        confs = wall_conf1[section[0] : section[1] + 1]

        fv = interpolate.interp1d(us, vs)
        fc = interpolate.interp1d(us, confs)

        is_polarized = False
        if min(us) < 0.1 and max(us) > 0.9:
            us_low = [uv.x for uv in uvs1_projected[section[0] : section[1] + 1] if uv.x < 0.5]
            us_high = [uv.x for uv in uvs1_projected[section[0] : section[1] + 1] if uv.x > 0.5]
            if min(us_high) - max(us_low) > 0.1:
                is_polarized = True

        if not is_polarized:
            start_u_idx = math.ceil((min(us) - 0.5 / RES) / (1 / RES))
            end_u_idx = math.floor((max(us) - 0.5 / RES) / (1 / RES))
            ranges = [[start_u_idx, end_u_idx]]
        else:
            start_u_idx = math.ceil((min(us) - 0.5 / RES) / (1 / RES))
            end_u_idx = math.floor((max(us) - 0.5 / RES) / (1 / RES))
            ranges = [[0, start_u_idx], [end_u_idx, RES - 1]]

        for [start_u_idx, end_u_idx] in ranges:
            us_new = original_us[start_u_idx : (end_u_idx + 1)]
            try:
                new_vs = fv(us_new)
                new_cs = fc(us_new)
            except Exception:
                # import pdb; pdb.set_trace()
                continue
            does_update = (
                (final_vs[start_u_idx : (end_u_idx + 1)] == 0) + (new_vs > final_vs[start_u_idx : (end_u_idx + 1)])
            ).astype(float)
            final_vs[start_u_idx : (end_u_idx + 1)] = (does_update) * new_vs + (1 - does_update) * final_vs[
                start_u_idx : (end_u_idx + 1)
            ]
            final_cs[start_u_idx : (end_u_idx + 1)] = (does_update) * new_cs + (1 - does_update) * final_cs[
                start_u_idx : (end_u_idx + 1)
            ]
    return final_vs, final_cs


def ray_cast_and_generate_dwo_xy(dwo_pred, shape):
    xy_from = ray_cast_by_u(dwo_pred[0], shape)
    xy_to = ray_cast_by_u(dwo_pred[1], shape)
    return [xy_from, xy_to]
