from typing import List

import math
import matplotlib
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union

from stitching.transform import transform_xy_by_pose
from stitching.models.feature2d import Feature2dXy
from stitching.models.locations import ORIGIN_POSE, Pose, Point2d
from stitching.models.floor_map_object import FloorMapObject

matplotlib.use("macOSX")


TANGO_COLOR_PALETTE = [
    [252, 233, 79],
    [237, 212, 0],
    [196, 160, 0],
    [252, 175, 62],
    [245, 121, 0],
    [206, 92, 0],
    [233, 185, 110],
    [193, 125, 17],
    [143, 89, 2],
    [138, 226, 52],
    [115, 210, 22],
    [78, 154, 6],
    [114, 159, 207],
    [52, 101, 164],
    [32, 74, 135],
    [173, 127, 168],
    [117, 80, 123],
    [92, 53, 102],
    [239, 41, 41],
    [204, 0, 0],
    [164, 0, 0],
    # [238, 238, 236],
    # [211, 215, 207],
    # [186, 189, 182],
    [136, 138, 133],
    [85, 87, 83],
    [46, 52, 54],
]


def draw_dwo_xy_top_down_canvas(axis, fig, filename, dwos_cluster_all):
    colors = {
        'door': 'red',
        'window': 'blue',
        'opening': 'green',
    }
    for panoid, dwos in dwos_cluster_all.items():
        for dwo in dwos:
            x = [dwo[0].x, dwo[1].x]
            y = [dwo[0].y, dwo[1].y]
            # import pdb; pdb.set_trace()
            axis.plot(x, y, color=colors[dwo[2]], linewidth=0.8)

    axis.set_aspect("equal")
    # fig.savefig(filename, dpi = 300)


def draw_shape_in_top_down_canvas(
        axis: SubplotBase, xys: List[Feature2dXy],
        color: str, confidences=None, pose: Pose = ORIGIN_POSE, linewidth=2):
    if pose:
        xys_transformed = []
        for xy in xys:
            xys_transformed.append(transform_xy_by_pose(xy, pose))
    else:
        xys_transformed = xys
    x = [xy.x for xy in xys_transformed]
    y = [xy.y for xy in xys_transformed]
    if confidences:
        for i in range(len(x)-2):
            confidence = confidences[i]
            if confidence is None:
                axis.plot(x[i:(i+2)], y[i:(i+2)], color='red', linewidth=linewidth)
            else:
                confidence = min(confidence, 0.1)
                confidence = max(confidence, 0.03)
                confidence = (0.1 - confidence) / 0.07
                distance = math.sqrt((x[i+1] - pose.position.x)**2 + (y[i+1] - pose.position.y)**2)
                # if distance < 5.0 / 3.2:
                #     axis.plot(x[i:(i+2)], y[i:(i+2)], color=color, linewidth=linewidth, alpha=confidence)
                # else:
                #     axis.plot(x[i:(i+2)], y[i:(i+2)], color='white', linewidth=linewidth)
                if True:
                    axis.plot(x[i:(i+2)], y[i:(i+2)], color=color, linewidth=linewidth, alpha=confidence)
                else:
                    axis.plot(x[i:(i+2)], y[i:(i+2)], color='white', linewidth=linewidth)
    else:
        for i in range(len(x)-1):
            axis.plot(x[i:(i+2)], y[i:(i+2)], color=color, linewidth=linewidth)
            # distance = math.sqrt((x[i+1] - pose.position.x)**2 + (y[i+1] - pose.position.y)**2)
            # if distance < 5.0 / 3.2:
            #     axis.plot(x[i:(i+2)], y[i:(i+2)], color=color, linewidth=linewidth)
            # else:
            #     axis.plot(x[i:(i+2)], y[i:(i+2)], color='white', linewidth=linewidth)


def draw_shape_in_top_down_canvas_fill(
        axis: SubplotBase, xys: List[Feature2dXy],
        color: str, pose: Pose = ORIGIN_POSE):
    if pose:
        xys_transformed = []
        for xy in xys:
            xys_transformed.append(transform_xy_by_pose(xy, pose))
    else:
        xys_transformed = xys
    x = [xy.x for xy in xys_transformed]
    y = [xy.y for xy in xys_transformed]
    axis.fill(x, y, color=color)


def draw_camera_in_top_down_canvas(axis: SubplotBase, pose: Pose, color: str, size: int=40):
    axis.scatter(pose.position.x, pose.position.y, s=size, facecolors="none", edgecolors=color, linewidths=1)


def draw_all_room_shapes_with_given_poses_and_shapes(filename, floor_map, panoid_refs, predictions, confidences, poses, groups):
    floor_map_obj = FloorMapObject(floor_map)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    for i_group, group in enumerate(groups):
        # color = hsv2rgb(i_group / len(groups), 1, 1)
        i_color = ((i_group) % 8) * 3 + int(i_group / 8)
        color = TANGO_COLOR_PALETTE[(i_color) % 24]
        # color = (color[0]/255, color[1]/255, color[2]/255)
        for panoid in group:
            room_shape = predictions[panoid]
            pose = poses[panoid]

            shape = [
                Point2d(x, room_shape.boundary.xy[1][i])
                for i, x in enumerate(room_shape.boundary.xy[0])
            ]
            shape.append(
                Point2d(room_shape.boundary.xy[0][0], room_shape.boundary.xy[1][0])
            )
            confidence = confidences[panoid] if confidences else None
            # draw_shape_in_top_down_canvas(axis, shape, "black", confidences=confidence, pose=pose, linewidth=0.5)
            # draw_camera_in_top_down_canvas(axis, pose, (color[0]/255, color[1]/255, color[2]/255), size=20)

            draw_shape_in_top_down_canvas(axis, shape, "black", confidences=confidence, pose=pose, linewidth=0.5)
            draw_camera_in_top_down_canvas(axis, pose, "blue", size=20)

    # axis.set_xlim([-1.1, 1.7])
    # axis.set_ylim([-1.5, 2.5])
    axis.set_aspect("equal")
    fig.savefig(filename)
    return [axis, fig]


def draw_all_room_shapes_with_poses(filename, floor_map, panoid_refs, arkit_points=[], axis=None):
    if not axis:
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)

    floor_map_obj = FloorMapObject(floor_map)
    shapes_union = []
    for panoid in panoid_refs:
        rsid = floor_map['panos'][panoid]['room_shape_id']
        room_shape = floor_map['room_shapes'][rsid]
        pose_ref = floor_map_obj.get_pano_global_pose(panoid)

        shape = [Point2d(v['x'], v['y']) for v in room_shape['vertices']]
        xys = shape
        xys_transformed = []
        for xy in xys:
            xys_transformed.append(transform_xy_by_pose(xy, pose_ref))
        shapes_union.append(Polygon([[xy.x, xy.y] for xy in xys_transformed]))

        shape.append(
            Point2d(room_shape['vertices'][0]['x'], room_shape['vertices'][0]['y'])
        )
        draw_shape_in_top_down_canvas(axis, shape, "black", pose=pose_ref, linewidth=0.5)
        draw_camera_in_top_down_canvas(axis, pose_ref, "black", size=10)

    for pt in arkit_points:
        pose = Pose(
            position=Point2d(x=pt[0].x, y=pt[0].y), rotation=0
        )
        draw_camera_in_top_down_canvas(axis, pose, "blue", size=10)

    # axis.set_xlim([-1.1, 1.7])
    # axis.set_ylim([-1.5, 2.5])
    axis.set_aspect("equal")
    if filename:
        fig.savefig(filename)

    return cascaded_union(shapes_union)
