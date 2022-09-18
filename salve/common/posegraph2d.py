"""Class to represent 2d pose graphs, render them, and compute error between two of them.

TODO: transformFrom() operation on Similarity(2) needs to be patched from current hacky solution.
Can use Pose2 instead of Similarity(2) from the get-go.
"""
from __future__ import annotations

import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import gtsfm.utils.geometry_comparisons as gtsfm_geometry_comparisons
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Point3, Pose3, Rot3, Similarity3
from scipy.spatial.transform import Rotation

import salve.utils.io as io_utils
import salve.utils.ransac as ransac
import salve.utils.rotation_utils as rotation_utils
from salve.common.pano_data import FloorData, PanoData
from salve.common.sim2 import Sim2


try:
    import salve.visualization.utils
except Exception as e:
    print("Open3D could not be loaded, skipping...")
    print("Exception: ", e)


REDTEXT = "\033[91m"
ENDCOLOR = "\033[0m"

# averaged over 1575 buildings and 2453 valid scales.
ZIND_AVERAGE_SCALE_METERS_PER_COORDINATE = 3.5083


class PoseGraph2d(NamedTuple):
    """Pose graph for a single floor.

    Note: edges are not included here, since there are different types of adjacency (spatial vs. visible)

    Attributes:
        building_id: unique ID for ZInD building.
        floor_id: unique ID for floor of a specific ZInD building.
        nodes: Mapping from panorama ID to PanoData, for each panorama. PanoData contains the pano's pose and optionally
            W/D/Os, layout, and room category.
        scale_meters_per_coordinate: scaling factor from "world normalized" to "world metric" coordinate system.
    """

    building_id: int
    floor_id: str
    nodes: Dict[int, PanoData]
    scale_meters_per_coordinate: float

    def pano_ids(self) -> List[int]:
        """Return a list of the panorama IDs encapsulated in this pose graph."""
        return list(self.nodes.keys())

    def __repr__(self) -> str:
        """ """
        n_nodes = len(self.nodes.keys())
        return f"Graph has {n_nodes} nodes in Building {self.building_id}, {self.floor_id}: {self.nodes.keys()}"

    def get_camera_height_m(self, pano_id: int) -> float:
        """Obtain the actual height of a RICOH Theta camera, from the saved ZinD dataset information.

        Args:
            pano_id:
            TODO: this is same for every pano on a floor, argument is unnecessary (can be removed)
               but programmatically verify this first.

        Returns:
            camera_height_m: camera height above the floor, in meters.
        """
        if pano_id not in self.nodes:
            print(f"Pano id {pano_id} not found among {self.nodes.keys()}")
            import pdb; pdb.set_trace()

        # from zillow_floor_map["scale_meters_per_coordinate"][floor_id]
        worldmetric_s_worldnormalized = self.scale_meters_per_coordinate

        # from pano_data["floor_plan_transformation"]["scale"]
        worldnormalized_s_egonormalized = self.nodes[pano_id].global_Sim2_local.scale

        worldmetric_s_egonormalized = worldmetric_s_worldnormalized * worldnormalized_s_egonormalized

        cam_height_egonormalized = 1.0
        camera_height_m = worldmetric_s_egonormalized * cam_height_egonormalized

        # print(f"Camera height (meters): {camera_height_m:.2f}")
        return camera_height_m

    @classmethod
    def from_floor_data(cls, building_id: str, fd: FloorData, scale_meters_per_coordinate: float) -> PoseGraph2d:
        """ """
        # print(f"scale_meters_per_coordinate: {scale_meters_per_coordinate:.2f}")
        return cls(
            building_id=building_id,
            floor_id=fd.floor_id,
            nodes={p.id: p for p in fd.panos},
            scale_meters_per_coordinate=scale_meters_per_coordinate,
        )

    @classmethod
    def from_wRi_list(cls, wRi_list: List[np.ndarray], building_id: str, floor_id: str) -> PoseGraph2d:
        """Generate a 2d pose graph from a list of 2x2 rotations.

        Fill other pano metadata with dummy values. Alternatively, could populate them from the GT pose graph.
        """
        nodes = {}
        for i, wRi in enumerate(wRi_list):
            if wRi is None:
                continue

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local=Sim2(R=wRi, t=np.zeros(2), s=1.0),
                room_vertices_local_2d=np.zeros((0, 2)),
                image_path="",
                label="",
                doors=None,
                windows=None,
                openings=None,
            )

        # just use the average scale over ZinD, when it is unknown.
        return cls(
            building_id=building_id,
            floor_id=floor_id,
            nodes=nodes,
            scale_meters_per_coordinate=ZIND_AVERAGE_SCALE_METERS_PER_COORDINATE,
        )

    @classmethod
    def from_wSi_list(cls, wSi_list: List[Optional[Sim2]], gt_floor_pose_graph: PoseGraph2d) -> PoseGraph2d:
        """Create a 2d pose graph, given a subset of global poses, and ground truth pose graph.

        GT pose graph input provides only GT W/D/O and GT Layout, not poses.
        """
        # TODO: try spanning tree version, vs. Shonan version
        wRi_list = [wSi.rotation if wSi else None for wSi in wSi_list]
        wti_list = [wSi.translation if wSi else None for wSi in wSi_list]

        return PoseGraph2d.from_wRi_wti_lists(wRi_list, wti_list, gt_floor_pose_graph)

    def as_3d_pose_graph(self) -> List[Optional[Pose3]]:
        """Return a version of the 2d pose graph that has trivially been lifted to 3d.

        Returns:
            wTi_list: list of length N, where N is the one greater than the largest index
                in the dictionary.
        """
        num_images = max(self.nodes.keys()) + 1
        wTi_list = [None] * num_images
        for i, pano_obj in self.nodes.items():

            wRi = pano_obj.global_Sim2_local.rotation
            wti = pano_obj.global_Sim2_local.translation
            wti = np.array([wti[0], wti[1], 0.0])
            wRi = rotation_utils.rot2x2_to_Rot3(wRi)

            wTi = Pose3(wRi, Point3(wti))

            wTi_list[i] = wTi

        return wTi_list

    @classmethod
    def from_wRi_wti_lists(
        cls, wRi_list: List[np.ndarray], wti_list: List[np.ndarray], gt_floor_pg: PoseGraph2d
    ) -> PoseGraph2d:
        """Generate 2d pose graph from global rotations and global translations of panos.

        Args:
            wRi_list: list of 2x2 matrices.
            wti_list: list of (2,) vectors.
            gt_floor_pg: ground truth floor pose graph. We ignore its poses, but
                scrape it for other pano metadata (image paths, building id, floor id, room vertices)
                to use to populate the new pose graph object.
        """
        building_id = gt_floor_pg.building_id
        floor_id = gt_floor_pg.floor_id

        nodes = {}
        for i, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
            if wRi is None or wti is None:
                continue

            # update the global pose associated with each WDO object
            global_Sim2_local = Sim2(R=wRi, t=wti, s=1.0)
            doors = copy.deepcopy(gt_floor_pg.nodes[i].doors)
            windows = copy.deepcopy(gt_floor_pg.nodes[i].windows)
            openings = copy.deepcopy(gt_floor_pg.nodes[i].openings)

            for door in doors:
                door.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            for window in windows:
                window.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            for opening in openings:
                opening.global_Sim2_local = copy.deepcopy(global_Sim2_local)

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local=global_Sim2_local,
                room_vertices_local_2d=gt_floor_pg.nodes[i].room_vertices_local_2d,
                image_path=gt_floor_pg.nodes[i].image_path,
                label=gt_floor_pg.nodes[i].label,
                doors=doors,
                windows=windows,
                openings=openings,
            )

        # When scale is unknown, use average value over ZInD.
        return cls(
            building_id=building_id,
            floor_id=floor_id,
            nodes=nodes,
            scale_meters_per_coordinate=ZIND_AVERAGE_SCALE_METERS_PER_COORDINATE,
        )

    @classmethod
    def from_aligned_est_poses_and_inferred_layouts(
        cls, aligned_est_floor_pose_graph: PoseGraph2d, inferred_floor_pose_graph: PoseGraph2d
    ) -> PoseGraph2d:
        """Combine estimated global poses with inferred room layouts.

        Args:
            aligned_est_floor_pose_graph: estimated global poses, aligned to GT for eval, with GT layouts.
            inferred_floor_pose_graph: oracle global poses, with inferred layouts.

        Returns:
            inferred_aligned_pg: estimated global poses with inferred room layouts.
        """
        nodes = {}
        # estimated pano data w/ pose info
        for i, epd in aligned_est_floor_pose_graph.nodes.items():

            # inferred panorama data with W/D/O detections + predicted layout.
            ipd = inferred_floor_pose_graph.nodes[i]

            nodes[i] = PanoData(
                id=i,
                global_Sim2_local=epd.global_Sim2_local,
                room_vertices_local_2d=ipd.room_vertices_local_2d,
                image_path=ipd.image_path,
                label=ipd.label,
                doors=ipd.doors,
                windows=ipd.windows,
                openings=ipd.openings,
            )

        print(f"\t\t\tScale is: {aligned_est_floor_pose_graph.scale_meters_per_coordinate:.1f}")
        inferred_aligned_pg = cls(
            building_id=aligned_est_floor_pose_graph.building_id,
            floor_id=aligned_est_floor_pose_graph.floor_id,
            nodes=nodes,
            scale_meters_per_coordinate=aligned_est_floor_pose_graph.scale_meters_per_coordinate,
        )
        return inferred_aligned_pg

    def as_json(self, json_fpath: str) -> None:
        """ """
        pass

    def measure_aligned_abs_pose_error(self, gt_floor_pg: PoseGraph2d) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Measure pose errors between two pose graphs that have already been aligned.

        Args:
            gt_floor_pg: ground truth pose graph.

        Returns:
            mean_rot_err: average rotation error per camera, measured in degrees.
            mean_trans_err: average translation error per camera.
            rot_errors: array of (K,) rotation errors, measured in degrees.
            trans_errors: array of (K,) translation errors.
        """
        aTi_list_gt = gt_floor_pg.as_3d_pose_graph()  # reference
        bTi_list_est = self.as_3d_pose_graph()

        mean_rot_err, mean_trans_err, rot_errors, trans_errors = ransac.compute_pose_errors_3d(
            aTi_list_gt, bTi_list_est
        )
        return mean_rot_err, mean_trans_err, rot_errors, trans_errors

    def measure_unaligned_abs_pose_error(
        self, gt_floor_pg: PoseGraph2d
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Measure the absolute pose errors (in both rotations and translations) for each localized pano.

        Applicable for pose graphs that have NOT been aligned yet.

        Args:
            gt_floor_pg:

        Returns:
            mean_rot_err: average rotation error per camera, measured in degrees.
            mean_trans_err: average translation error per camera.
            rot_errors: array of (K,) rotation errors, measured in degrees.
            trans_errors: array of (K,) translation errors.
        """

        # get the new aligned estimated pose graph
        aligned_est_pose_graph, aligned_bTi_list_est = self.align_by_Sim3_to_ref_pose_graph(ref_pose_graph=gt_floor_pg)

        aTi_list_gt = gt_floor_pg.as_3d_pose_graph()  # reference

        mean_rot_err, mean_trans_err, rot_errors, trans_errors = ransac.compute_pose_errors_3d(
            aTi_list_gt, aligned_bTi_list_est
        )
        return mean_rot_err, mean_trans_err, rot_errors, trans_errors

    def align_by_Sim3_to_ref_pose_graph(self, ref_pose_graph: PoseGraph2d) -> PoseGraph2d:
        """
        TODO: should it be a class method?
        """

        aTi_list_ref = ref_pose_graph.as_3d_pose_graph()  # reference
        bTi_list_est = self.as_3d_pose_graph()

        # if the estimate pose graph is missing a few nodes, pad it up to the GT list length
        pad_len = len(aTi_list_ref) - len(bTi_list_est)
        bTi_list_est.extend([None] * pad_len)

        # salve.visualization.utils.plot_3d_poses(aTi_list_gt=aTi_list_ref, bTi_list_est=bTi_list_est)

        # align the pose graphs
        aligned_bTi_list_est, aSb = ransac.ransac_align_poses_sim3_ignore_missing(aTi_list_ref, bTi_list_est)

        # salve.visualization.utils.plot_3d_poses(aTi_list_gt=aTi_list_ref, bTi_list_est=aligned_bTi_list_est)

        # TODO(johnwlambert): assumes all nodes have the same scale, which is not true.
        random_ref_pano_id = list(ref_pose_graph.nodes.keys())[0]
        gt_scale = ref_pose_graph.nodes[random_ref_pano_id].global_Sim2_local.scale
        aligned_est_pose_graph = self.apply_Sim3(a_Sim3_b=aSb, gt_scale=gt_scale)
        return aligned_est_pose_graph, aligned_bTi_list_est

    def apply_Sim3(self, a_Sim3_b: Similarity3, gt_scale: float) -> PoseGraph2d:
        """Create a new pose instance of the entire pose graph, after applying a Similarity(2) transform to every pose.

        The Similarity(2) transformation is computed by projecting a Similarity(3) transformation to 2d.
        """
        aligned_est_pose_graph = copy.deepcopy(self)

        a_Sim2_b = convert_Sim3_to_Sim2(a_Sim3_b)

        # for each pano, just update it's pose only
        for i in self.nodes.keys():

            pano_data_i = aligned_est_pose_graph.nodes[i]

            b_Sim2_i = aligned_est_pose_graph.nodes[i].global_Sim2_local
            a_Sim2_i = a_Sim2_b.compose(b_Sim2_i)
            # equivalent of `transformFrom()` on Pose2 object.
            pano_data_i.global_Sim2_local = Sim2(
                R=a_Sim2_i.rotation, t=a_Sim2_i.translation * a_Sim2_i.scale, s=gt_scale
            )

            for j in range(len(pano_data_i.windows)):
                pano_data_i.windows[j] = pano_data_i.windows[j].apply_Sim2(a_Sim2_b, gt_scale=gt_scale)

            for j in range(len(pano_data_i.openings)):
                pano_data_i.openings[j] = pano_data_i.openings[j].apply_Sim2(a_Sim2_b, gt_scale=gt_scale)

            for j in range(len(pano_data_i.doors)):
                pano_data_i.doors[j] = pano_data_i.doors[j].apply_Sim2(a_Sim2_b, gt_scale=gt_scale)

            # replace this PanoData with the new aligned version.
            aligned_est_pose_graph.nodes[i] = pano_data_i

        return aligned_est_pose_graph


    def measure_avg_abs_rotation_err(self, gt_floor_pg: PoseGraph2d) -> float:
        """After aligning two (potentially rotation-only) pose graphs by Karcher, evaluate rotation angle differences.

        If `self` is the estimate, then we measure our error w.r.t. GT argument.
        An alternative eval method would be to measure how the absolute rotations satisfy the individual binary
        measurement constraints introduced by the ground truth pose graph.

        Args:
            gt_floor_pg: ground truth pose graph for a single floor.

        Returns:
            mean_relative_rot_err: average error, aggregated over all relative rotations.
        """
        # First, perform Karcher-mean alignment of the rotation-only pose graphs.
        # Use only the number of estimated images as #nodes (not # of GT images).
        num_images = max(self.nodes.keys()) + 1

        def convert_posegraph2d_to_Rot3_list(pg: PoseGraph2d) -> List[Optional[Rot3]]:
            """Convert 2d pose graph to list of Rot(3) objects."""
            wRi_list = [None] * num_images
            for i, pano_obj in pg.nodes.items():
                wRi = pano_obj.global_Sim2_local.rotation
                wRi_list[i] = rotation_utils.rot2x2_to_Rot3(wRi)
            return wRi_list

        aRi_list = convert_posegraph2d_to_Rot3_list(pg=gt_floor_pg)
        bRi_list = convert_posegraph2d_to_Rot3_list(pg=self)

        # Obtained transformed input rotations (previously known as "bRi_list", but now living in "a" frame).
        aRi_list_ = gtsfm_geometry_comparisons.align_rotations(aRi_list=aRi_list, bRi_list=bRi_list)

        def _convert_Rot3_to_Sim2(R: Rot3) -> Sim2:
            return Sim2(R=R.matrix()[:2,:2], t=np.zeros(2), s=1.0)

        errs = []
        for pano_id, (aRi, aRi_) in enumerate(zip(aRi_list, aRi_list_)):
            if aRi is None or aRi_ is None:
                continue
            aSi = _convert_Rot3_to_Sim2(aRi)
            aSi_ = _convert_Rot3_to_Sim2(aRi_)

            theta_deg_est = aSi.theta_deg
            theta_deg_gt = aSi_.theta_deg

            print(f"\tPano {pano_id} error (post-alignment): GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f} deg.")

            # We must wrap around the angles at 360.
            err = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print(
            f"Mean absolute rot. error: {mean_err:.1f}. "
            f"Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos."
        )
        return mean_err

    def measure_avg_rel_rotation_err(
        self, gt_floor_pg: PoseGraph2d, gt_edges: List[Tuple[int, int]], verbose: bool = True
    ) -> float:
        """Measure average relative rotation error over relative pose edges.

        Because evaluation here is relative, rather than absolute, pose-graph alignment is not required.

        Args:
            gt_floor_pg: ground truth pose graph.
            gt_edges: list of pano-pano edges to use for evaluation. These generally should represent a list of
                (i1,i2) pairs representing panorama pairs where a W/D/O is found closeby between the two.

        Returns:
            Scalar indicating mean relative rotation error.
        """
        errs = []
        for (i1, i2) in gt_edges:

            if not (i1 in self.nodes and i2 in self.nodes):
                continue

            wTi1_gt = gt_floor_pg.nodes[i1].global_Sim2_local
            wTi2_gt = gt_floor_pg.nodes[i2].global_Sim2_local
            i2Ti1_gt = wTi2_gt.inverse().compose(wTi1_gt)

            wTi1 = self.nodes[i1].global_Sim2_local
            wTi2 = self.nodes[i2].global_Sim2_local
            i2Ti1 = wTi2.inverse().compose(wTi1)

            theta_deg_est = i2Ti1.theta_deg
            theta_deg_gt = i2Ti1_gt.theta_deg

            if verbose:
                print(f"\tPano pair ({i1},{i2}): GT {theta_deg_gt:.1f} vs. {theta_deg_est:.1f}")

            # need to wrap around at 360
            err = rotation_utils.wrap_angle_deg(theta_deg_gt, theta_deg_est)
            errs.append(err)

        mean_err = np.mean(errs)
        print_str = f"Mean relative rot. error: {mean_err:.1f}. "
        print_str += f"Estimated rotation for {len(self.nodes)} of {len(gt_floor_pg.nodes)} GT panos"
        print_str += f", estimated {len(errs)} / {len(gt_edges)} GT edges"
        print(REDTEXT + print_str + ENDCOLOR)

        print(REDTEXT + "Rotation Errors: " + str(np.round(errs, 1)) + ENDCOLOR)

        return mean_err

    def draw_edge(self, i1: int, i2: int, color: str) -> None:
        """ """
        t1 = self.nodes[i1].global_Sim2_local.transform_from(np.zeros((1, 2)))
        t2 = self.nodes[i2].global_Sim2_local.transform_from(np.zeros((1, 2)))

        t1 = t1.squeeze()
        t2 = t2.squeeze()

        plt.plot([t1[0], t2[0]], [t1[1], t2[1]], c=color, linestyle="dotted", alpha=0.6)

    def save_as_zind_data_json(self, save_fpath: str) -> None:
        """ """
        os.makedirs(Path(save_fpath).parent, exist_ok=True)

        partialroom_pano_dict = defaultdict(list)

        floor_merger_dict = {}
        # for each complete room: "complete_room_{complete_room_id}"
        # for each partial room "partial_room_{partial_room_id}"
        # for each pano "pano_{pano_id}"

        for pano_id, pano_data in self.nodes.items():

            fname_stem = Path(pano_data.image_path).stem
            fname_stem = fname_stem.replace(f"{self.floor_id}_", "")
            k = fname_stem.find("_pano_")

            # partial_room_id e.g. partial_room_06
            partial_room_id = fname_stem[:k]
            assert pano_id == pano_data.id

            DUMMY_VAL_INF = 99999

            # should be vstacked 3x2 arrays, for (pt1, pt2, (bottomz, topz))
            doors = []
            for d in pano_data.doors:
                doors.append(d.pt1)
                doors.append(d.pt2)
                doors.append((-DUMMY_VAL_INF, DUMMY_VAL_INF))

            windows = []
            for w in pano_data.windows:
                windows.append(w.pt1)
                windows.append(w.pt2)
                windows.append((-DUMMY_VAL_INF, DUMMY_VAL_INF))

            openings = []
            for o in pano_data.openings:
                openings.append(o.pt1)
                openings.append(o.pt2)
                openings.append((-DUMMY_VAL_INF, DUMMY_VAL_INF))

            v = pano_data.room_vertices_local_2d

            # from rdp import rdp
            # # See https://cartography-playground.gitlab.io/playgrounds/douglas-peucker-algorithm/
            # # https://rdp.readthedocs.io/en/latest/

            # import matplotlib.pyplot as plt
            # plt.close("all")
            # plt.figure(figsize=(10,8))
            # plt.axis("equal")
            # for i in range(2):

            #     if i == 1:
            #         # Run Ramer-Douglas-Peucker
            #         # render the simplified polygon on the second iteration
            #         v = rdp(v, epsilon=0.02)
            #         color = 'g'
            #         print(f"-> To {v.shape}")
            #     elif i == 0:
            #         print(f"From {v.shape}")
            #         color = 'k'
            #         #import pdb; pdb.set_trace()
            #         xmin, ymin = np.amin(v, axis=0)
            #         xmax, ymax = np.amax(v, axis=0)

            #     plt.plot(v[:, 0], v[:, 1], 10, color=color)
            #     plt.scatter(v[:, 0], v[:, 1], 10, color='r', marker='.')

            # plt.xlim([xmin - 1,xmax + 1])
            # plt.ylim([ymin - 1,ymax + 1])
            # plt.tight_layout()

            # plt.show()
            # plt.close("all")

            vertices = np.round(pano_data.room_vertices_local_2d, 3).tolist()

            pano_dict = {
                "is_primary": None,
                "is_inside": None,
                "layout_complete": None,
                "camera_height": 1.0,
                "floor_number": None,
                "label": pano_data.label,
                "floor_plan_transformation": {"rotation": None, "translation": None, "scale": None},
                # raw layout consists of 2-tuples
                "layout_raw": {"doors": doors, "vertices": vertices, "windows": windows, "openings": openings},
                "is_ceiling_flat": None,
                "image_path": pano_data.image_path,
                "layout_visible": None,
                "checksum": None,
                "ceiling_height": None,
            }
            partialroom_pano_dict[partial_room_id] += [pano_dict]

        # dummy ID for complete room ID
        floor_merger_dict = {"complete_room_99999": partialroom_pano_dict}

        # this is the dictionary that will be serialized to disk.
        save_dict = {
            "redraw": {},
            "floorplan_to_redraw_transformation": {},  # not needed for SfM.
            "scale_meters_per_coordinate": {self.floor_id: self.scale_meters_per_coordinate},
            "merger": {self.floor_id: floor_merger_dict},
        }
        io_utils.save_json_file(json_fpath=save_fpath, data=save_dict)


def convert_Sim3_to_Sim2(a_Sim3_b: Similarity3) -> Sim2:
    """Convert a Similarity(3) object to a Similarity(2) object by."""
    # we only care about the rotation about the upright Z axis
    a_Rot2_b = a_Sim3_b.rotation().matrix()[:2, :2]
    theta_deg = rotation_utils.rotmat2theta_deg(a_Rot2_b)

    rx, ry, rz = Rotation.from_matrix(a_Sim3_b.rotation().matrix()).as_euler("xyz", degrees=True)

    MAX_ALLOWED_RX_DEV = 0.1
    MAX_ALLOWED_RY_DEV = 0.1
    if np.absolute(rx) > MAX_ALLOWED_RX_DEV or np.absolute(ry) > MAX_ALLOWED_RY_DEV:
        import pdb

        pdb.set_trace()

    assert np.isclose(rz, theta_deg, atol=0.1)

    atb = a_Sim3_b.translation()
    MAX_ALLOWED_TZ_DEG = 0.1
    if np.absolute(atb[2]) > MAX_ALLOWED_TZ_DEG:
        import pdb

        pdb.set_trace()

    a_Sim2_b = Sim2(R=a_Rot2_b, t=atb[:2], s=a_Sim3_b.scale())
    return a_Sim2_b


def get_single_building_pose_graphs(building_id: str, pano_dir: str, json_annot_fpath: str) -> Dict[str, PoseGraph2d]:
    """Retrieve ground truth 2d pose graphs for all floors for a specific ZInD building.

    TODO: consider simplifying arguments to this function by merging with `get_gt_pose_graph`.

    Args:
        building_id: unique ID of ZInD building.
        pano_dir: path to `{ZInD}/{building_id}/panos` directory.
        json_annot_fpath: path to `{ZInD}/{building_id}/zind_data.json"

    Returns:
        floor_pg_dict: mapping from floor_id to pose graph.
    """
    # Note: `floor_map_json` has 3 keys: 'scale_meters_per_coordinate', 'merger', 'redraw'.
    floor_map_json = io_utils.read_json_file(json_annot_fpath)
    scale_meters_per_coordinate_dict = floor_map_json["scale_meters_per_coordinate"]

    if "merger" not in floor_map_json:
        print(f"Building {building_id} missing `merger` data, skipping...")
        return

    floor_pg_dict = {}

    merger_data = floor_map_json["merger"]
    for floor_id, floor_data in merger_data.items():

        scale_meters_per_coordinate = scale_meters_per_coordinate_dict[floor_id]
        if scale_meters_per_coordinate is None:
            # Impute value!
            # There are floor plans with no scale (specifically, 'floor_XX' : None),
            # and this is typically caused by issues in calibration.
            # See https://github.com/zillow/zind/blob/main/data_organization.md#glossary-of-terms
            # to fill in this missing data by using the average across the other floors,
            valid_scales = [v for v in scale_meters_per_coordinate_dict.values() if v is not None]
            avg_valid_scale = np.mean(valid_scales) if len(valid_scales) > 0 else None
            if avg_valid_scale is not None:
                scale_meters_per_coordinate = avg_valid_scale
            else:
                # and if all floors in a tour have none use the average across the dataset
                # (that should be a relatively stable number)
                scale_meters_per_coordinate = ZIND_AVERAGE_SCALE_METERS_PER_COORDINATE

        fd = FloorData.from_json(floor_data, floor_id)
        pg = PoseGraph2d.from_floor_data(
            building_id=building_id, fd=fd, scale_meters_per_coordinate=scale_meters_per_coordinate
        )

        floor_pg_dict[floor_id] = pg

    return floor_pg_dict


def get_gt_pose_graph(building_id: int, floor_id: str, raw_dataset_dir: str) -> PoseGraph2d:
    """Obtain ground truth pose graph...

    Args:
        building_id: unique ID of ZInD building.
        floor_id: unique floor ID within a ZInD building.
        raw_dataset_dir: Path to where ZInD dataset is stored on disk (after download from Bridge API).

    Returns:
        Pose graph for a single floor...
    """
    pano_dir = f"{raw_dataset_dir}/{building_id}/panos"
    json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
    floor_pg_dict = get_single_building_pose_graphs(building_id, pano_dir, json_annot_fpath)
    return floor_pg_dict[floor_id]


def compute_available_floors_for_building(building_id: str, raw_dataset_dir: str) -> List[str]:
    """Compute the list of available floors for a ZInD building.

    Args:
        building_id: unique ID of ZInD building.
        raw_dataset_dir: Path to where ZInD dataset is stored on disk (after download from Bridge API).

    Returns:
        List of available floors, by their floor ID.
    """
    json_annot_fpath = f"{raw_dataset_dir}/{building_id}/zind_data.json"
    if not Path(json_annot_fpath):
        raise ValueError(
            "Ground truth annotations missing in downloaded copy of ZInD, please re-download."
            f" {json_annot_fpath} missing."
        )
    floor_map_json = io_utils.read_json_file(json_annot_fpath)

    if "merger" not in floor_map_json:
        raise ValueError(f"Building {building_id} missing `merger` data.")

    merger_data = floor_map_json["merger"]
    available_floor_ids = list(merger_data.keys())
    return available_floor_ids

