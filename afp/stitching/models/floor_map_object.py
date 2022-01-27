from copy import deepcopy

from stitching.transform import get_global_coords_2d_from_room_cs
from stitching.models.locations import Point2d, Pose


class FloorMapObject:
    def __init__(self, floor_map):
        self.data = floor_map
        self._generate_room_shape_floor_shape_association()
        self.floor_ids_by_panoid = {}
        self.panoids_by_order = {}
        for panoid, pano in self.data['panos'].items():
            order = str(pano['order'])
            self.panoids_by_order[order] = panoid

        for fsid, floor_shape in self.data['floor_shapes'].items():
            for rsid, room_shape in floor_shape['room_shapes'].items():
                for panoid, pano in self.data['room_shapes'][rsid]['panos'].items():
                    self.floor_ids_by_panoid[panoid] = fsid

    def _generate_room_shape_floor_shape_association(self):
        self.fsids = {}
        for fsid, floor_shape in self.data["floor_shapes"].items():
            for rsid, room_shape in floor_shape["room_shapes"].items():
                self.fsids[rsid] = fsid

    def get_panoids_with_floor_number(self, number):
        self.floor_numbers_by_panoid = {}
        for fsid, floor_shape in self.data['floor_shapes'].items():
            floor_number = floor_shape['floor_number']
            for rsid, room_shape in floor_shape['room_shapes'].items():
                for panoid, pano in self.data['room_shapes'][rsid]['panos'].items():
                    self.floor_numbers_by_panoid[panoid] = floor_number
        return [panoid for panoid, floor_number in self.floor_numbers_by_panoid.items() if floor_number == number]

    def get_panoids_with_floor_id(self, floor_shape_id):
        return [panoid for panoid, floor_id in self.floor_ids_by_panoid.items() if floor_id == floor_shape_id]

    def get_floor_map_scale(self):
        fsid_first = [*self.data["floor_shapes"].keys()][0]
        return self.data["floor_shapes"][fsid_first]["scale"]

    def get_pano_global_pose(self, panoid: str) -> Pose:
        room_shape_id = self.data["panos"][panoid]["room_shape_id"]
        room_shape_pano = self.data["room_shapes"][room_shape_id]["panos"][panoid]
        pose = Pose(
            position=Point2d(x=room_shape_pano["position"]["x"], y=room_shape_pano["position"]["y"]),
            rotation=room_shape_pano["rotation"],
        )
        return self.get_global_pose_from_pose_in_room_cs(room_shape_id, pose)

    def get_global_pose_from_pose_in_room_cs(self, room_shape_id: str, pose: Pose) -> Pose:
        floor_shape_id = self.fsids[room_shape_id]
        floor_shape_room_shape = self.data["floor_shapes"][floor_shape_id]["room_shapes"][room_shape_id]
        position_global = get_global_coords_2d_from_room_cs(
            [pose.position.x, pose.position.y],
            floor_shape_room_shape["position"]["x"],
            floor_shape_room_shape["position"]["z"],
            floor_shape_room_shape["rotation"],
            floor_shape_room_shape["scale"],
        )[0]
        rotation_global = pose.rotation + floor_shape_room_shape["rotation"]
        return Pose(position=Point2d(x=position_global[0], y=position_global[1]), rotation=rotation_global)

    def get_room_pose_from_pose_to_existing_pano(self, panoid_ref: str, pose: Pose):
        # Not yet implemented.
        return None

    def get_panoid_by_pano_order(self, order):
        return self.panoids_by_order[str(order)]

    def get_room_shape_global(self, room_shape_id, pose=None):
        room_shape_original = self.data['room_shapes'][room_shape_id]
        room_shape = deepcopy(room_shape_original)
        if pose:
            xz = [-pose.position.x, pose.position.y]
            rotation = pose.rotation
            scale = 1
        else:
            floor_shape_id = self.fsids[room_shape_id]
            floor_shape_room_shape = self.data["floor_shapes"][floor_shape_id]["room_shapes"][room_shape_id]
            xz = [floor_shape_room_shape["position"]["x"], floor_shape_room_shape["position"]["z"]]
            rotation = floor_shape_room_shape["rotation"]
            scale = floor_shape_room_shape["scale"]

        for type in ['doors', 'windows', 'openings']:
            for entityid, door in room_shape_original[type].items():
                position_global = get_global_coords_2d_from_room_cs(
                    [door['position'][0]['x'], door['position'][0]['y']],
                    xz[0], xz[1], rotation, scale,
                )[0]
                room_shape[type][entityid]['position'][0] = {
                    'x': position_global[0], 'y': position_global[1]
                }
                position_global = get_global_coords_2d_from_room_cs(
                    [door['position'][1]['x'], door['position'][1]['y']],
                    xz[0], xz[1], rotation, scale,
                )[0]
                room_shape[type][entityid]['position'][1] = {
                    'x': position_global[0], 'y': position_global[1]
                }

        room_shape['vertices'] = []
        for position in room_shape_original['vertices']:
            position_global = get_global_coords_2d_from_room_cs(
                [position['x'], position['y']],
                xz[0], xz[1], rotation, scale,
            )[0]
            room_shape['vertices'].append({
                'x': position_global[0], 'y': position_global[1]
            })

        return room_shape
