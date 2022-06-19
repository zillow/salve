""" TODO: ADD DOCSTRING """

from typing import Any, Dict, List

import numpy as np
from schematics.types import StringType
from shapely.geometry import Point, Polygon

from salve.stitching.utilities import get_dwo_edge_feature2ds_from_prediction
from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT, WDO_CODE
from salve.stitching.shape import generate_shapely_polygon_from_room_shape_vertices
from salve.stitching.transform import ray_cast_by_u, uv_to_xy
from salve.stitching.io.abstract_loader import AbstractLoader
from salve.stitching.models.feature2d import Feature2dXy
from salve.stitching.models.locations import Point2d

SUPPORTED_PREDICTION_CATEGORIES = ["total", "partial_v1", "joint_madori_v1"]


class PanoDataLayer:
    def __init__(self, type: Any, shape: Any, dwo: Any, position: Any = [0, 0], rotation: float = 0) -> None:
        """TODO

        Args:
            type:
            shape:
            dwo:
            position:
            rotation:
        """
        # Valid types include types in SUPPORTED_PREDICTION_CATEGORIES
        # and "annotated"
        self.type = type
        # Shapely Polygon
        self.shape = shape
        self.dwo = dwo
        self.position = position
        self.rotation = rotation
        self.is_inside_shape = Point(position).within(shape)
        self.is_origin = (abs(position[0]) + abs(position[1])) < 1e-5


class PredictionCategoryType(StringType):
    def validate_content(self, value: Any) -> None:
        """TODO

        Args:
            value: TODO

        Raises:
            ValueError: If TODO
        """
        if value not in SUPPORTED_PREDICTION_CATEGORIES:
            raise ValueError(f"Incorrect prediction category received: {value}")


class PanoObject:
    def __init__(
        self,
        floor_map_guid: str,
        panoid: str,
        loader: AbstractLoader,
        prediction_types: List[str] = [],
        floor_map: dict = None,
    ) -> None:
        """Store shape information, e.g. door / window / opening, room shape, as something like:
            self.data_layer = {'annotated': PanoDataLayer, 'total': PanoDataLayer}
        """
        self.data_layer = {}
        self.floor_map_guid = floor_map_guid
        self.panoid = panoid
        self.camera_height = DEFAULT_CAMERA_HEIGHT
        self.vanishing_angle = None
        if floor_map:
            self._load_room_shape_from_floor_map(floor_map)
            self._load_vanishin_angle_from_floor_map(floor_map)
        if prediction_types:
            self._load_predictions(loader, prediction_types)

    def get_corner_feature2d(self, type: Any) -> List[Feature2dXy]:
        """TODO

        Args:
            type: TODO

        Returns:
            corner_feature2d: TODO
        """
        if type not in self.data_layer:
            raise Exception(
                f"MissingTourDataFile: Data layer {type} cannot be found in PanoObject. It's either not initialized or invalid type."
            )
        shape = self.data_layer[type].shape
        xys_shapely = np.array(shape.boundary.xy)
        corner_feature2d = [
            Feature2dXy.fromPoint2d(Point2d(x=xys_shapely[:, i][0], y=xys_shapely[:, i][1]), "corner")
            for i in range(xys_shapely.shape[1] - 1)
        ]
        return corner_feature2d

    def get_dwo_feature2d(self, type: str) -> List[Any]:
        """TODO

        Args:
            type: TODO

        Returns:
            dwo_feature2ds: TODO
        """
        if type not in self.data_layer:
            raise Exception(
                f"MissingTourDataFile: Data layer {type} cannot be found in PanoObject. It's eithernot initialized or invalid type."
            )
        dwos = self.data_layer[type].dwo
        dwo_feature2ds = []
        for dwo in dwos:
            dwo_feature2ds += dwo
        return dwo_feature2ds

    def _load_vanishin_angle_from_floor_map(self, floor_map: Any) -> None:
        """TODO """
        self.vanishing_angle = floor_map["panos"][self.panoid]["vanishing_angle"]

    def _load_room_shape_from_floor_map(self, floor_map: Dict[str, Any]) -> None:
        """TODO

        Args:
            floor_map: TODO
        """
        if not floor_map["panos"][self.panoid]["room_shape_id"]:
            return
        self.rsid = floor_map["panos"][self.panoid]["room_shape_id"]
        room_shape_raw = floor_map["room_shapes"][self.rsid]
        self.camera_height = room_shape_raw["panos"][self.panoid]["height"]

        room_shape_shapely = generate_shapely_polygon_from_room_shape_vertices(room_shape_raw["vertices"])
        dwos = self._load_dwos_from_floor_map(room_shape_raw)
        position = room_shape_raw["panos"][self.panoid]["position"]
        position = [position["x"], position["y"]]
        rotation = room_shape_raw["panos"][self.panoid]["rotation"]
        self.data_layer["annotated"] = PanoDataLayer("annotated", room_shape_shapely, dwos, position, rotation)

    def _load_dwos_from_floor_map(self, room_shape_raw: Dict[str, Any]) -> List[Any]:
        """TODO

        Args:
            room_shape_raw: TODO

        Returns:
            dwos: Any
        """
        dwos = []
        DWO_TYPES = {"doors": "door", "windows": "window", "openings": "opening"}
        for type_name, type in DWO_TYPES.items():
            for wdo_id in room_shape_raw[type_name]:
                dwo_position = room_shape_raw[type_name][wdo_id]["position"]
                dwos.append(
                    [
                        Feature2dXy.fromPoint2d(Point2d(x=dwo_position[0]["x"], y=dwo_position[0]["y"]), type),
                        Feature2dXy.fromPoint2d(Point2d(x=dwo_position[1]["x"], y=dwo_position[1]["y"]), type),
                    ]
                )
        return dwos

    def _check_prediction_jsons(self, predictions: Dict[Any, Any], type: str) -> None:
        """TODO

        Args:
            predictions: TODO
            type: TODO

        Raises:
            Exception: If ...
        """
        if "room_shape" not in predictions:
            raise Exception(
                f"InvalidRoomShapeFromPrediction: Input room shape of prediction type {type} for panoid {self.panoid}"
                " does not inlcude field room shape."
            )

        MIN_NUMBER_OF_CORNERS = 6
        number_predicted_corners = len(predictions["room_shape"])
        if number_predicted_corners < MIN_NUMBER_OF_CORNERS:
            raise Exception(
                f"InvalidRoomShapeFromPrediction: Input predicted room shape of type {type} for panoid {self.panoid} "
                f"include insufficient number of corners {number_predicted_corners}. "
                f"Expecting more than {MIN_NUMBER_OF_CORNERS} predicted corners."
            )

        if not isinstance(predictions["wdo"], list) or not isinstance(predictions["wdo"][0], list):
            raise Exception(
                f"InvalidDwoFromPrediction: Received unexpected input wdo prediction for panoid {self.panoid}."
            )

    def _load_predictions(self, loader: AbstractLoader, prediction_types: List[str]) -> None:
        """

        Args:
            loader:
            prediction_types:
        """
        # TODO: Implement file loading status.
        for type in prediction_types:
            if type not in SUPPORTED_PREDICTION_CATEGORIES:
                raise Exception(f"InternalImplementationError: Invalid prediction type {type} received.")
            pred = {
                "room_shape": loader.get_room_shape_predictions(self.panoid, type=type),
                "wdo": loader.get_dwo_predictions(self.panoid),
            }

            self._check_prediction_jsons(pred, type)

            shape = self._load_room_shape_polygon_from_predictions(pred["room_shape"])
            if type == "total":
                dwos = self._ray_cast_and_generate_dwo_xy(pred["wdo"], shape)
            else:
                dwos = self._load_dwos_from_predictions(pred)

            self.data_layer[type] = PanoDataLayer(type, shape, dwos)

    def _load_room_shape_polygon_from_predictions(self, room_shape_pred: Dict[str, Any]) -> Polygon:
        """

        Args:
            room_shape_pred: TODO

        Returns:
            Polygon: TODO
        """
        flag = True
        xys = []
        for corner in room_shape_pred:
            if not flag:
                uv = Point2d(x=corner[0], y=corner[1])
                xy = uv_to_xy(uv, self.camera_height)
                xys.append([xy.x, xy.y])
            flag = not flag
        return Polygon(xys)

    def _load_dwos_from_predictions(self, dwo_pred: Any) -> Any:
        """

        Args:
            dwo_pred

        Returns:

        """
        return get_dwo_edge_feature2ds_from_prediction(dwo_pred, self.camera_height)

    def _ray_cast_and_generate_dwo_xy(self, dwo_pred: Any, shape: Any) -> List[Any]:
        """

        Args:
            dwo_pred: TODO
            shape: TODO

        Returns:
            dwos: TODO
        """
        dwos = []
        for wdo in dwo_pred[0]:
            type = WDO_CODE[int(wdo[0]) - 1]
            confidence = wdo[1]
            if confidence > 0.5 and (type == "door" or type == "window"):
                xy_from = ray_cast_by_u(wdo[2], shape)
                xy_to = ray_cast_by_u(wdo[4], shape)
                if xy_from and xy_to:
                    dwos.append([Feature2dXy.fromPoint2d(xy_from, type), Feature2dXy.fromPoint2d(xy_to, type)])
        return dwos
