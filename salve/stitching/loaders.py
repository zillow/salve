""" TODO: ADD DOCSTRING """

import abc
import json
import logging
import os
from typing import Any, Dict, List

from salve.stitching.constants import (
    JOINT_MADORI_V1_FILENAME,
    ROOM_SHAPE_PARTIAL_V1_FILENAME,
    ROOM_SHAPE_TOTAL_FILENAME,
    WDO_FILENAME1,
    WDO_FILENAME2,
)

DEFAULT_DATA_TYPE = {"rse": ["partial_v1"], "dwo": ["rcnn"]}

logger = logging.getLogger()


class AbstractLoader(abc.ABC):
    @abc.abstractclassmethod
    def get_room_shape_predictions(self, pano_id: str, type: str = "partial") -> dict:
        pass  # pragma: no cover

    @abc.abstractclassmethod
    def get_dwo_predictions(self, pano_id: str) -> dict:
        pass  # pragma: no cover


class MemoryLoader(AbstractLoader):
    def __init__(
        self,
        data_root: str,
        data_type: Dict[str, List[str]] = DEFAULT_DATA_TYPE,
    ) -> None:
        """TODO

        Args:
            data_root: TODO
            data_type: TODO
        """
        self.data_root = data_root
        self.data_type = data_type
        self._data: Dict[str, Dict[str, Any]] = {"per_pano_predictions": {}}
        self._check_data_type()
        self._load_predictions()

    def _check_data_type(self) -> None:
        """TODO"""
        if "rse" not in self.data_type:
            raise Exception("InternalImplementationError")
        if "dwo" not in self.data_type:
            raise Exception("InternalImplementationError")
        if not self.data_type["rse"]:
            raise Exception("InternalImplementationError")
        if not self.data_type["dwo"]:
            raise Exception("InternalImplementationError")

    def _load_predictions(self) -> None:
        """Load predicted layout and D/W/O predictions from pre-trained model.

        TODO, e.g. `cd3af40860` TODO: re-export the data just use ZInD panorama IDs.
        """
        folders = os.listdir(self.data_root)
        panoids = [item for item in folders if len(item) == 10 and not item.startswith(".")]
        # pano IDs here are length-10 strings, e.g. '3b91f0daef'
        for panoid in panoids:

            # Load layout predictions.
            self._data["per_pano_predictions"][panoid] = {"rse": {}, "dwo": {}}
            for rse_type in self.data_type["rse"]:
                self._data["per_pano_predictions"][panoid]["rse"][rse_type] = None
                self._load_room_shape_predictions(panoid, rse_type)

            # Load D/W/O predictions.
            for dwo_type in self.data_type["dwo"]:
                self._data["per_pano_predictions"][panoid]["dwo"][dwo_type] = None
                self._load_dwo_predictions(panoid, dwo_type)

    def _load_room_shape_predictions(self, panoid: str, type: str = "partial_v1") -> None:
        """Load layout prediction for requested model type (e.g. joint, total geometry, partial geometry, etc.)

        Args:
            panoid: unique panorama identifier, e.g. '3b91f0daef'
            type: requested model type, to obtain a specific type of predictions.
        """
        if type == "total":
            file_name = ROOM_SHAPE_TOTAL_FILENAME
        elif type == "partial_v1":
            file_name = ROOM_SHAPE_PARTIAL_V1_FILENAME
        elif type == "joint_madori_v1":
            file_name = JOINT_MADORI_V1_FILENAME
        else:
            raise Exception(f"InternalImplementationError: Unrecognized type {type}")

        prediction_path = self._get_prediction_file_path(panoid, file_name)

        if not os.path.isfile(os.path.abspath(prediction_path)):
            logger.warning(f"memory_loader: prediction_path {prediction_path} doesn't exist.")
            return

        with open(prediction_path) as f:
            if type == "partial_v1" or type == "joint_madori_v1":
                content = json.load(f)[0]
            elif type == "total":
                content = json.load(f)

            if "predictions" in content:
                if "room_shape" in content["predictions"]:
                    content = content["predictions"]["room_shape"]
                else:
                    content = content["predictions"]
                self._data["per_pano_predictions"][panoid]["rse"][type] = content["corners_in_uv"]
            else:
                self._data["per_pano_predictions"][panoid]["rse"][type] = content["uv"]

    def _load_dwo_predictions(self, panoid: str, type: str) -> None:
        """TODO

        Args:
            panoid: TODO
            type: TODO
        """
        if type == "rcnn":
            prediction_path = self._get_prediction_file_path(panoid, WDO_FILENAME1)
            if not os.path.isfile(prediction_path):
                prediction_path = self._get_prediction_file_path(panoid, WDO_FILENAME2)
        else:
            raise Exception(f"InternalImplementationError: Unrecognized type {type}")

        if not os.path.isfile(prediction_path):
            logger.warning("")
            return

        with open(prediction_path) as f:
            self._data["per_pano_predictions"][panoid]["dwo"][type] = json.load(f)["predictions"]

    def _get_prediction_file_path(self, panoid: str, file_name: str) -> str:
        """Get path to model prediction, as concatenated directory names and file name.

        Predictions from multiple models are stored under a single panorama's directory.
        """
        return os.path.join(self.data_root, panoid, file_name)

    def get_room_shape_predictions(self, panoid: str, type: str = "partial_v1") -> Dict[Any, Any]:
        """TODO"""
        return self._data["per_pano_predictions"][panoid]["rse"][type]

    def get_dwo_predictions(self, panoid: str, type: str = "rcnn") -> Dict[Any, Any]:
        """TODO"""
        return self._data["per_pano_predictions"][panoid]["dwo"][type]
