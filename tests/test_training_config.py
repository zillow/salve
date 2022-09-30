"""Unit tests on TrainingConfig"""

import hydra
from hydra.utils import instantiate

from salve.training_config import TrainingConfig


def test_initialize_training_config() -> None:
    """ """
    config_name = "2021_10_26_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"

    with hydra.initialize_config_module(config_module="salve.configs"):
        # config is relative to the `salve` module
        cfg = hydra.compose(config_name=config_name)
        args = instantiate(cfg.TrainingConfig)

    assert isinstance(args, TrainingConfig)
