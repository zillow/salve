"""Unit tests to make sure data augmentation / preprocessing transforms work correctly."""

import copy
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import salve.utils.transform as transform_utils

_TEST_DATA_ROOT = Path(__file__).parent.parent / "test_data"

# Sample renderings taken from `Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8`.
_BUILDING_RENDERINGS_ROOT = _TEST_DATA_ROOT / "Renderings" / "gt_alignment_approx" / "1208"

IMG_FNAME_CEILING_1 = "pair_58___door_0_0_rotated_ceiling_rgb_floor_01_partial_room_04_pano_5.jpg"
IMG_FNAME_CEILING_2 = "pair_58___door_0_0_rotated_ceiling_rgb_floor_01_partial_room_07_pano_8.jpg"
IMG_FNAME_FLOOR_1 = "pair_58___door_0_0_rotated_floor_rgb_floor_01_partial_room_04_pano_5.jpg"
IMG_FNAME_FLOOR_2 = "pair_58___door_0_0_rotated_floor_rgb_floor_01_partial_room_07_pano_8.jpg"


def _get_pair_image_data() -> Tuple[np.ndarray, np.ndarray]:
    """ """
    x1f = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_FLOOR_1)
    x2f = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_FLOOR_2)
    return x1f, x2f


def _get_quadruplet_image_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve sample data from building 1208 (ceiling and floor texture renderings for panos 5 and 8)."""

    x1c = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_CEILING_1)
    x2c = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_CEILING_2)
    x1f = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_FLOOR_1)
    x2f = imageio.imread(_BUILDING_RENDERINGS_ROOT / IMG_FNAME_FLOOR_2)
    return x1c, x2c, x1f, x2f


def test_compose_quadruplet() -> None:
    """Ensures that composing together many transforms can yield tensors of expected shape."""
    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()

    # As if using 2 input modalities (2 renderings per pano).
    Resize = transform_utils.ResizeQuadruplet
    Crop = transform_utils.CropQuadruplet
    ToTensor = transform_utils.ToTensorQuadruplet
    Normalize = transform_utils.NormalizeQuadruplet
    RandomHorizontalFlip = transform_utils.RandomHorizontalFlipQuadruplet
    RandomVerticalFlip = transform_utils.RandomVerticalFlipQuadruplet
    Compose = transform_utils.ComposeQuadruplet

    # Corresponds to ImageNet mean and std.
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.120000000000005, 57.375]

    resize_h, resize_w = 300, 300
    train_h, train_w = 200, 200

    transform_list = [
        Resize(size=(resize_h, resize_w)),
        Crop(size=(train_h, train_w), crop_type="rand", padding=mean),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]
    transform = transform_utils.ComposeQuadruplet(transforms=transform_list)
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    # All outputs should be Pytorch tensors.
    assert isinstance(x1c_, torch.Tensor)
    assert isinstance(x2c_, torch.Tensor)
    assert isinstance(x1f_, torch.Tensor)
    assert isinstance(x2f_, torch.Tensor)

    # Shape should be preserved.
    assert x1c_.shape == (3, 200, 200)
    assert x2c_.shape == (3, 200, 200)
    assert x1f_.shape == (3, 200, 200)
    assert x2f_.shape == (3, 200, 200)


def test_normalize_quadruplet() -> None:
    """Ensures that pixel value normlization by mean and standard deviation is correct."""
    img_h, img_w = 501, 501
    fill_value = 220
    img = np.ones((3, img_h, img_w), dtype=np.float32) * fill_value
    img = torch.from_numpy(img)

    # Make 4 separate identical images for quadruplet.
    x1c, x2c, x1f, x2f = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)

    mean = [110, 110, 110]
    std = [55, 55, 55]

    transform = transform_utils.NormalizeQuadruplet(mean=mean, std=std)
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    expected_output = torch.ones((3, img_h, img_w), dtype=torch.float32) * 2.0
    assert torch.allclose(x1c_, expected_output)
    assert torch.allclose(x2c_, expected_output)
    assert torch.allclose(x1f_, expected_output)
    assert torch.allclose(x2f_, expected_output)


def test_totensor_quadruplet() -> None:
    """Ensures that ToTensor() transform converts HWC numpy arrays to CHW Pytorch tensors, preserving dims."""
    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()
    transform = transform_utils.ToTensorQuadruplet()
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    # All outputs should be Pytorch tensors.
    assert isinstance(x1c_, torch.Tensor)
    assert isinstance(x2c_, torch.Tensor)
    assert isinstance(x1f_, torch.Tensor)
    assert isinstance(x2f_, torch.Tensor)

    # Shape should be preserved.
    assert x1c_.shape == (3, 501, 501)
    assert x2c_.shape == (3, 501, 501)
    assert x1f_.shape == (3, 501, 501)
    assert x2f_.shape == (3, 501, 501)


def test_resize_quadruplet() -> None:
    """Ensures we can resize a (501,501) RGB image to a lower-resolution of (234,234)."""
    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()

    resize_h, resize_w = 234, 234

    transform = transform_utils.ResizeQuadruplet(size=(resize_h, resize_w))
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    assert x1c_.shape == (234, 234, 3)
    assert x2c_.shape == (234, 234, 3)
    assert x1f_.shape == (234, 234, 3)
    assert x2f_.shape == (234, 234, 3)


@pytest.mark.parametrize(
    "crop_type",
    ["center", "rand"],
)
def test_crop_quadruplet(crop_type: str) -> None:
    """Ensures we can take a (224,224) crop from a (501,501) RGB image, with two different crop types."""
    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()

    train_h, train_w = 224, 224
    # Using ImageNet's mean below.
    mean = [123.675, 116.28, 103.53]

    transform = transform_utils.CropQuadruplet(size=(train_h, train_w), crop_type=crop_type, padding=mean)
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    assert x1c_.shape == (224, 224, 3)
    assert x2c_.shape == (224, 224, 3)
    assert x1f_.shape == (224, 224, 3)
    assert x2f_.shape == (224, 224, 3)


def test_random_horizontal_flip_quadruplet() -> None:
    """Ensures that random horizontal flip provides correct result in deterministic setting."""

    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()
    # Flip prob of 1.0 means deterministic horizontal flip.
    transform = transform_utils.RandomHorizontalFlipQuadruplet(p=1.0)
    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    # Should horizontally mirror Pano 1's ceiling rendering.
    assert np.allclose(x1c[:, ::-1, :], x1c_)

    # Should horizontally mirror Pano 2's ceiling rendering.
    assert np.allclose(x2c[:, ::-1, :], x2c_)

    # Should horizontally mirror Pano 1's floor rendering.
    assert np.allclose(x1f[:, ::-1, :], x1f_)

    # Should horizontally mirror Pano 2's floor rendering.
    assert np.allclose(x2f[:, ::-1, :], x2f_)

    # Shape should be preserved.
    assert x1c_.shape == (501, 501, 3)
    assert x2c_.shape == (501, 501, 3)
    assert x1f_.shape == (501, 501, 3)
    assert x2f_.shape == (501, 501, 3)


def test_random_vertical_flip_quadruplet() -> None:
    """Ensures that vertical horizontal flip provides correct result in deterministic setting."""
    x1c, x2c, x1f, x2f = _get_quadruplet_image_data()
    # Flip prob of 1.0 means deterministic vertical flip.
    transform = transform_utils.RandomVerticalFlipQuadruplet(p=1.0)

    x1c_, x2c_, x1f_, x2f_ = transform(x1c, x2c, x1f, x2f)

    # Should vertically mirror Pano 1's ceiling rendering.
    assert np.allclose(x1c[::-1, :, :], x1c_)

    # Should vertically mirror Pano 2's ceiling rendering.
    assert np.allclose(x2c[::-1, :, :], x2c_)

    # Should vertically mirror Pano 1's floor rendering.
    assert np.allclose(x1f[::-1, :, :], x1f_)

    # Should vertically mirror Pano 2's floor rendering.
    assert np.allclose(x2f[::-1, :, :], x2f_)

    # Shape should be preserved.
    assert x1c_.shape == (501, 501, 3)
    assert x2c_.shape == (501, 501, 3)
    assert x1f_.shape == (501, 501, 3)
    assert x2f_.shape == (501, 501, 3)


def test_photometric_shift_quadruplet() -> None:
    """Ensures we can apply several photometric transforms to 4 images.

    These photometric transforms include brightness, constrast, saturation, and hue changes.
    """

    for trial in range(10):

        image1, image2, image3, image4 = _get_quadruplet_image_data()
        grid_row1 = np.hstack([image1, image2, image3, image4])

        transform = transform_utils.PhotometricShiftQuadruplet(
            jitter_types=["brightness", "contrast", "saturation", "hue"]
        )
        image1_, image2_, image3_, image4_ = transform(image1, image2, image3, image4)

        # Shape should be preserved.
        assert image1_.shape == (501, 501, 3)
        assert image2_.shape == (501, 501, 3)
        assert image3_.shape == (501, 501, 3)
        assert image4_.shape == (501, 501, 3)

        # All images should be modified from the corresponding original in some way.
        assert not np.allclose(image1_, image1)
        assert not np.allclose(image2_, image2)
        assert not np.allclose(image3_, image3)
        assert not np.allclose(image4_, image4)

        visualize = False
        if visualize:
            grid_row2 = np.hstack([image1, image2, image3, image4])
            grid = np.vstack([grid_row1, grid_row2])

            plt.figure(figsize=(16, 10))
            plt.tight_layout()
            plt.axis("off")
            plt.imshow(grid)
            plt.show()


def test_photometric_shift_quadruplet_no_jitter_types_is_identity() -> None:
    """Ensures that executing the PhotometricShiftQuadruplet transform with no jitter types is an identity op to 4 images."""
    for trial in range(10):
        np.random.seed(trial)
        image1, image2, image3, image4 = _get_quadruplet_image_data()

        transform = transform_utils.PhotometricShiftQuadruplet(jitter_types=[])
        image1_, image2_, image3_, image4_ = transform(image1, image2, image3, image4)

        assert np.allclose(image1, image1_)
        assert np.allclose(image2, image2_)
        assert np.allclose(image3, image3_)
        assert np.allclose(image4, image4_)
