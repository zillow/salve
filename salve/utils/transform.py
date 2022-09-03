"""
Utilities for data augmentation and preprocessing.
"""

import collections
import random
import math
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numbers
import numpy as np
import torchvision
import torch
from torch import Tensor

# helper types
ArrayPair = Tuple[np.ndarray, np.ndarray]
ArrayQuadruplet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ArraySextuplet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

TensorPair = Tuple[Tensor, Tensor]
TensorQuadruplet = Tuple[Tensor, Tensor, Tensor, Tensor]
TensorSextuplet = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class ComposePair(object):
    """Composes transforms together into a chain of operations, for 2 aligned inputs."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> TensorPair:
        """ """
        for t in self.transforms:
            image1, image2 = t(image1, image2)
        return image1, image2


class ComposeQuadruplet(object):
    """Composes transforms together into a chain of operations, for 4 aligned inputs."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> TensorQuadruplet:
        """ """
        for t in self.transforms:
            image1, image2, image3, image4 = t(image1, image2, image3, image4)
        return image1, image2, image3, image4


class ComposeSextuplet(object):
    """Composes transforms together into a chain of operations, for 6 aligned inputs."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> TensorSextuplet:
        """ """
        for t in self.transforms:
            image1, image2, image3, image4, image5, image6 = t(image1, image2, image3, image4, image5, image6)
        return image1, image2, image3, image4, image5, image6


def to_tensor_op(img: np.ndarray) -> Tensor:
    """Convert to PyTorch tensor."""
    # convert from HWC to CHW for collate/batching into NCHW
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img_tensor, torch.FloatTensor):
        img_tensor = img_tensor.float()
    return img_tensor


class ToTensorPair(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> TensorPair:
        """ """
        if not all([isinstance(img, np.ndarray) for img in [image1, image2]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray [eg: data readed by cv2.imread()].\n")

        if not all([img.ndim == 3 for img in [image1, image2]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")

        image1 = to_tensor_op(image1)
        image2 = to_tensor_op(image2)

        return image1, image2


class ToTensorQuadruplet(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> TensorQuadruplet:
        """ """
        if not all([isinstance(img, np.ndarray) for img in [image1, image2, image3, image4]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray [eg: data readed by cv2.imread()].\n")

        if not all([img.ndim == 3 for img in [image1, image2, image3, image4]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")

        image1 = to_tensor_op(image1)
        image2 = to_tensor_op(image2)
        image3 = to_tensor_op(image3)
        image4 = to_tensor_op(image4)

        return image1, image2, image3, image4


class ToTensorSextuplet(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> TensorSextuplet:
        """ """
        if not all([isinstance(img, np.ndarray) for img in [image1, image2, image3, image4, image5, image6]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray [eg: data readed by cv2.imread()].\n")

        if not all([img.ndim == 3 for img in [image1, image2, image3, image4, image5, image6]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")

        image1 = to_tensor_op(image1)
        image2 = to_tensor_op(image2)
        image3 = to_tensor_op(image3)
        image4 = to_tensor_op(image4)
        image5 = to_tensor_op(image5)
        image6 = to_tensor_op(image6)

        return image1, image2, image3, image4, image5, image6


class NormalizePair(object):
    """Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std."""
    def __init__(self, mean, std=None):
        """ """
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image1: Tensor, image2: Tensor) -> TensorPair:
        """ """
        for t, m, s in zip(image1, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image2, self.mean, self.std):
            t.sub_(m).div_(s)

        return image1, image2


class NormalizeQuadruplet(object):
    """Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std."""
    def __init__(self, mean, std=None):
        """ """
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image1: Tensor, image2: Tensor, image3: Tensor, image4: Tensor) -> TensorQuadruplet:
        """ """
        for t, m, s in zip(image1, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image2, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image3, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image4, self.mean, self.std):
            t.sub_(m).div_(s)

        return image1, image2, image3, image4


class NormalizeSextuplet(object):
    """Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std."""
    def __init__(self, mean, std=None):
        """ """
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(
        self, image1: Tensor, image2: Tensor, image3: Tensor, image4: Tensor, image5: Tensor, image6: Tensor
    ) -> TensorSextuplet:
        """ """
        for t, m, s in zip(image1, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image2, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image3, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image4, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image5, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(image6, self.mean, self.std):
            t.sub_(m).div_(s)

        return image1, image2, image3, image4, image5, image6


class ResizePair(object):
    """Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w)."""
    def __init__(self, size: Tuple[int, int]) -> None:
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> ArrayPair:
        """ """
        h, w = self.size
        image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_LINEAR)

        return image1, image2


class ResizeQuadruplet(object):
    """Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w)."""
    def __init__(self, size: Tuple[int, int]) -> None:
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> ArrayQuadruplet:
        """ """
        h, w = self.size
        image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_LINEAR)
        image3 = cv2.resize(image3, (w, h), interpolation=cv2.INTER_LINEAR)
        image4 = cv2.resize(image4, (w, h), interpolation=cv2.INTER_LINEAR)

        return image1, image2, image3, image4


class ResizeSextuplet(object):
    """Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w)."""
    def __init__(self, size: Tuple[int, int]) -> None:
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> ArraySextuplet:
        """ """
        h, w = self.size
        image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_LINEAR)
        image3 = cv2.resize(image3, (w, h), interpolation=cv2.INTER_LINEAR)
        image4 = cv2.resize(image4, (w, h), interpolation=cv2.INTER_LINEAR)
        image5 = cv2.resize(image5, (w, h), interpolation=cv2.INTER_LINEAR)
        image6 = cv2.resize(image6, (w, h), interpolation=cv2.INTER_LINEAR)

        return image1, image2, image3, image4, image5, image6


# class RandScale(object):
#     # Randomly resize image & label with scale factor in [scale_min, scale_max]
#     def __init__(self, scale, aspect_ratio=None):
#         assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
#         if isinstance(scale, collections.Iterable) and len(scale) == 2 \
#                 and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
#                 and 0 < scale[0] < scale[1]:
#             self.scale = scale
#         else:
#             raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
#         if aspect_ratio is None:
#             self.aspect_ratio = aspect_ratio
#         elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
#                 and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
#                 and 0 < aspect_ratio[0] < aspect_ratio[1]:
#             self.aspect_ratio = aspect_ratio
#         else:
#             raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

#     def __call__(self, image, label):
#         temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
#         temp_aspect_ratio = 1.0
#         if self.aspect_ratio is not None:
#             temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
#             temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
#         scale_factor_x = temp_scale * temp_aspect_ratio
#         scale_factor_y = temp_scale / temp_aspect_ratio
#         image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
#         label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
#         return image, label


class CropBase(object):
    """Crops the given N-tuple of ndarray images (H*W*C or H*W)."""

    def __init__(
        self, size, crop_type: str = "center", padding: Optional[Tuple[int, int, int]] = None, ignore_label: int = 255
    ) -> None:
        """
        Args:
            size: Desired output size of the crop. Either (h,w) tuple representing (crop height, crop width), or may
               be an a single integer, in which case a square crop (size, size) is made.
        """
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif (
            isinstance(size, collections.Iterable)
            and len(size) == 2
            and isinstance(size[0], int)
            and isinstance(size[1], int)
            and size[0] > 0
            and size[1] > 0
        ):
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")

        if crop_type == "center" or crop_type == "rand":
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center\n")

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError("ignore_label should be an integer number\n")


class CropPair(CropBase):
    """Crops the given pair of ndarray images (H*W*C or H*W)."""

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> ArrayPair:
        """ """
        h, w, _ = image1.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")

            # use self.ignore_label for padding for label maps
            image1 = pad_image(image1, pad_h, pad_w, padding_vals=self.padding)
            image2 = pad_image(image2, pad_h, pad_w, padding_vals=self.padding)

        h, w, _ = image1.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        image1 = image1[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image2 = image2[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        return image1, image2


class CropQuadruplet(CropBase):
    """Crops the given quadruplet of ndarray images (H*W*C or H*W)."""

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> ArrayQuadruplet:
        """ """
        h, w, _ = image1.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")

            # use self.ignore_label for padding for label maps
            image1 = pad_image(image1, pad_h, pad_w, padding_vals=self.padding)
            image2 = pad_image(image2, pad_h, pad_w, padding_vals=self.padding)
            image3 = pad_image(image3, pad_h, pad_w, padding_vals=self.padding)
            image4 = pad_image(image4, pad_h, pad_w, padding_vals=self.padding)

        h, w, _ = image1.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        image1 = image1[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image2 = image2[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image3 = image3[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image4 = image4[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        return image1, image2, image3, image4


class CropSextuplet(CropBase):
    """Crops the given 6-tuple of ndarray images (H*W*C or H*W)."""

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> ArraySextuplet:
        """ """
        h, w, _ = image1.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")

            # use self.ignore_label for padding for label maps
            image1 = pad_image(image1, pad_h, pad_w, padding_vals=self.padding)
            image2 = pad_image(image2, pad_h, pad_w, padding_vals=self.padding)
            image3 = pad_image(image3, pad_h, pad_w, padding_vals=self.padding)
            image4 = pad_image(image4, pad_h, pad_w, padding_vals=self.padding)
            image5 = pad_image(image5, pad_h, pad_w, padding_vals=self.padding)
            image6 = pad_image(image6, pad_h, pad_w, padding_vals=self.padding)

        h, w, _ = image1.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        image1 = image1[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image2 = image2[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image3 = image3[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image4 = image4[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image5 = image5[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image6 = image6[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        return image1, image2, image3, image4, image5, image6


def pad_image(img: np.ndarray, pad_h: int, pad_w: int, padding_vals: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    """ """
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    img = cv2.copyMakeBorder(
        img, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding_vals
    )
    return img


# class RandRotate(object):
#     # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
#     def __init__(self, rotate, padding, ignore_label=255, p=0.5):
#         assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
#         if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
#             self.rotate = rotate
#         else:
#             raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
#         assert padding is not None
#         assert isinstance(padding, list) and len(padding) == 3
#         if all(isinstance(i, numbers.Number) for i in padding):
#             self.padding = padding
#         else:
#             raise (RuntimeError("padding in RandRotate() should be a number list\n"))
#         assert isinstance(ignore_label, int)
#         self.ignore_label = ignore_label
#         self.p = p

#     def __call__(self, image, label):
#         if random.random() < self.p:
#             angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
#             h, w = label.shape
#             matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
#             image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
#             label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
#         return image, label


class RandomHorizontalFlipPair(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> ArrayPair:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)

        return image1, image2


class RandomHorizontalFlipQuadruplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> ArrayQuadruplet:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            image3 = cv2.flip(image3, 1)
            image4 = cv2.flip(image4, 1)

        return image1, image2, image3, image4


class RandomHorizontalFlipSextuuplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> ArraySextuplet:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            image3 = cv2.flip(image3, 1)
            image4 = cv2.flip(image4, 1)
            image5 = cv2.flip(image5, 1)
            image6 = cv2.flip(image6, 1)

        return image1, image2, image3, image4, image5, image6


class RandomVerticalFlipPair(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> ArrayPair:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)

        return image1, image2


class RandomVerticalFlipQuadruplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> ArrayQuadruplet:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
            image3 = cv2.flip(image3, 0)
            image4 = cv2.flip(image4, 0)

        return image1, image2, image3, image4


class RandomVerticalFlipSextuplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image3: np.ndarray,
        image4: np.ndarray,
        image5: np.ndarray,
        image6: np.ndarray,
    ) -> ArraySextuplet:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
            image3 = cv2.flip(image3, 0)
            image4 = cv2.flip(image4, 0)
            image5 = cv2.flip(image5, 0)
            image6 = cv2.flip(image6, 0)

        return image1, image2, image3, image4, image5, image6


# class RandomGaussianBlur(object):
#     def __init__(self, radius=5):
#         self.radius = radius

#     def __call__(self, image, label):
#         if random.random() < 0.5:
#             image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
#         return image, label


class PhotometricShiftQuadruplet(object):
    """Apply photometric shifts to each image in an 4-tuple independently.
    
    Note: we find that utitilization of this transform at training time does not improve generalization.
    """

    def __init__(self, jitter_types: List[str] = ["brightness", "contrast", "saturation", "hue"]) -> None:
        """Initialize photometric shift parameters.

        We use the `ColorJitter` transfrom from `torchvision`. 
        Ref: https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
        From the official documentation:

        `brightness` (float or tuple of python:float (min, max)) – How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
        Should be non negative numbers.

        `contrast` (float or tuple of python:float (min, max)) – How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
        Should be non negative numbers.

        `saturation` (float or tuple of python:float (min, max)) – How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
        Should be non negative numbers.

        `hue` (float or tuple of python:float (min, max)) – How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

        brightness basically performs weighted average with all zeros. 0.5 will blend with [0.5,1.5] range

        contrast is multiplicative. hue changes the color from red to yellow, etc.

        hue should be changed in a very minor way, only. (never more than 0.05). Hue changes the color itself

        Args:
            jitter_types: types of jitter to apply.
        """
        self.jitter_types = jitter_types

        self.brightness_jitter = 0.5 if "brightness" in jitter_types else 0
        self.contrast_jitter = 0.5 if "contrast" in jitter_types else 0
        self.saturation_jitter = 0.5 if "saturation" in jitter_types else 0
        self.hue_jitter = 0.05 if "hue" in jitter_types else 0

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies photometric shifts to each image in an 4-tuple independently."""

        jitter = torchvision.transforms.ColorJitter(
            brightness=self.brightness_jitter,
            contrast=self.contrast_jitter,
            saturation=self.saturation_jitter,
            hue=self.hue_jitter,
        )

        imgs = [image1, image2, image3, image4]
        jittered_imgs = []
        for img in imgs:
            # HWC -> CHW
            img_pytorch = torch.from_numpy(img).permute(2, 0, 1)
            img_pytorch = jitter(img_pytorch)
            # CHW -> HWC
            jittered_imgs.append(img_pytorch.numpy().transpose(1, 2, 0))

        image1, image2, image3, image4 = jittered_imgs
        return image1, image2, image3, image4
