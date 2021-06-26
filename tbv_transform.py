import collections
import random
import math
from typing import Callable, List, Optional, Tuple

import cv2
import numbers
import numpy as np

import torch
from torch import Tensor


class ComposeQuadruplet(object):
    """
    Composes transforms together into a chain of operations, for 4 aligned inputs.
    """

    def __init__(self, transform: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ """
        for t in self.segtransform:
            image1, image2, image3, image4 = t(image1, image2, image3, image4)
        return image1, image2, image3, image4


class ToTensorQuadruple(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray, image4: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ """
        if not all([isinstance(img, np.ndarray) for img in [image1, image2, image3, image4]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray [eg: data readed by cv2.imread()].\n")

        if not all([img.ndim == 3 for img in [image1, image2, image3, image4]]):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")

        def to_tensor_op(img: np.ndarray) -> Tensor:
            """Convert to PyTorch tensor."""
            # convert from HWC to CHW for collate/batching into NCHW
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))
            if not isinstance(img_tensor, torch.FloatTensor):
                img_tensor = img_tensor.float()
            return img_tensor

        image1 = to_tensor_op(image1)
        image2 = to_tensor_op(image2)
        image3 = to_tensor_op(image3)
        image4 = to_tensor_op(image4)

        return image1, image2, image3, image4


class NormalizeQuadruplet(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        """ """
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(
        self, image1: Tensor, image2: Tensor, image3: Tensor, image4: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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


class ResizeQuadruplet(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(
        self, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image1 = cv2.resize(image1, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image3 = cv2.resize(image3, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image4 = cv2.resize(image4, self.size[::-1], interpolation=cv2.INTER_LINEAR)

        return image1, image2, image3, image4


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


# class Crop(object):
#     """Crops the given ndarray image (H*W*C or H*W).
#     Args:
#         size (sequence or int): Desired output size of the crop. If size is an
#         int instead of sequence like (h, w), a square crop (size, size) is made.
#     """
#     def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
#         if isinstance(size, int):
#             self.crop_h = size
#             self.crop_w = size
#         elif isinstance(size, collections.Iterable) and len(size) == 2 \
#                 and isinstance(size[0], int) and isinstance(size[1], int) \
#                 and size[0] > 0 and size[1] > 0:
#             self.crop_h = size[0]
#             self.crop_w = size[1]
#         else:
#             raise (RuntimeError("crop size error.\n"))
#         if crop_type == 'center' or crop_type == 'rand':
#             self.crop_type = crop_type
#         else:
#             raise (RuntimeError("crop type error: rand | center\n"))
#         if padding is None:
#             self.padding = padding
#         elif isinstance(padding, list):
#             if all(isinstance(i, numbers.Number) for i in padding):
#                 self.padding = padding
#             else:
#                 raise (RuntimeError("padding in Crop() should be a number list\n"))
#             if len(padding) != 3:
#                 raise (RuntimeError("padding channel is not equal with 3\n"))
#         else:
#             raise (RuntimeError("padding in Crop() should be a number list\n"))
#         if isinstance(ignore_label, int):
#             self.ignore_label = ignore_label
#         else:
#             raise (RuntimeError("ignore_label should be an integer number\n"))

#     def __call__(self, image, label):
#         h, w = label.shape
#         pad_h = max(self.crop_h - h, 0)
#         pad_w = max(self.crop_w - w, 0)
#         pad_h_half = int(pad_h / 2)
#         pad_w_half = int(pad_w / 2)
#         if pad_h > 0 or pad_w > 0:
#             if self.padding is None:
#                 raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
#             image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
#             label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
#         h, w = label.shape
#         if self.crop_type == 'rand':
#             h_off = random.randint(0, h - self.crop_h)
#             w_off = random.randint(0, w - self.crop_w)
#         else:
#             h_off = int((h - self.crop_h) / 2)
#             w_off = int((w - self.crop_w) / 2)
#         image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
#         label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
#         return image, label


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


class RandomHorizontalFlipQuadruplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            image3 = cv2.flip(image3, 1)
            image4 = cv2.flip(image4, 1)

        return image1, image2, image3, image4


class RandomVerticalFlipQuadruplet(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
            image3 = cv2.flip(image3, 0)
            image4 = cv2.flip(image4, 0)

        return image1, image2, image3, image4


# class RandomGaussianBlur(object):
#     def __init__(self, radius=5):
#         self.radius = radius

#     def __call__(self, image, label):
#         if random.random() < 0.5:
#             image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
#         return image, label
