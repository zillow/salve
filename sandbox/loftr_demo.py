
"""Match with kornia."""

from pathlib import Path

import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch

import gtsfm.utils.viz as viz_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints

# 480 x 640
INFERENCE_HEIGHT = 320 # 480
INFERENCE_WIDTH = 640 # 960

def match_with_loftr(fpath1: Path, fpath2: Path):
	""" """
	img1 = cv2.imread(str(fpath1))
	img2 = cv2.imread(str(fpath2))

	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	img1 = cv2.resize(img1, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
	img2 = cv2.resize(img2, (INFERENCE_WIDTH, INFERENCE_HEIGHT))

	# plt.imshow(img1)
	# plt.show()

	# plt.imshow(img2)
	# plt.show()

	img1 = torch.from_numpy(img1)[None][None] / 255.
	img2 = torch.from_numpy(img2)[None][None] / 255.

	# img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).type(torch.float32)
	# img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).type(torch.float32)

	loftr = K.feature.LoFTR("indoor")
	matches = loftr({"image0": img1, "image1": img2})

	import pdb; pdb.set_trace()

	idxs = np.argsort(-matches['confidence'])
	sorted_confs = matches['confidence'][idxs]

	kps0 = matches['keypoints0'][idxs]
	kps1 = matches['keypoints1'][idxs]

	MIN_THRESH = 0.9
	num_to_keep = (sorted_confs > MIN_THRESH).sum()
	kps0 = kps0[:num_to_keep]
	kps1 = kps1[:num_to_keep]

	img1 = cv2.imread(str(fpath1))
	img2 = cv2.imread(str(fpath2))

	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

	img1 = cv2.resize(img1, (INFERENCE_WIDTH, INFERENCE_HEIGHT)) # or as 640
	img2 = cv2.resize(img2, (INFERENCE_WIDTH, INFERENCE_HEIGHT))

	# 

	image_i1 = Image(value_array=img1)
	image_i2 = Image(value_array=img2)

	kps_i1 = Keypoints(coordinates=kps0.numpy())
	kps_i2 = Keypoints(coordinates=kps1.numpy())

	N, _ = kps0.shape

	corr_idxs_i1i2 = np.stack([np.arange(N), np.arange(N)], axis=1)

	result = viz_utils.plot_twoview_correspondences(
	    image_i1=image_i1,
	    image_i2=image_i2,
	    kps_i1=kps_i1,
	    kps_i2=kps_i2,
	    corr_idxs_i1i2=corr_idxs_i1i2,
	    inlier_mask=None,
	    dot_color=None,
	    max_corrs=50
	)
	
	plt.imshow(result.value_array)
	plt.show()
		

if __name__ == "__main__":
	""" """
	fpath1 = "/Users/johnlambert/Downloads/ZInD/0715/panos/floor_01_partial_room_09_pano_0.jpg"
	fpath2 = "/Users/johnlambert/Downloads/ZInD/0715/panos/floor_01_partial_room_04_pano_3.jpg"
	match_with_loftr(Path(fpath1), Path(fpath2))