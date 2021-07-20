
import numpy as np
import torch
import torch.nn.functional as F

from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud
from scipy.interpolate import griddata  # not quite the same as `matplotlib.mlab.griddata`


def interp_dense_grid_from_sparse(
    bev_img: np.ndarray, points: np.ndarray, rgb_values: np.ndarray, grid_h: int, grid_w: int, is_semantics: bool
) -> np.ndarray:
    """
    Args:
       points: (N,2) or (N,3) array of (x,y,z) or (x,y)
    """
    grid_coords = get_mesh_grid_as_point_cloud(min_x=0, max_x=grid_w - 1, min_y=0, max_y=grid_h - 1)
    # Note: `xi` -- Points at which to interpolate data.
    interp_rgb_vals = griddata(
        points=points[:, :2], values=rgb_values, xi=grid_coords, method="nearest" if is_semantics else "linear"
    )  # ) # # or method='linear', method='cubic'

    # can swap axes arbitrarily
    Y = grid_coords[:, 1].astype(np.int32)
    X = grid_coords[:, 0].astype(np.int32)
    bev_img[Y, X, :] = interp_rgb_vals
    return bev_img


def remove_hallucinated_content(sparse_bev_img: np.ndarray, interp_bev_img: np.ndarray, K: int = 41) -> np.ndarray:
    """
    Args:
        K: kernel size, e.g. 3 for 3x3, 5 for 5x5
    """
    H, W, _ = interp_bev_img.shape

    # check if any channel is populated
    mul_bev_img = sparse_bev_img[:,:,0] * sparse_bev_img[:,:,1] * sparse_bev_img[:,:,2]

    mul_bev_img = torch.from_numpy(mul_bev_img).reshape(1, 1, H, W)
    nonempty = (mul_bev_img > 0).type(torch.float32)

    # box filter to sum neighbors
    weight = torch.ones(1,1,K,K).type(torch.float32)

    if torch.cuda.is_available():
    	weight = weight.cuda()
    	nonempty = nonempty.cuda()
    
    counts = F.conv2d(input=nonempty, weight=weight, bias=None, stride=1, padding = K//2)

    if torch.cuda.is_available():
    	counts = counts.cpu()

    #in_channels=1, out_channels=1, kernel_size=K, stride=1, padding = K//2)
    #counts = conv(nonempty)
    mask = counts > 0
    mask = mask.numpy().reshape(H,W).astype(np.float32)

    # CHW -> HWC
    mask = np.tile(mask, (3,1,1) ).transpose(1,2,0)

    unhalluc_img = (mask * interp_bev_img).astype(np.uint8)
    return unhalluc_img


def test_remove_hallucinated_content() -> None:
	""" """
	sparse_bev_img = np.array(
		[
			[0,2,0,4,0,0],
			[0,0,0,0,0,0],
			[0,2,0,0,0,0],
			[0,0,0,0,0,0],
			[0,2,0,0,0,0],
			[0,0,0,0,0,0],
		])
	# simulate 3-channel image
	sparse_bev_img = np.stack([sparse_bev_img, sparse_bev_img, sparse_bev_img], axis=-1)
	

	interp_bev_img = np.array(
		[
			[1,2,3,4,5,6],
			[1,2,3,4,5,6],
			[1,2,3,4,5,6],
			[1,2,3,4,5,6],
			[1,2,3,4,5,6],
			[1,2,3,4,5,6],
		])
	# simulate 3-channel image
	interp_bev_img = np.stack([interp_bev_img, interp_bev_img, interp_bev_img], axis=-1)

	bev_img = remove_hallucinated_content(sparse_bev_img, interp_bev_img, K=3)
	expected_slice = np.array(
		[
			[1, 2, 3, 4, 5, 0],
			[1, 2, 3, 4, 5, 0],
			[1, 2, 3, 0, 0, 0],
			[1, 2, 3, 0, 0, 0],
			[1, 2, 3, 0, 0, 0],
			[1, 2, 3, 0, 0, 0]
		], dtype=np.uint8)

	for i in range(3):
		assert np.allclose(bev_img[:,:,i], expected_slice)


def test_remove_hallucinated_content_largekernel():
	""" """
	sparse_bev_img = np.random.randint(low=0, high=255, size=(2000,2000,3))
	interp_bev_img = np.random.randint(low=0, high=255, size=(2000,2000,3))

	import time
	start = time.time()
	bev_img = remove_hallucinated_content(sparse_bev_img, interp_bev_img, K=41)
	end = time.time()
	duration = end - start
	print(f"Took {duration} sec.")

if __name__ == '__main__':
	test_remove_hallucinated_content()

	test_remove_hallucinated_content_largekernel()

