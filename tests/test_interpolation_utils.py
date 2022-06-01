"""Unit tests to ensure interpolation is performed correctly."""

import numpy as np

import salve.utils.interpolation_utils as interpolation_utils


def test_interp_dense_grid_from_sparse_collinear() -> None:
    """Ensure we can avoid interpolation when we have points collinear in x, or when all collinear in y.

    Without the check, Scipy would raise the following error:
        scipy.spatial.qhull.QhullError: QH6013 qhull input error: input is less than 3-dimensional
        since all points have the same x coordinate    0
    """
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]

    bev_img = np.zeros((10, 10, 3))
    # provided as (x,y) tuples
    points = np.array([[0, 0], [0, 3], [0, 2], [0, 4]])
    rgb_values = np.array([RED, GREEN, RED, GREEN])
    grid_h = 10
    grid_w = 10
    
    # since all collinear, interpolation is impossible
    dense_grid = interpolation_utils.interp_dense_grid_from_sparse(
        bev_img=bev_img, points=points, rgb_values=rgb_values, grid_h=grid_h, grid_w=grid_w, is_semantics=False
    )
    expected_dense_grid = np.zeros((10, 10, 3))
    assert np.allclose(dense_grid, expected_dense_grid)


    # now, check for points collinear in y.
    bev_img = np.zeros((10, 10, 3))
    points = np.array([[0, 0], [3,0], [2,0], [4,0]])
    # since all collinear, interpolation is impossible
    dense_grid = interpolation_utils.interp_dense_grid_from_sparse(
        bev_img=bev_img, points=points, rgb_values=rgb_values, grid_h=grid_h, grid_w=grid_w, is_semantics=False
    )


def test_interp_dense_grid_from_sparse_insufficient_points_simplex() -> None:
    """Try to interpolate a dense grid using an insufficient number of samples.

    We reproduce:
    scipy.spatial.qhull.QhullError: QH6214 qhull input error: not enough points(2) to construct initial simplex (need 4)
    """
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]

    bev_img = np.zeros((10, 10, 3))
    points = np.array([[1, 1], [5, 5]])
    rgb_values = np.array([RED, GREEN])
    grid_h = 10
    grid_w = 10

    dense_grid = interpolation_utils.interp_dense_grid_from_sparse(
        bev_img=bev_img, points=points, rgb_values=rgb_values, grid_h=grid_h, grid_w=grid_w, is_semantics=False
    )
    assert np.allclose(dense_grid, bev_img)


def test_interp_dense_grid_from_sparse() -> None:
    """ """
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]
    BLUE = [0, 0, 255]
    bev_img = np.zeros((4, 4, 3))

    # provided as (x,y) tuples
    points = np.array([[0, 0], [0, 3], [3, 3], [3, 0]])
    rgb_values = np.array([RED, GREEN, BLUE, RED])
    grid_h = 4
    grid_w = 4

    dense_grid = interpolation_utils.interp_dense_grid_from_sparse(bev_img, points, rgb_values, grid_h, grid_w, is_semantics=False)
    assert isinstance(dense_grid, np.ndarray)
    assert dense_grid.shape == (4, 4, 3)
    # import matplotlib.pyplot as plt
    # plt.imshow(dense_grid)
    # plt.show()



def test_remove_hallucinated_content() -> None:
    """ """
    sparse_bev_img = np.array(
        [
            [0, 2, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    # simulate 3-channel image
    sparse_bev_img = np.stack([sparse_bev_img, sparse_bev_img, sparse_bev_img], axis=-1)

    interp_bev_img = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    # simulate 3-channel image
    interp_bev_img = np.stack([interp_bev_img, interp_bev_img, interp_bev_img], axis=-1)

    bev_img = interpolation_utils.remove_hallucinated_content(sparse_bev_img, interp_bev_img, K=3)
    expected_slice = np.array(
        [
            [1, 2, 3, 4, 5, 0],
            [1, 2, 3, 4, 5, 0],
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    for i in range(3):
        assert np.allclose(bev_img[:, :, i], expected_slice)


def test_remove_hallucinated_content_largekernel() -> None:
    """ """
    sparse_bev_img = np.random.randint(low=0, high=255, size=(2000, 2000, 3))
    interp_bev_img = np.random.randint(low=0, high=255, size=(2000, 2000, 3))

    import time

    start = time.time()
    bev_img = interpolation_utils.remove_hallucinated_content(sparse_bev_img, interp_bev_img, K=41)
    end = time.time()
    duration = end - start
    print(f"Took {duration} sec.")


if __name__ == "__main__":
    #test_remove_hallucinated_content()
    #test_remove_hallucinated_content_largekernel()
    test_interp_dense_grid_from_sparse_collinear()

