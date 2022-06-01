
"""Unit tests to make sure data augmentation / preprocessing transforms work correctly."""

import time

import imageio
import matplotlib.pyplot as plt

import salve.utils.transform as transform_utils


def test_PhotometricShiftQuadruplet() -> None:
    """ """
    img_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_07_14/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1583"

    # for i, img in enumerate([image1,image2,image3,image4]):
    #     plt.subplot(2,4,1 + i)
    #     plt.imshow(img)

    for trial in range(100):
    
        # image1 = imageio.imread(f"{img_dir}/pair_5___opening_1_0_rotated_ceiling_rgb_floor_01_partial_room_01_pano_2.jpg")
        # image2 = imageio.imread(f"{img_dir}/pair_5___opening_1_0_rotated_ceiling_rgb_floor_01_partial_room_02_pano_3.jpg")
        # image3 = imageio.imread(f"{img_dir}/pair_5___opening_1_0_rotated_floor_rgb_floor_01_partial_room_01_pano_2.jpg")
        # image4 = imageio.imread(f"{img_dir}/pair_5___opening_1_0_rotated_floor_rgb_floor_01_partial_room_02_pano_3.jpg")

        image1 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_06_pano_13.jpg")
        image2 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_02_pano_11.jpg")
        image3 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_06_pano_13.jpg")
        image4 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_02_pano_11.jpg")

        grid_row1 = np.hstack([image1, image2, image3, image4])

        print(f"Mean before: {image1.mean():.1f}, {image2.mean():.1f}, {image3.mean():.1f}, {image4.mean():.1f}")

        transform = transform_utils.PhotometricShiftQuadruplet()
        image1, image2, image3, image4 = transform(image1, image2, image3, image4)
        # for i, img in enumerate([image1,image2,image3,image4]):
        #     plt.subplot(2,4,1 + 4 + i)
        #     plt.imshow(img)

        print(f"Mean after: {image1.mean():.1f}, {image2.mean():.1f}, {image3.mean():.1f}, {image4.mean():.1f}")

        grid_row2 = np.hstack([image1, image2, image3, image4])
        grid = np.vstack([grid_row1, grid_row2])

        imageio.imwrite(f"photometric_shift_examples/all_types/all_types_{trial}_0.02_grid.jpg", grid)
        #imageio.imwrite(f"photometric_shift_examples/hue/hue_{trial}_0.02_grid.jpg", grid)

        # plt.figure(figsize=(16,10))
        # plt.tight_layout()
        # plt.axis("off")
        # plt.imshow(grid)
        # plt.show()


def test_PhotometricShiftQuadruplet_unmodified() -> None:
    """ """
    img_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_07_14/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1583"

    for trial in range(40):
        np.random.seed(trial)
        print(trial)
        image1 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_06_pano_13.jpg")
        image2 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_02_pano_11.jpg")
        image3 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_06_pano_13.jpg")
        image4 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_02_pano_11.jpg")

        transform = transform_utils.PhotometricShiftQuadruplet(jitter_types=[])
        image1_, image2_, image3_, image4_ = transform(image1, image2, image3, image4)

        assert np.allclose(image1, image1_)
        assert np.allclose(image2, image2_)
        assert np.allclose(image3, image3_)
        assert np.allclose(image4, image4_)

def test_PhotometricShiftQuadruplet_speed():
    """ """

    img_dir = "/Users/johnlam/Downloads/DGX-rendering-2021_07_14/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1583"

    transform = transform_utils.PhotometricShiftQuadruplet(jitter_types=[])

    image1 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_06_pano_13.jpg")
    image2 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_02_pano_11.jpg")
    image3 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_ceiling_rgb_floor_01_partial_room_06_pano_13.jpg")
    image4 = imageio.imread(f"{img_dir}/pair_28___opening_1_0_identity_floor_rgb_floor_01_partial_room_02_pano_11.jpg")

    start = time.time()

    for _ in range(10000):
        image1_, image2_, image3_, image4_ = transform(image1, image2, image3, image4)

    end = time.time()
    duration = end - start
    print(f"Took {duration} sec.")


if __name__ == '__main__':
    #test_PhotometricShift_unmodified()
    #test_PhotometricShift()

    test_PhotometricShift_speed()

