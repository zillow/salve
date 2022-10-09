

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

img_pairs = [
	# (
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_floor_rgb_floor_01_partial_room_20_pano_0.jpg",
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_floor_rgb_floor_01_partial_room_21_pano_30.jpg"
	# ),
	# (
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_ceiling_rgb_floor_01_partial_room_01_pano_5.jpg",
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_ceiling_rgb_floor_01_partial_room_02_pano_11.jpg"
	# ),
	# (
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_floor_rgb_floor_01_partial_room_01_pano_5.jpg",
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_floor_rgb_floor_01_partial_room_02_pano_11.jpg"
	# ),
	# (
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_ceiling_rgb_floor_01_partial_room_20_pano_0.jpg",
	# 	"/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_ceiling_rgb_floor_01_partial_room_21_pano_30.jpg"
	# )
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0767/pair_41___opening_0_1_rotated_floor_rgb_floor_01_partial_room_05_pano_17.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0767/pair_41___opening_0_1_rotated_floor_rgb_floor_01_partial_room_03_pano_20.jpg"
	# ),
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0767/pair_41___opening_0_1_rotated_ceiling_rgb_floor_01_partial_room_03_pano_20.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres/gt_alignment_approx/0767/pair_41___opening_0_1_rotated_ceiling_rgb_floor_01_partial_room_05_pano_17.jpg"
	# )


	# (
	# 	"/Users/johnlam/Downloads/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8/gt_alignment_approx/0339/pair_49___door_3_1_rotated_ceiling_rgb_floor_02_partial_room_02_pano_30.jpg",
	# 	"/Users/johnlam/Downloads/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8/gt_alignment_approx/0339/pair_49___door_3_1_rotated_ceiling_rgb_floor_02_partial_room_12_pano_38.jpg"
	# ),
	# (
	# 	"/Users/johnlam/Downloads/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8/gt_alignment_approx/0339/pair_49___door_3_1_rotated_floor_rgb_floor_02_partial_room_02_pano_30.jpg",
	# 	"/Users/johnlam/Downloads/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8/gt_alignment_approx/0339/pair_49___door_3_1_rotated_floor_rgb_floor_02_partial_room_12_pano_38.jpg"
	# )
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0068/pair_51___door_2_0_rotated_floor_rgb_floor_01_partial_room_14_pano_17.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0068/pair_51___door_2_0_rotated_floor_rgb_floor_01_partial_room_16_pano_19.jpg"
	# )
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0255/pair_38___opening_0_0_identity_floor_rgb_floor_01_partial_room_13_pano_5.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0255/pair_38___opening_0_0_identity_floor_rgb_floor_01_partial_room_13_pano_4.jpg"
	# )
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/incorrect_alignment/0003/pair_10011___door_0_0_identity_floor_rgb_floor_01_partial_room_13_pano_102.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/incorrect_alignment/0003/pair_10011___door_0_0_identity_floor_rgb_floor_01_partial_room_16_pano_38.jpg"
	# )
	# (
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/incorrect_alignment/0003/pair_11023___door_0_0_identity_floor_rgb_floor_01_partial_room_24_pano_3.jpg",
	# 	"/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/incorrect_alignment/0003/pair_11023___door_0_0_identity_floor_rgb_floor_01_partial_room_11_pano_71.jpg"
	# )
	(
		"/Users/johnlambert/Downloads/0715_teaser/pair_11___opening_1_0_rotated_floor_rgb_floor_01_partial_room_09_pano_0.jpg",
		"/Users/johnlambert/Downloads/0715_teaser/pair_11___opening_1_0_rotated_floor_rgb_floor_01_partial_room_04_pano_3.jpg"
	)
]

blended_img_save_fpath = "0715_teaser_pair_11___opening_1_0_rotated_floor_rgb_floor_01_partial_room_09_pano_0_pano3.jpg"
#blended_img_save_fpath = "/Users/johnlam/Downloads/incorrect_alignment_0003_pair_11023___door_0_0_identity_floor_rgb_floor_01_pano3_pano71.jpg"
#blended_img_save_fpath = "/Users/johnlam/Downloads/incorrect_alignment_0003_pair_10011___door_0_0_identity_floor_rgb_floor_01_pano38_pano102.jpg"

# /Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0068/pair_90___door_0_1_rotated_floor_rgb_floor_01_partial_room_07_pano_31.jpg
# /Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout/gt_alignment_approx/0068/pair_90___door_0_1_rotated_floor_rgb_floor_01_partial_room_10_pano_32.jpg 


# blended_img_save_fpath = "/Users/johnlam/Downloads/building_0255_pair_38___opening_0_0_identity_floor_rgb_floor_01_pano4_pano5.jpg"
#blended_img_save_fpath = "/Users/johnlam/Downloads/building_0068_pair_51___door_2_0_rotated_floor_rgb_floor_01_pano17_pano19.jpg"

# blended_img_save_fpath = "/Users/johnlam/Downloads/building_339_pair_49___door_3_1_rotated_floor_rgb_floor_02_pano_30_pano_38.jpg"

# "/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_floor_rgb_floor_01_pano0_pano30_blended.jpg"
# "/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_ceiling_rgb_floor_01_pano5_pano11_blended.jpg"
# "/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0115/pair_10_floor_rgb_floor_01_pano5_pano11_blended.jpg"
# "/Users/johnlam/Downloads/sample_bev_visualizations/old_building_id_0495/pair_6_ceiling_rgb_floor_01_pano0_pano30_blended.jpg"


def convert_black_to_white(img: np.ndarray) -> np.ndarray:
	""" """
	img_white = img.copy()
	img_white[ img[:,:,0] == 0] = 255
	return img_white


def main() -> None:

	blend_images = lambda img1, img2: ((img1.astype(np.float32) + img2.astype(np.float32) ) / 2 ).astype(np.uint8)

	for fpath1, fpath2 in img_pairs:

		img1 = imageio.imread(fpath1)
		img2 = imageio.imread(fpath2)

		# img1_white = convert_black_to_white(img1)
		# img2_white = convert_black_to_white(img2)

		img1_white_fpath = f"{Path(fpath1).parent}/{Path(fpath1).stem}_white_background.jpg"
		img2_white_fpath = f"{Path(fpath2).parent}/{Path(fpath2).stem}_white_background.jpg"

		# imageio.imwrite(img1_white_fpath, img1_white)
		# imageio.imwrite(img2_white_fpath, img2_white)

		blended_img = blend_images(img1, img2)
		# blended_img = blend_images(img1_white, img2_white)

		imageio.imwrite(blended_img_save_fpath, blended_img)

		plt.imshow(blended_img)
		plt.show()
		quit()

		# plt.imshow(img1_white)
		# plt.show()

		# plt.imshow(img2_white)
		# plt.show()



if __name__ == "__main__":
	main()

