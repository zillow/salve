
import imageio
import time

HOME_DIR = "/data/johnlam"
#HOME_DIR = "/mnt/data/johnlam"


def main():
	fpath = f"{HOME_DIR}/ZinD_BEV_RGB_only_2021_07_14_v3/gt_alignment_approx/1427/pair_283___door_2_0_identity_ceiling_rgb_floor_01_partial_room_17_pano_53.jpg"

	fpath = "/Users/johnlam/Downloads/jlambert-auto-floorplan/pair_283___door_2_0_identity_ceiling_rgb_floor_01_partial_room_17_pano_53.jpg"

	start = time.time()

	for _ in range(1000):
		img = imageio.imread(fpath)

	end = time.time()
	duration = end - start
	print(f"Took {duration} sec.")


if __name__ == "__main__":
	main()
