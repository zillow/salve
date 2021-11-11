
"""Recursively find corrupted images in a directory using many workers."""

import glob
from multiprocessing import Pool

import imageio


def verify_img(i: int, fpath: str) -> None:
	""" """
	if i % 5000 == 0:
		print(f"\tOn {i}")
	try:
		imageio.imread(fpath)
	except:
		print(f"Failed on {fpath}")


def find_corrupted_images(raw_dataset_dir: str) -> None:
	""" """
	fpaths = glob.glob(f"{raw_dataset_dir}/**/*.png", recursive=True)
	print("Found ", len(fpaths))

	args = [(i, fpath) for i, fpath in enumerate(fpaths)]
	num_processes = 30

	with Pool(num_processes) as p:
		p.starmap(verify_img, args)


if __name__ == '__main__':

	#raw_dataset_dir = "/mnt/data/johnlam/ZinD_07_11_BEV_RGB_only_2021_08_04_layoutimgs"
	#raw_dataset_dir = "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_06_25"
	raw_dataset_dir = "/home/johnlam/ZinD_Bridge_API_HoHoNet_Depth_Maps"
	find_corrupted_images(raw_dataset_dir)