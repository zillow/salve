

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(hypotheses_save_root):
	""" """
	traj_distance_dict = defaultdict(list)

	# by distance within the capture trajectory

	building_ids = [Path(fpath).stem for fpath in glob.glob(f"{hypotheses_save_root}/*") if Path(fpath).is_dir()]
	building_ids.sort()

	for building_id in building_ids:

		floor_ids = [Path(fpath).stem for fpath in glob.glob(f"{hypotheses_save_root}/{building_id}/*") if Path(fpath).is_dir()]
		for floor_id in floor_ids:

			for label_type in ["gt_alignment_approx", "gt_alignment_exact", "incorrect_alignment"]: # 

				json_fpaths = glob.glob(f"{hypotheses_save_root}/{building_id}/{floor_id}/{label_type}/*.json")
				#import pdb; pdb.set_trace()

				for json_fpath in json_fpaths:
					i, j = Path(json_fpath).stem.split('_')[:2]
					i, j = int(i), int(j)
					abs_dist = np.absolute(i - j)

					traj_distance_dict[label_type] += [abs_dist]


	for i, (label_type, array) in enumerate(traj_distance_dict.items()):

		plt.subplot(1,3,i+1)
		plt.hist(array, bins=np.arange(50)) 
		plt.title(label_type)
	#plt.show()
	plt.savefig("capture_order_histogram_2021_08_25.jpg")




if __name__ == "__main__":

	#hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_07_11_alignment_hypotheses_2021_08_04_Sim3"
	hypotheses_save_root = "/mnt/data/johnlam/ZinD_07_11_alignment_hypotheses_2021_08_04_Sim3"

	main(hypotheses_save_root)