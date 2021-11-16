



import argoverse.utils.subprocess_utils as subprocess_utils

from afp.dataset.zind_partition import DATASET_SPLITS


MEGATRON_HOSTNAME = "johnlam@172.22.152.131"
MEGATRON_DATAROOT = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"
SE1_TRANSFER_DIR = "/data/johnlam/test1"


def main() -> None:
	""" """

	for label_type in ["gt_alignment_approx", "incorrect_alignment"]:

		for building_id in DATASET_SPLITS["test"]:

			cmd = f"rsync -rvz --ignore-existing {MEGATRON_HOSTNAME}:{MEGATRON_DATAROOT}/{label_type}/{building_id} {SE1_TRANSFER_DIR}/{label_type}/{building_id}"
			print(cmd)
			#subprocess_utils.run_command(cmd)


if __name__ == "__main__":
	main()