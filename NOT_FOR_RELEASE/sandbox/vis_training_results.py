

import matplotlib.pyplot as plt

from argoverse.utils.json_utils import read_json_file

def main(json_results_fpath):
	""" """

	data = read_json_file(json_results_fpath)

	train_mAcc = data["train_mAcc"]
	val_mAcc = data["val_mAcc"]

	plt.plot( range(len(val_mAcc)), val_mAcc, color="g", label="Val mAcc")
	plt.plot( range(len(train_mAcc)), train_mAcc, color="r", label="Train mAcc")

	plt.xlabel("Training Epoch")
	plt.ylabel("Mean Accuracy")

	plt.legend()
	plt.show()



if __name__ == "__main__":

	json_results_fpath = "/Users/johnlam/Downloads/ZinD_trained_models_2021_08_06/2021_08_07_12_36_04/results-2021_08_07_12_36_04-2021_08_06_resnet50_ceiling_floor_layout.json"
	main(json_results_fpath)