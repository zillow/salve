
"""
Use predicted layout, but use ray to intersect with the layout mesh. 

Total visible geometry prod model -- predictions
Trained with furnished and unfurnished rooms. The post processing is using line regression.
the branch name is discontinuous-vertices.
    https://gitlab.zgtools.net/zillow/rmx/research/floorplanautomation/layout/horizon_net/-/tree/discontinuous-vertices

Alternative: Joint model w/ 3D DWO prediction, but partial room shape

production model recently trained by Ethan. It was trained with L1 loss and trained with furnished rooms and more data. But it’s a partial room shape model, which means the openings are random. We can re-trigger the inference on any old data. But it’s a joint model with 3D DWO prediction. But we have to think of the strategy there to explain the joint model in the paper. @johnlam If you need, we can schedule the re-run on all ZInD data. It might take quite a while

random openings can be ok for snapping, because as long as we can snap at some other door or window in the room, a classification model can predict to ignore the random opening, and still snap together the home (at least that's my thought, we'll see in practice)

Next step: post-processing pipeline to turn room merge boxes into walls
"""


def layout_inference() -> None:
	""" """

	model_name = "discontinuous_vertices_model_0321.pth"

	cmd = "python inference_discontinuous.py"
	cmd += "--pth ../../ckpt/data_nonmadori_zo_discontinuous_depth/best_valid.pth"
	cmd += " --data_dir /bigdata/willhu/floor_map_data/room_shape/datasets/zind_local_all_prepared_for_inference/"
	cmd += " --output_dir /bigdata/willhu/floor_map_data/room_shape/inference_output/zind_local_all_prepared_for_inference_TEST/"
	cmd += " --visualize"

	



if __name__ == "__main__":
	layout_inference()