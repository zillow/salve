
**SALVe: Semantic Alignment Verification for Floorplan Reconstruction from Sparse Panoramas** (ECCV 2022, Official Repo) <br>
[[PDF]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910632.pdf) [[Supp. Mat]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910632-supp.pdf) <br>
[John Lambert](https://johnwlambert.github.io/), Yuguang Li, Ivaylo Boyadzhiev, Lambert Wixson, Manjunath Narayana, Will Hutchcroft, [James Hays](https://faculty.cc.gatech.edu/~hays/), [Frank Dellaert](https://dellaert.github.io/), [Sing Bing Kang](http://www.singbingkang.com/). <br>
Presented at ECCV 2022. Link to [SALVe video](https://www.youtube.com/watch?v=WBOVn0LC7dI&feature=emb_imp_woyt) (5 min).

# Repository Structure

Code is organized under the `salve` directory as follows:
- `salve`:
    - `algorithms`: graph and SfM related algorithms, e.g. rotation averaging, Pose(2) SLAM, spanning tree computation, cycle consistency
    - `baselines`: wrappers around other SfM implementations (OpenSfM, OpenMVG) that we use as baseline comparisons.
    - `common`: shared classes for room layout data, W/D/O objects, and 2d pose graphs
    - `configs`: Hydra config files for model training
    - `dataset`: dataloaders
    - `models`: CNN model architectures, implemented in Pytorch
    - `stitching`: room layout stitching
    - `utils`: utilities for rasterization, rendering graphs, Sim(3) alignment, precision/recall computation, and more.

## Installation
To install the `salve` library, clone the repository and run:
```
git clone https://github.com/johnwlambert/salve.git
cd salve/
pip install -e .
```

Set `SALVE_REPO_DIRPATH` to wherever you have cloned the `salve` repo.

# Dependencies

We use Conda to manage dependencies. Please install the environment on Linux using:
```bash
conda env create -f environment_ubuntu-latest.yml
```
or on Mac
```bash
conda env create -f environment_macos-latest.yml
```

**Notes about libraries used**
- We use Facebook's `hydra` library for configuration files.
- We use the [`rdp`](https://github.com/fhirschmann/rdp) library for polygon simplification.
- We use the [`GTSAM`](https://github.com/borglab/gtsam) library for back-end SLAM/pose graph optimization.
- We use the [`GTSFM`](https://github.com/borglab/gtsfm) library for some geometry-related functionality, including pose graph alignment.

Note: Hydra, rdp, GTSAM and GTSFM are all included in the Conda environment.

## Verify Installation

Run the unit tests:
```bash
pytest tests/
```

## Download ZInD

Register on [Bridge API](https://bridgedataoutput.com/register/zgindoor) and request access. Then copy the server token and run the following commands:

```bash
git clone https://github.com/zillow/zind.git
cd zind
python download_data.py --num_process 10 \
    --output_folder {SAVE_DIR} \
    --server_token {BRIDGE_API_SERVER_TOKEN} \
    --verbose
```

For example, `SAVE_DIR` could be `/mnt/data/johnlam/zind_bridgeapi_2021_10_05`, and `BRIDGE_API_SERVER_TOKEN` could be a 32-character alphanumeric sequence. More instructions can be found in the [ZInD Github repo](https://github.com/zillow/zind#registration-for-download-request).

## Running Reconstruction

Make sure you are within the conda environment (`salve-v1`).

**Download ModifiedHorizonNet (MHNet) predictions.**
Download and unzip the custom `ModifiedHorizonNet` predictions from [Google Drive here](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/data/ZInD_HorizonNet_predictions.tar.gz).



```bash
unzip {PATH_TO_HORIZONNET PREDS ZIP FILE}/ZInD_HorizonNet_predictions.tar.gz
ZILLOW_HORIZON_NET_ZIND_PREDICTIONS_DIRPATH = {}
```

**Vanishing angle file Extraction.**
Run the following command:
```
python scripts/split_vanishing_angle_file.py --csv assets/zind_vanishing_angles.csv --out {PATH_TO_PREDICTION_DIRECTORY}/vanishing_angle
```


### Generate alignment hypotheses
Run SALVe model inference by first generating alignment hypotheses:
```bash
python scripts/export_alignment_hypotheses.py \
    --num_processes {NUM. DESIRED PROCS.} \
    --raw_dataset_dir {PATH TO ZIND} \
    --hypotheses_save_root {DIRECTORY WHERE TO DUMP OUTPUT} \
    --wdo_source horizon_net \
    --split test \
    --mhnet_predictions_data_root {DIRECTORY TO MHNET PREDS} \
     2>&1 | tee alignment_hypotheses_generation_output.log
```
Using 20-30 processes is recommended, and even with 30 processes, the generation may take 1-2 hours to complete.

Replace `--split test` with `train` or `val` if desired.

### Generate depth maps with HoHoNet
To run HoHoNet inference, clone the HoHoNet repo:
```bash
cd ..
git clone https://github.com/sunset1995/HoHoNet.git
cd HoHoNet
```
Now, download the HoHoNet model by executing:
```bash
./{SALVE_REPO_DIRPATH}/scripts/download_monodepth_model.sh
```
You should now see a model checkpoint at:
```bash
ls -ltrh HoHoNet/ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth
```
From within the `HoHoNet` dir, execute:
```bash
python {SALVE_REPO_DIRPATH}/scripts/batch_hohonet_inference.py \
    --raw_dataset_dir {PATH TO ZIND} \
    --depth_save_root {PATH TO SAVE DEPTH MAPS} \
    --num_processes {NUM GPU PROCESSES BASED ON GPU RAM}
```
Each process will likely consume around 4.3 GB of GPU RAM.

### Render BEV texture maps.
Run:
```bash
export PYTHONPATH=./
python {SALVE_REPO_DIRPATH}/scripts/render_dataset_bev.py \
    --num_processes {NUM. DESIRED PROCESSES} \
    --raw_dataset_dir {PATH TO ZIND} \
    --hypotheses_save_root {PATH TO PRE-GENERATED ALIGNMENT HYPOTHESES} \
    --depth_save_root {PATH TO WHERE DEPTH MAPS WILL BE OR HAVE BEEN SAVED TO} \
    --bev_save_root {PATH WHERE BEV TEXTURE MAPS WILL BE SAVED TO} \
    --split test
```
Replace `--split test` with `train` or `val` if desired. If you wish to test out SALVe on only a single building, then instead of using the `--split` CLI arg, use the `--building_id` arg instead, e.g. `--building_id 0001`.

If you see an error message like:
```
HoHoNet's lib could not be loaded, skipping...
Exception:  No module named 'lib'
```
then you have not configured `HoHoNet` properly above.

### Run SALVe Verifier Inference
Next, we'll send the pairs of BEV texture maps to the model for scoring. You should have pass the folder where the model checkpoint is stored as `model_save_dirpath`. You can create this yourself (if performing inference with a pretrained model), or it would be automatically created during training.

```bash
python scripts/test.py \
    --gpu_ids {COMMA SEPARATED GPU ID LIST} \
    --model_ckpt_fpath {PATH TO TRAINED PYTORCH MODEL CHECKPOINT} \
    --config_name {FILE NAME OF YAML MODEL CONFIG} \
    --serialization_save_dir {PATH WHERE SERIALIZED PREDS WILL BE SAVED TO}
```
Please note that the YAML config must be the name of a config under `salve/configs/*`.

For example, if your model checkpoint was stored in a directory accessible at:
```
/data/johnlambert/models/2021_11_19_21_42_11
```
and contained pytorch model checkpoint (.pth) `train_ckpt.pth`, then we would use via CLI
```
--model_ckpt_fpath /data/johnlambert/models/2021_11_19_21_42_11/train_ckpt.pth
```
In the config you'll see a line:
```
    data_root: /data/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres
```
this should be replaced with `bev_save_root` where renderings were saved to, above. When running `scripts/test.py`, the YAML’s `model_save_dirpath` will be ignored, since this is only used at training time.

Please note that `serialization_save_dir` is a new directory created at inference time, where predictions on the val or test set will be cached as JSON.

### Run Global SfM
Now, pass the front-end measurements to SfM:
```bash
python scripts/run_sfm.py \
     --raw_dataset_dir {PATH TO ZIND} \
     --method pgo \
     --serialized_preds_json_dir {PATH TO WHERE SERIALIZED PREDS WERE SAVED TO} \
     --hypotheses_save_root {PATH TO PRE-GENERATED ALIGNMENT HYPOTHESES}
```
Above, we use pose-graph optimization (`pgo`) as the global aggregation method for relative poses.

## Pretrained Models
We release 7 pretrained models:

| W/D/O + Layout Source | Input Modalities     | #Tours used for training | Arch. | Model filename (md5sum) |
| :-------------------: | :------------------: | :----------------------: | :---: | :--------------------: |
| MHNet                 | Ceiling + Floor RGB  | 435   | ResNet-152 | [1200ffbe47d836557d88fef052952337.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/1200ffbe47d836557d88fef052952337.pth) |
| MHNet                 | Ceiling + Floor RGB  | 587   | ResNet-152 | [9fcbb628bd5efffbdcc4ce55a9eb380d.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/9fcbb628bd5efffbdcc4ce55a9eb380d.pth) |
| MHNet                 | Ceiling RGB only     | 587   | ResNet-152 | [5c64123c134b829dd99beb3684582f61.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/5c64123c134b829dd99beb3684582f61.pth) |
| MHNet                 | Floor RGB only       | 587   | ResNet-152 | [a063532031f83aec97289466943bf52d.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/a063532031f83aec97289466943bf52d.pth) |
| MHNet                 | Rasterized Layout (Floor)| 877  | ResNet-152 | [6ac3f3e5fe6fa3d4bfae7c124d7787b3.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/6ac3f3e5fe6fa3d4bfae7c124d7787b3.pth) |
| GT W/D/O + GT Layout  | Ceiling + Floor RGB | 350    | ResNet-152 | [301f920ec795b9966aebc2367544d234.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/301f920ec795b9966aebc2367544d234.pth) |
| GT W/D/O + GT Layout  | Ceiling + Floor RGB | 817    | ResNet-152 | [b1198bad27aecb8a19f884abc920a731.pth](https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/models/b1198bad27aecb8a19f884abc920a731.pth) |


## Training a model

```bash
python scripts/train.py \
    --gpu_ids {COMMA SEPARATED GPU ID LIST} \
    --config_name {FILE NAME OF YAML CONFIG under salve/configs/}
```

A directory will be created which contains the config file used for training (yaml), and a pytorch model checkpoint (.pth), and JSON results, e.g.:
```
 train_ckpt.pth
 results-2021_11_19_21_42_11-2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.json
 2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.yaml
```

## Floor Map Stitching

```bash
python scripts/stitch_floor_plan.py --output-dir output_folder \
    --path-clusters ../tests/test_data/example_input_stiching/cluster_pred.json \
    --pred-dir ../tests/test_data/example_input_stiching/pano/ \
    --path-gt ../tests/test_data/example_input_stiching/floor_map_gt.json
```


## Run Static Analysis of Codebase

```bash
flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503,E731 salve
```

```bash
pytest tests --cov salve
coverage report
```

## Running SfM Baselines

In the paper, we compare with [OpenMVG](https://github.com/openMVG/openMVG) and [OpenSfM](https://github.com/mapillary/OpenSfM). You can run these baselines yourself as follows.

**OpenSfM**: First, clone the repository:
```bash
git clone --recursive https://github.com/mapillary/OpenSfM
cd OpenSfM
```
Now, build the library/binaries, as described [here](https://opensfm.org/docs/building.html).

Next, inside the `OpenSfM` directory, now run:
```bash
python {SALVE_REPO_DIRPATH}/salve/scripts/execute_opensfm.py \
   --raw_dataset_dir {PATH_TO_ZIND} \
   --opensfm_repo_root {TODO} \
   --overrides_fpath {TODO}
```

Next, evaluate the OpenSfM results:
```bash
python {SALVE_REPO_DIRPATH}/scripts/evaluate_sfm_baseline.py \
    --algorithm_name opensfm \
    --raw_dataset_dir {PATH_TO_ZIND} \
    --results_dir {PATH TO WHERE OPENSFM SCRIPT DUMPED RECONSTRUCTION RESULTS} \
    --save_dir {WHERE TO SAVE VISUALIZATIONS/RESULT SUMMARIES }
```

**OpenMVG**:

Follow the instructions [here](https://github.com/openMVG/openMVG/blob/develop/BUILD.md) on how to clone and build the repository.

After compilation, run the following script:
```bash
python scripts/execute_openmvg.py \
    --raw_dataset_dir {PATH_TO_ZIND} \
    --openmvg_sfm_bin {PATH_TO_DIRECTORY_CONTAINING_COMPILED_BINARIES} \
    --openmvg_demo_root {TODO}
```

Evaluate the OpenMVG results using:
```bash
python {SALVE_REPO_DIRPATH}/scripts/evaluate_sfm_baseline.py \
    --algorithm_name openmvg \
    --raw_dataset_dir {PATH_TO_ZIND} \
    --results_dir {PATH TO WHERE OPENMVG SCRIPT DUMPED RECONSTRUCTION RESULTS} \
    --save_dir {WHERE TO SAVE VISUALIZATIONS/RESULT SUMMARIES }
```


## FAQ

Q: For Open3d dependencies, I see `OSError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.27' not found` upon import?

A: You are on Ubuntu <=16.04, but you should be on Ubuntu >=18.04 (see [here](https://github.com/isl-org/Open3D/issues/4349)).

Q: How can I see some examples of ZInD's annotated floor plans?

A: `python scripts/viz_zind_annotated_floorplans`

Q: How was the Modified HorizonNet (MHNet) trained? How does it differ from HorizonNet?

A: This model was trained on a combination of ZInD data and data from additional furnished homes. Its outputs differ from HorizonNet. For legal reasons, we cannot release this dataset. We will not release the MHNet model weights, but we do provide the inference results on ZInD. See `horizon_net_schema.json` for the file format of predictions (one JSON file per panorama).

Q: How are vanishing points used in SALVe?

A: SALVe generates relative pose hypotheses for a given pair of panoramas by snapping the mid-points of identified doors or windows pairs from the predicted room shape of each panorama. The snapping is done in xy direction (horizontal plane), with the relative yaw angle defined by snapping the opposite outward normals of the door / window line segments. The computed panorama yaw angle is then refined by aligning the horizontal vanishing points of 2 panoramas.

Q: How can I visualize a loss plot, given a training log (JSON file) generated during training?

A: Run `python scripts/visualize_loss_plot.py`.
