
**SALVe: Semantic Alignment Verification for Floorplan Reconstruction from Sparse Panoramas** (ECCV 2022, Official Repo) [PDF]() <br>
[John Lambert](https://johnwlambert.github.io/), Yuguang Li, Ivaylo Boyadzhiev, Lambert Wixson, Manjunath Narayana, Will Hutchcroft, [James Hays](https://faculty.cc.gatech.edu/~hays/), [Frank Dellaert](https://dellaert.github.io/), [Sing Bing Kang](http://www.singbingkang.com/). <br>
Presented at ECCV 2022.

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

**Libraries for evaluation** We use `AvgMeter` classes from `mseg-semantic`. Install as follows:

```bash
git clone https://github.com/mseg-dataset/mseg-semantic.git
cd mseg-semantic
pip install -e .
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
Download and unzip the custom `ModifiedHorizonNet` predictions from [Google Drive here](https://drive.google.com/file/d/16JuxBgg368RL7dSpPjF9kKocLbo2ekAt/view?usp=sharing).



```bash
unzip {PATH_TO_HORIZONNET PREDS ZIP FILE}/ZInD_HorizonNet_predictions.tar.gz
ZILLOW_HORIZON_NET_ZIND_PREDICTIONS_DIRPATH = {}
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
Replace `--split test` with `train` or `val` if desired.

If you see an error message like:
```
HoHoNet's lib could not be loaded, skipping...
Exception:  No module named 'lib'
```
then you have not configured `HoHoNet` properly above.

### Run SALVe Verifier Inference
Next, we'll send the pairs of BEV texture maps to the model for scoring. You should have pass the folder where the model checkpoint is stored as `model_save_dirpath`. You can create this yourself (if performing inference with a pretrained model), or it would be automatically created during training.

```bash
python scripts/test.py --gpu_ids {COMMA SEPARATED GPU ID LIST} \
    --model_results_dir {PATH TO FOLDER CONTAINING TRAINED MODEL}\
    --config_fpath {PATH TO YAML MODEL CONFIG} \
    --serialization_save_dir {PATH WHERE SERIALIZED PREDS WILL BE SAVED TO}
```
Please note that the YAML config must be the path to a config in `salve/configs/*`.

For example, if your model checkpoint was stored in a directory accessible at:
```
/data/johnlam/models_for_lambert/2021_11_19_21_42_11
```
and contained pytorch model checkpoint (.pth) `train_ckpt.pth`, then we would use via CLI
```
--model_results_dir /data/johnlam/models_for_lambert/2021_11_19_21_42_11
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

- Custom HorizonNet W/D/O & Layout w/ photometric info:
- Rasterized layout only:
- GT WDO:

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
pytest tests --cov salve --ignore tests/test_export_alignment_hypotheses.py
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

Q: Why are Sim(2) objects used all over the place?
A:

Q: How are vanishing points used in SALVe?
A:

Q: How was the Modified HorizonNet (MHNet) trained? How does it differ from HorizonNet?
A:

straightenings , same line segments
line segments, buckets, xy, z are the main buckets
yaw angle w.r.t. the upright z-axis, 

left edge of the image to the first vanishing point
vanishing angle


(add a figure / illustration).


How many extra furnished homes were used for training Madori.
For legal reasons, we cannot release this dataset.
We have pretrained it, we are providing the results.
Yuguang will ask Ethan about this. 


ModifiedHorizonNet
MHNet
Output is different.

## Other TODOS

On DGX, you can find `ZInD` stored here:
`/mnt/data/johnlam/zind_bridgeapi_2021_10_05`

- Keep your training config and inference config in different files.  (Or different sections of the same file.)

Additional Notes:
- Batch 1 of Madori predictions: [here, on Google Drive](https://drive.google.com/drive/folders/1A7N3TESuwG8JOpx_TtkKCy3AtuTYIowk?usp=sharing)
(in batch 1 of the predictions, it looks like new_home_id matched to floor_map_guid_new. in batch 2, that matches to floormap_guid_prod)

No shared texture between (0,75) -- yet doors align it (garage to kitchen)





