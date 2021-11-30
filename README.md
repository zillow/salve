# jlambert-auto-floorplan

Code for the Auto-Floorplan (AFP) 2021 Internship project within RMX group.

Author: John Lambert

# Repository Structure

Code is organized under the `afp` directory as follows:
- `afp`:
    - `algorithms`: graph and SfM related algorithms, e.g. rotation averaging, Pose(2) SLAM, spanning tree computation, cycle consistency
    - `common`: shared classes for room layout data, WDO objects, and 2d pose graphs
    - `configs`: Hydra config files for model training
    - `dataset`: dataloaders
    - `models`: CNN model architectures, implemented in Pytorch
    - `utils`: utilities for rasterization, rendering graphs, Sim(3) alignment, precision/recall computation, and more.


# Installation
To install the Auto-Floorplan (`afp`) library, run:
```
pip install -e .
```

# Dependencies

We use Conda to manage dependencies. Please install the environment on Linux using:
```bash
conda env create -f environment_linux.yml
```
or on Mac
```bash
conda env create -f environment_mac.yml
```

We use the [`GTSAM`](https://github.com/borglab/gtsam) library for back-end SLAM/pose graph optimization. Pre-built python wheels for `GTSAM` are available for download [here](https://github.com/borglab/gtsam-manylinux-build/actions/runs/1126472520) on Github under "Artifacts" (there is no need to build GTSAM from scratch). The GTSAM libraries on `pypi` are out of date. You must be logged in to github.com in order to be able to download them.  If you're not logged in, it shows you them, but you can't download them.

Then, install the Python wheel for GTSAM via 
```
pip install /path/to/wheel/file.whl
```

**Rendering Training/Testing Data** If you'd like to render training or testing data, clone the `HoHoNet` repo.

**Libraries for evaluation** We use `AvgMeter` classes from `mseg-semantic`. Install as follows:

```bash
git clone https://github.com/mseg-dataset/mseg-semantic.git
pip install -e .
```

**Notes about libraries used**
- We use Facebook's `hydra` library for configuration files.
- We use the [`rdp`](https://github.com/fhirschmann/rdp) library for polygon simplification.

## Running Reconstruction


Run SALVe model inference by:
```bash
python scripts/test.py --gpu_ids {COMMA SEPARATED GPU ID LIST}
```

Now, pass the front-end measurements to SfM:
```bash
python run_sfm.py
```

## Pretrained Models

- Custom HorizonNet W/D/O & Layout w/ photometric info:
- Rasterized layout only:
- GT WDO:

## Training a model

```bash
python scripts/train.py --gpu_ids {COMMA SEPARATED GPU ID LIST}
```
