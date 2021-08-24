# jlambert-auto-floorplan

Code for the auto-Floorplan (AFP) 2021 Internship project within RMX group.

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
Run:
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

Then, install the Python wheel for GTSAM via `pip install /path/to/wheel/file.whl`. Wheels are available [here](https://github.com/borglab/gtsam-manylinux-build/actions/runs/1035308319) under "Artifacts". You must be logged in to github.com in order to be able to download them.  If you're not logged in, it shows you them, but you can't download them.

Next, clone the `HoHoNet` repo.

- We use Facebook's `hydra` library for configuration files.
- We use GTSAM for back-end SLAM/rotation graph optimization, and pose graph alignment.