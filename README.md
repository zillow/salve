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
    - `models`: CNN model architectures
    - `utils`: utilities for rasterization, rendering graphs, Sim(3) alignment, precision/recall computation, and more.


# Installation
Run:
```
pip install -e .
```