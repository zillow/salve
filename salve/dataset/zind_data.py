"""Dataset that reads ZinD data, and feeds it to a Pytorch dataloader."""

import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple, Union

import imageio
from torch.utils.data import Dataset
from torch import Tensor

from salve.dataset.zind_partition import DATASET_SPLITS
from salve.training_config import TrainingConfig


TRAIN_SPLIT_FRACTION = 0.85

# pano 1 layout, pano 2 layout
PathTwoTuple = Tuple[str, str, int]
TensorTwoTupleWithPaths = Tuple[Tensor, Tensor, int, str, str]

# pano 1 floor, pano 2 floor, pano 1 ceiling, pano 2 ceiling
PathFourTuple = Tuple[str, str, str, str, int]
TensorFourTupleWithPaths = Tuple[Tensor, Tensor, Tensor, Tensor, int, str, str]

# pano 1 ceiling, pano 2 ceiling, pano 1 floor, pano 2 floor, pano 1 layout, pano 2 layout
PathSixTuple = Tuple[str, str, str, str, str, str, int]
TensorSixTupleWithPaths = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, str, str]


def pair_idx_from_fpath(fpath: str) -> int:
    """ """
    fname_stem = Path(fpath).stem

    parts = fname_stem.split("___")[0].split("_")
    return int(parts[1])


def pano_id_from_fpath(fpath: str) -> int:
    """Retrieve the panorama ID from a specially-formatted file path.

    After an underscore delimiter, the pano ID is the last part of the file path, before the suffix.
    """
    fname_stem = Path(fpath).stem
    parts = fname_stem.split("_")
    return int(parts[-1])


def get_tuples_from_fpath_list(
    fpaths: List[str], label_idx: int, args: TrainingConfig
) -> List[Union[PathTwoTuple, PathFourTuple, PathSixTuple]]:
    """Given paths for a single floor of single building, extract training/test metadata from the filepaths.

    Note: pair_idx is unique for a (building, floor) but not for a building.

    Args:
        fpaths:
        label_idx: index of ground truth class to associate with 4-tuple
        modalities:

    Returns:
        tuples: list of tuples. If modalities are floor and ceiling, this is (ceiling 1, ceiling 2, floor 1, floor 2).
    """
    # put each file path into a dictionary, to group them
    pairidx_to_fpath_dict = defaultdict(list)

    for fpath in fpaths:
        pair_idx = pair_idx_from_fpath(fpath)
        pairidx_to_fpath_dict[pair_idx] += [fpath]

    tuples = []
    # extract the valid values -- must be a 4-tuple
    for pair_idx, pair_fpaths in pairidx_to_fpath_dict.items():

        if set(args.modalities) == set(["layout"]):
            expected_n_files = 2
        else:
            expected_n_files = 4

        if len(pair_fpaths) != expected_n_files:
            continue

        use_ceiling_texture = set(["ceiling_rgb_texture"]).issubset(set(args.modalities))
        use_floor_texture = set(["floor_rgb_texture"]).issubset(set(args.modalities))

        pair_fpaths.sort()
        if set(args.modalities) == set(["layout"]):

            fp1l, fp2l = pair_fpaths

            pano1_id = pano_id_from_fpath(fp1l)
            pano2_id = pano_id_from_fpath(fp2l)

            assert pano1_id != pano2_id

            assert "_floor_rgb_" in Path(fp1l).name
            assert "_floor_rgb_" in Path(fp2l).name

            assert f"_pano_{pano1_id}.jpg" in Path(fp1l).name
            assert f"_pano_{pano2_id}.jpg" in Path(fp2l).name

        elif use_ceiling_texture or use_floor_texture:

            fp1c, fp2c, fp1f, fp2f = pair_fpaths

            pano1_id = pano_id_from_fpath(fp1c)
            pano2_id = pano_id_from_fpath(fp2c)

            assert pano1_id != pano2_id

            # make sure each tuple is in sorted order (ceiling,ceiling) amd (floor,floor)
            assert "_ceiling_rgb_" in Path(fp1c).name
            assert "_ceiling_rgb_" in Path(fp2c).name

            assert "_floor_rgb_" in Path(fp1f).name
            assert "_floor_rgb_" in Path(fp2f).name

            assert f"_pano_{pano1_id}.jpg" in Path(fp1c).name
            assert f"_pano_{pano2_id}.jpg" in Path(fp2c).name

            assert f"_pano_{pano1_id}.jpg" in Path(fp1f).name
            assert f"_pano_{pano2_id}.jpg" in Path(fp2f).name

            if "layout" in args.modalities:
                # look up in other directory
                fp1l = fp1f.replace(args.data_root, args.layout_data_root)
                fp2l = fp2f.replace(args.data_root, args.layout_data_root)

                # some layout images may be missing
                if not (Path(fp1l).exists() and Path(fp2l).exists()):
                    continue

        if set(args.modalities) == set(["layout"]):
            tuples += [(fp1l, fp2l, label_idx)]

        elif set(args.modalities) == set(["ceiling_rgb_texture"]):
            tuples += [(fp1c, fp2c, label_idx)]

        elif set(args.modalities) == set(["floor_rgb_texture"]):
            tuples += [(fp1f, fp2f, label_idx)]

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture"]):
            tuples += [(fp1c, fp2c, fp1f, fp2f, label_idx)]

        elif set(args.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture", "layout"]):
            tuples += [(fp1c, fp2c, fp1f, fp2f, fp1l, fp2l, label_idx)]

        """
        # make sure nothing corrupted enters the dataset
        try:
            img = imageio.imread(fp0)
            img = imageio.imread(fp1)
            img = imageio.imread(fp2)
            img = imageio.imread(fp3)
        except:
            print("Corrupted in ", fp0, fp1, fp2, fp3)
            continue
        """
    return tuples


def get_available_building_ids(dataset_root: str) -> List[str]:
    """Retrieve names of subdirectories, casted to integer.

    Args:
        dataset_root: path to directory to search under.

    Returns:
        building_ids: list of integers, representing ZInD building subfolder names.
    """
    building_ids = [Path(fpath).stem for fpath in glob.glob(f"{dataset_root}/*") if Path(fpath).is_dir()]
    building_ids = sorted(building_ids, key=lambda x: int(x))
    return building_ids


def make_dataset(
    split: str, data_root: str, args: TrainingConfig
) -> List[Union[PathTwoTuple, PathFourTuple, PathSixTuple]]:
    """Gather a list of all examples within a dataset split.

    Args:
        split: dataset split from which to search for examples under `data_root`.
        data_root: directory where BEV renderings are located.
        args: training hyperparameters, including dataset specification.

    Returns:
        data_list: list of tuples, where each tuple represents file paths + GT labels for a single example.
            Note: is_match = 1 means True.
    """
    if not Path(data_root).exists():
        raise RuntimeError("Dataset root directory does not exist on this machine. Exitting...")

    data_list = []
    logging.info(f"Populating data list for split {split}...")

    # TODO: search from both folders instead
    available_building_ids = get_available_building_ids(dataset_root=f"{data_root}/gt_alignment_approx")

    # We use official ZinD splits.
    split_building_ids = list(set(DATASET_SPLITS[split]).intersection(set(available_building_ids)))

    logging.info(f"{split} split building ids: {split_building_ids}")
    print(f"{split} split building ids: {split_building_ids}")

    label_dict = {"gt_alignment_approx": 1, "incorrect_alignment": 0}  # is_match = True

    for label_name, label_idx in label_dict.items():
        for building_id in split_building_ids:

            logging.info(
                f"\t{label_name}: On building {building_id} -- "
                f"so far, for split {split}, found {len(data_list)} tuples..."
            )

            for floor_id in ["floor_00", "floor_01", "floor_02", "floor_03", "floor_04"]:
                fpaths = glob.glob(f"{data_root}/{label_name}/{building_id}/pair_*___*_rgb_{floor_id}_*.jpg")
                # here, pair_id will be unique

                if len(fpaths) == 0:
                    continue

                tuples = get_tuples_from_fpath_list(fpaths, label_idx, args)
                if len(tuples) == 0:
                    continue
                data_list.extend(tuples)

    logging.info(f"Data list for split {split} has {len(data_list)} tuples.")
    print(f"Data list for split {split} has {len(data_list)} tuples.")
    return data_list


class ZindData(Dataset):
    """Dataloader for loading rendered BEV data from ZInD."""

    def __init__(self, split: str, transform: Callable, args: TrainingConfig) -> None:
        """Initialize dataloader."""

        self.transform = transform
        if set(args.modalities) == set(["layout"]):
            data_root = args.layout_data_root
        else:
            # some RGB
            data_root = args.data_root

        logging.info(f"Base data root is {data_root} for {args.modalities}")

        self.data_list = make_dataset(split, data_root=data_root, args=args)
        self.modalities = args.modalities

    def __len__(self) -> int:
        """Fetch number of examples within a data split."""
        return len(self.data_list)

    def __getitem__(
        self, index: int
    ) -> Union[TensorTwoTupleWithPaths, TensorFourTupleWithPaths, TensorSixTupleWithPaths]:
        """Fetch info associated w/ a single example, either from the train, val, or test set.

        Note: is_match = 1 means True.
        """
        if set(self.modalities) == set(["layout"]):

            x1l_fpath, x2l_fpath, is_match = self.data_list[index]
            x1l = imageio.imread(x1l_fpath)
            x2l = imageio.imread(x2l_fpath)

            x1l, x2l = self.transform(x1l, x2l)
            return x1l, x2l, is_match, x1l_fpath, x2l_fpath

        elif set(self.modalities) == set(["ceiling_rgb_texture"]):
            x1c_fpath, x2c_fpath, is_match = self.data_list[index]
            x1c = imageio.imread(x1c_fpath)
            x2c = imageio.imread(x2c_fpath)
            x1c, x2c = self.transform(x1c, x2c)
            return x1c, x2c, is_match, x1c_fpath, x2c_fpath

        elif set(self.modalities) == set(["floor_rgb_texture"]):
            x1f_fpath, x2f_fpath, is_match = self.data_list[index]
            x1f = imageio.imread(x1f_fpath)
            x2f = imageio.imread(x2f_fpath)
            x1f, x2f = self.transform(x1f, x2f)
            return x1f, x2f, is_match, x1f_fpath, x2f_fpath

        elif set(self.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture"]):
            # floor, then ceiling
            x1c_fpath, x2c_fpath, x1f_fpath, x2f_fpath, is_match = self.data_list[index]

            x1c = imageio.imread(x1c_fpath)
            x2c = imageio.imread(x2c_fpath)
            x1f = imageio.imread(x1f_fpath)
            x2f = imageio.imread(x2f_fpath)
            x1c, x2c, x1f, x2f = self.transform(x1c, x2c, x1f, x2f)
            return x1c, x2c, x1f, x2f, is_match, x1f_fpath, x2f_fpath

        elif set(self.modalities) == set(["ceiling_rgb_texture", "floor_rgb_texture", "layout"]):

            x1c_fpath, x2c_fpath, x1f_fpath, x2f_fpath, x1l_fpath, x2l_fpath, is_match = self.data_list[index]

            x1c = imageio.imread(x1c_fpath)
            x2c = imageio.imread(x2c_fpath)
            x1f = imageio.imread(x1f_fpath)
            x2f = imageio.imread(x2f_fpath)
            x1l = imageio.imread(x1l_fpath)
            x2l = imageio.imread(x2l_fpath)
            x1c, x2c, x1f, x2f, x1l, x2l = self.transform(x1c, x2c, x1f, x2f, x1l, x2l)
            return x1c, x2c, x1f, x2f, x1l, x2l, is_match, x1f_fpath, x2f_fpath

        else:
            raise RuntimeError(f"Unsupported modalities. {str(self.modalities)}")
