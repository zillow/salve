"""Runs Shabani et al's source code."""

import glob
import shutil
import time
from pathlib import Path

import cv2
import imageio


width = 1024
height = 512


import subprocess
from typing import Optional, Tuple


def run_command(cmd: str, return_output: bool = False) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Block until system call completes
    Args:
        cmd: string, representing shell command
    Returns:
        Tuple of (stdout, stderr) output if return_output is True, else None
    """
    (stdout_data, stderr_data) = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()

    if return_output:
        return stdout_data, stderr_data
    return None, None


def resize_images(zind_raw_dataset_dir) -> None:
    """Resize panoramas to 1024x512 in (width, height), and convert to PNG file format."""

    building_ids = [
        # "0564",
        # "0519",
        # "1214",
        # "0308",
        # "0438",
        # "0715",
        #"1494"
        "1130"
    ]

    for building_id in building_ids:

        shutil.rmtree(f"dataset/{building_id}/images")
        img_fpaths = glob.glob(f"dataset/{building_id}/images_full_res/*.jpg")
        for img_fpath in img_fpaths:
            print("resizing ", img_fpath)
            img = imageio.imread(img_fpath)
            img_resized = cv2.resize(img, (width, height))

            id = Path(img_fpath).stem.split("_")[-1]
            new_fpath = f"dataset/{building_id}/images/aligned_{id}.png"

            Path(f"dataset/{building_id}/images").mkdir(exist_ok=True, parents=True)
            imageio.imwrite(new_fpath, img_resized)


def time_with_varying_num_images() -> None:
    """ """

    # intentionally choose images with a fully connected structure. from 0715
    # img_fnames = [
    #     "aligned_0.png",
    #     "aligned_2.png",
    #     "aligned_3.png",
    #     "aligned_4.png",
    #     "aligned_7.png",
    #     "aligned_8.png",
    #     "aligned_9.png",
    # ]

    # from 1130, floor 03
    img_fnames = [
        "aligned_6.png", # -> 16
        "aligned_5.png", # -> 15
        "aligned_3.png", # -> 13
        "aligned_7.png", # -> 17
        "aligned_8.png", # -> 18
        "aligned_9.png", # -> 19
    ]



    #building_id = "0715"
    # building_id = "9715"
    building_id = "1130"

    # update test.txt for just this building ID.
    f = open(f"results_summary_{building_id}.txt", "w")

    for num_panos in [4,5,6]: # 7
        # clear the old dataset dir
        if Path("dataset").exists():
            shutil.rmtree("dataset")
        # clear the old outputs dir.
        if Path("outputs").exists():
            shutil.rmtree("outputs")

        Path(f"dataset/{building_id}/images").mkdir(exist_ok=True, parents=True)

        # copy over the images
        for img_fname in img_fnames[:num_panos]:
            src = f"dataset_full/{building_id}/images/{img_fname}"
            dst = f"dataset/{building_id}/images/{img_fname}"
            shutil.copyfile(src, dst)

        # save the start time
        start = time.time()

        # execute run.sh tee, send output to file.
        cmd = f"bash run.sh 2>&1 | tee shabani_stdout_output_{num_panos}_panos_{building_id}.log"
        run_command(cmd)

        # save the end time
        end = time.time()
        duration = end - start
        print(f"Took {duration:.2f} sec. to complete for {num_panos} panos.")
        f.write(f"Took {duration:.2f} sec. to complete for {num_panos} panos.\n")

        # count the number of files in the `outputs` dir for this building id.
        num_hypotheses = len(list(Path(f"outputs/test/{building_id}").glob("*.png")))
        print(f"Generated {num_hypotheses} hypotheses for {num_panos} panos.")
        f.write(f"Generated {num_hypotheses} hypotheses for {num_panos} panos.\n")
        f.flush()

        cmd = f"mv outputs outputs_{num_panos}_panos_{building_id}"
        run_command(cmd)

    f.close()


if __name__ == "__main__":
    zind_raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"
    #resize_images(zind_raw_dataset_dir)
    time_with_varying_num_images()

