
import glob
from multiprocessing import Pool

import imageio.v2 as imageio

"""
/srv/scratch/jlambert30/salve/2022_09_29_zind_texture_map_renderings_test/incorrect_alignment/1169/pair_614___window_0_0_identity_ceiling_rgb_floor_02_partial_room_09_pano_45.jpg


/srv/scratch/jlambert30/salve/2022_09_29_zind_texture_map_renderings_test/incorrect_alignment/1169/pair_614___window_0_0_identity_ceiling_rgb_floor_02_partial_room_09_pano_45.jpg`` with iomode `ri`.


Based on the extension, the following plugins might add capable backends:
  pyav:  pip install imageio[pyav] /srv/scratch/jlambert30/salve/2022_09_29_zind_texture_map_renderings_test/incorrect_alignment/1169/pair_614___window_0_0_identity_ceiling_rgb_floor_02_partial_room_09_pano_45.jpg
NOT_FOR_RELEASE/find_corrupt_files.py:16: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  imageio.imread(fpath)


"""

def read_image(i: int, num_images: int, fpath: str) -> None:
    """ """
    if i % 10000 == 0:
        print(f"On {i}/{num_images}")
    try:
        imageio.imread(fpath)
    except Exception as e:
        print("Corrupt: ", e, fpath)


def main(bev_save_root: str) -> None:
    """ """
    fpaths = glob.glob(f"{bev_save_root}/**/*.jpg", recursive=True)
    num_images = len(fpaths)

    args = []
    for i, fpath in enumerate(fpaths):
        args += [(i, num_images, fpath)]

    num_processes = 20
    with Pool(num_processes) as p:
        p.starmap(read_image, args)


if __name__ == "__main__":

    bev_save_root = "/srv/scratch/jlambert30/salve/2022_09_29_zind_texture_map_renderings_test"
    main(bev_save_root=bev_save_root)



