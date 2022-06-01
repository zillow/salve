
"""Unit tests on OpenSfM."""

from types import SimpleNamespace

import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
from gtsam import Cal3Bundler

import salve.baselines.opensfm as opensfm_utils


SKYDIO_32_FNAMES = [
    "S1014644.JPG",
    "S1014645.JPG",
    "S1014646.JPG",
    "S1014647.JPG",
    "S1014648.JPG",
    "S1014649.JPG",
    "S1014650.JPG",
    "S1014651.JPG",
    "S1014652.JPG",
    "S1014653.JPG",
    "S1014654.JPG",
    "S1014655.JPG",
    "S1014656.JPG",
    "S1014684.JPG",
    "S1014685.JPG",
    "S1014686.JPG",
    "S1014687.JPG",
    "S1014688.JPG",
    "S1014689.JPG",
    "S1014690.JPG",
    "S1014691.JPG",
    "S1014692.JPG",
    "S1014693.JPG",
    "S1014694.JPG",
    "S1014695.JPG",
    "S1014696.JPG",
    "S1014724.JPG",
    "S1014725.JPG",
    "S1014726.JPG",
    "S1014734.JPG",
    "S1014735.JPG",
    "S1014736.JPG",
    ]


def test_measure_opensfm_localization_accuracy() -> None:
    """Check that poses are decoded correctly from a Skydio-32 crane mast sequence."""

    # TODO: write unit test
    reconstruction_json_fpath = "/Users/johnlam/Downloads/OpenSfM/data/skydio-32/reconstruction.json"
    reconstructions = opensfm_utils.load_opensfm_reconstructions_from_json(reconstruction_json_fpath)

    for r, reconstruction in enumerate(reconstructions):

        # point_cloud = np.zeros((0,3))
        # rgb = np.zeros((0,3))

        point_cloud = reconstruction.points
        rgb = reconstruction.rgb

        wTi_list = [reconstruction.pose_dict[fname] if fname in reconstruction.pose_dict else None for fname in SKYDIO_32_FNAMES]
        N = len(wTi_list)

        fx = reconstruction.camera.focal * 1000
        px = reconstruction.camera.width / 2
        py = reconstruction.camera.height / 2

        calibrations = [Cal3Bundler(fx=fx, k1=0, k2=0, u0=px, v0=py)] * N
        args = SimpleNamespace(**{"point_rendering_mode": "point"})
        open3d_vis_utils.draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)


if __name__ == "__main__":
    test_measure_opensfm_localization_accuracy()

