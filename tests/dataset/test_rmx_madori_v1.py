"""Unit tests to ensure we can properly merge objects that have been split by the panorama seam."""

import salve.dataset.rmx_madori_v1 as madori_utils
from salve.dataset.rmx_madori_v1 import RmxMadoriV1DWO


def test_merge_wdos_straddling_img_border_windows() -> None:
    """
    On ZinD Building 0000, Pano 17
    """
    windows = []
    windows_merged = madori_utils.merge_wdos_straddling_img_border(wdo_instances=windows)
    assert len(windows_merged) == 0
    assert isinstance(windows_merged, list)


def test_merge_wdos_straddling_img_border_doors() -> None:
    """
    On ZinD Building 0000, Pano 17
    """
    doors = [
        RmxMadoriV1DWO(s=0.14467253176930597, e=0.3704789833822092),
        RmxMadoriV1DWO(s=0.45356793743890517, e=0.46920821114369504),
        RmxMadoriV1DWO(s=0.47702834799608995, e=0.5278592375366569),
        RmxMadoriV1DWO(s=0.5376344086021505, e=0.5865102639296188),
        RmxMadoriV1DWO(s=0.6217008797653959, e=0.8084066471163245),
    ]
    doors_merged = madori_utils.merge_wdos_straddling_img_border(wdo_instances=doors)

    assert doors == doors_merged
    assert len(doors_merged) == 5  # should be same as input


def test_merge_wdos_straddling_img_border_openings() -> None:
    """
    On ZinD Building 0000, Pano 17

    Other good examples are:
    Panos 16, 22, 33 for building 0000.
    Pano 21 for building 0001,
    """
    openings = [
        RmxMadoriV1DWO(s=0.0009775171065493646, e=0.10361681329423265),
        RmxMadoriV1DWO(s=0.9354838709677419, e=1.0),
    ]
    openings_merged = madori_utils.merge_wdos_straddling_img_border(wdo_instances=openings)

    assert len(openings_merged) == 1
    assert openings_merged[0] == RmxMadoriV1DWO(s=0.9354838709677419, e=0.10361681329423265)
