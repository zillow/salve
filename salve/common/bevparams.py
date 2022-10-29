"""Parameters for creating bird's eye view images."""

import numpy as np

from salve.common.sim2 import Sim2

# the default resolution for rendering BEV images.

# DEFAULT_BEV_IMG_H_PX = 2000
# DEFAULT_BEV_IMG_W_PX = 2000
# DEFAULT_METERS_PER_PX = 0.005

# DEFAULT_BEV_IMG_H_PX = 1000
# DEFAULT_BEV_IMG_W_PX = 1000
# DEFAULT_METERS_PER_PX = 0.01

DEFAULT_BEV_IMG_H_PX = 500
DEFAULT_BEV_IMG_W_PX = 500
DEFAULT_METERS_PER_PX = 0.02


FULL_RES_METERS_PER_PX = 0.005

# at 2000 x 2000 px image @ 0.005 m/px resolution, this thickness makes sense.
FULL_RES_LINE_WIDTH_PX = 30


class BEVParams:
    """Representation of a BEV texture map on a regular grid at a specified resolution.

    For example, 1000 pixels * (0.005 m / px) = 5 meters in each direction.
    """

    def __init__(
        self,
        img_h: int = DEFAULT_BEV_IMG_H_PX,
        img_w: int = DEFAULT_BEV_IMG_W_PX,
        meters_per_px: float = DEFAULT_METERS_PER_PX,
    ) -> None:
        """Construct BEVParams object from image dimensions and resolution of interest.

        Attributes
            img_h: texture map height (in pixels).
            img_w: texture map width (in pixels).
            meters_per_px: resolution, representing the ratio of (#meters/1 pixel) in the grid.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.meters_per_px = meters_per_px

        # Number of pixels in horizontal direction from center.
        h_px = img_w / 2

        # Number of pixels in vertical direction from center.
        v_px = img_h / 2

        # Get grid boundaries in meters.
        xmin_m = -int(h_px * meters_per_px)
        xmax_m = int(h_px * meters_per_px)
        ymin_m = -int(v_px * meters_per_px)
        ymax_m = int(v_px * meters_per_px)

        xlims = [xmin_m, xmax_m]
        ylims = [ymin_m, ymax_m]

        self.xlims = xlims
        self.ylims = ylims

    @property
    def bevimg_Sim2_world(self) -> Sim2:
        """Generate Sim(2) transformation s.t. p_bevimg = bevimg_Sim2_world * p_world.

        Resolution given as #m/px, so we invert it to obtain #px/m, the scale factor.
        Scaling factor from world -> bird's eye view image: #px/m * #meters => #pixels.
        """
        grid_xmin, grid_xmax = self.xlims
        grid_ymin, grid_ymax = self.ylims
        return Sim2(R=np.eye(2), t=np.array([-grid_xmin, -grid_ymin]), s=1 / self.meters_per_px)


def get_line_width_by_resolution(resolution: float) -> int:
    """Compute an appropriate polyline width, in pixels, for a specific rendering resolution.
    Note: this is not dependent upon image size -- solely dependent upon image resolution.
    Can have a tiny image at high resolution.

    Args:
        resolution:

    Returns:
        line_width: line width (thickness) in pixels to use for rendering polylines with OpenCV. Must be an integer.
    """
    scale = resolution / FULL_RES_METERS_PER_PX

    # Larger scale means lower resolution, so we make the line width more narrow.
    line_width = FULL_RES_LINE_WIDTH_PX / scale

    line_width = round(line_width)
    # Must be at least 1 pixel thick.
    return max(line_width, 1)
