""" TODO: ADD DOCSTRING """

from salve.stitching.models.feature2d import Feature2dU


class WallFeature:
    def __init__(self, start: Feature2dU, end: Feature2dU, type: str) -> None:
        """TODO

        Args:
            start: TODO
            end: TODO
            type: TODO
        """
        self.start = start
        self.end = end
        self.type = type
