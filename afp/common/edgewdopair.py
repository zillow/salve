"""Contains information about a putative alignment hypothesis.
"""

from typing import NamedTuple


class EdgeWDOPair(NamedTuple):
    i1: int
    i2: int
    alignment_object: str
    i1_wdo_idx: int
    i2_wdo_idx: int

    @classmethod
    def from_wdo_pair_uuid(cls, i1: int, i2: int, wdo_pair_uuid: str) -> "EdgeWDOPair":
        """ """
        alignment_object, i1_wdo_idx, i2_wdo_idx = wdo_pair_uuid.split("_")
        return cls(i1=i1, i2=i2, alignment_object=alignment_object, i1_wdo_idx=i1_wdo_idx, i2_wdo_idx=i2_wdo_idx)
