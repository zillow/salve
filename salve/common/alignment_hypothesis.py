"""Class definition and utility for relative pose hypotheses (from a W/D/O alignment)."""

from typing import List, NamedTuple

from salve.common.sim2 import Sim2

class AlignmentHypothesis(NamedTuple):
    """Represents a relative pose hypothesis between two panoramas.

    Attributes:
        i2Ti1: relative pose.
        wdo_alignment_object: either 'door', 'window', or 'opening'
        i1_wdo_idx: this is the WDO index for Pano i1 (known as i)
        i2_wdo_idx: this is the WDO index for Pano i2 (known as j)
        configuration: either identity or rotated
    """

    i2Ti1: Sim2
    wdo_alignment_object: str
    i1_wdo_idx: int
    i2_wdo_idx: int
    configuration: str


def prune_to_unique_sim2_objs(possible_alignment_info: List[AlignmentHypothesis]) -> List[AlignmentHypothesis]:
    """
    Only useful for GT objects, that might have exact equality? (confirm how GT can actually have exact values)
    """
    pruned_possible_alignment_info = []

    for j, alignment_hypothesis in enumerate(possible_alignment_info):
        is_dup = any(
            [
                alignment_hypothesis.i2Ti1 == inserted_alignment_hypothesis.i2Ti1
                for inserted_alignment_hypothesis in pruned_possible_alignment_info
            ]
        )
        # has not been used yet
        if not is_dup:
            pruned_possible_alignment_info.append(alignment_hypothesis)

    num_orig_objs = len(possible_alignment_info)
    num_pruned_objs = len(pruned_possible_alignment_info)

    verbose = False
    if verbose:
        logger.info(f"Pruned from {num_orig_objs} to {num_pruned_objs}")
    return pruned_possible_alignment_info

