""" """


from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=False)
class TwoViewEstimationReport:
    """TODO(john): improve documentation here.

    Args:
        gt_class: ground truth category, 0 represents negative (WDO mismatch), and 1 represents positive (a WDO match)
        R_error_deg: error in rotation, measured ...
        U_error_deg: error in translation (Unit2), measured in ...
        confidence: scalar-valued confidence in [0,1] range
    """

    gt_class: int
    R_error_deg: Optional[float] = None
    U_error_deg: Optional[float] = None
    confidence: Optional[float] = None
