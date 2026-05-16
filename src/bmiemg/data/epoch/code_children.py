# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .Code import Code



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PhaseCode(Code):
    pass

@dataclass
class ArmCode(Code):
    pass

@dataclass
class TrialCode(Code):
    pass

@dataclass
class MovementCode(Code):
    pass
