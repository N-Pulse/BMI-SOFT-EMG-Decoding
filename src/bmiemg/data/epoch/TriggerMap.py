# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .code_children import (
    PhaseCode,
    ArmCode,
    TrialCode,
    MovementCode,
    TargetMovementCode
)



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TriggerMap:
    special_triggers: dict[int, str]
    phase_code: PhaseCode
    arm_code: ArmCode
    trial_code: TrialCode
    mov_code: MovementCode
    target_code: TargetMovementCode

    @property
    def movement_id(self) -> tuple[int, int]:
        """Get (pos, key)"""
        key = next(k for k, v in self.phase_code.code_dict.items() if v == "move")
        pos = self.phase_code.code_pos

        return (pos, key)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Default maps
# ──────────────────────────────────────────────────────
V1_TRIGGER_MAP: TriggerMap = TriggerMap(
    special_triggers = {
        9701: "resting state, eyes open",
        9702: "resting state, eyes closed",
        8888: "start LabRecorder",
        9999: "a test marker (to test befiore the experiment start)",
        8899: "experiment finished",
    },
    phase_code = PhaseCode(
        code_dict = {
            1: "cue",
            2: "prep",
            3: "move",
            4: "return",
            5: "iti",
        },
        code_pos = 0
    ),
    arm_code = ArmCode(
        code_dict = {
            1: "Left",
            2: "Right",
        },
        code_pos = 1
    ),
    trial_code = TrialCode(
        code_dict = {
            1:	"Palm or fist up",
            2:	"Palm or fist side",
            3:	"Palm or fist down",
        },
        code_pos = 2
    ),
    mov_code = MovementCode(
        code_dict = {
            1:	"openhand_slow_3sec",
            2:	"openhand_fast_1sec",
            3:	"closetofist_allfingerstogether_slow_3sec",
            4:	"closetofist_allfingerstogether_fast_1sec",
            5:	"closetofist_allfingerstogether_normalforce_3sec / closetofist_allfingerstogether_3sec_normalforce",
            6:	"closetofist_allfingerstogether_3sec_maxforce",
            7:	"open_fourfingers_together_3sec",
            8:	"close_fourfingers_together_3sec",
            9:	"open_onlythumb_3sec",
            10:	"close_onlythumb_3sec",
            11:	"wrist_palmarflexion_maxforce_3sec",
            12:	"wrist_palmarflexion_normalforce_3sec",
            13:	"wrist_dorsiflexion_normalforce_3sec",
            14:	"wrist_dorsiflexion_maxforce_3sec",
            15:	"grasp_cup_3sec",
            16:	"grasp_donut_3sec",
            17:	"grasping_pinching_pen_allfingers_3sec / grasping_pinching_pen_2fingers_3sec",
            18:	"close_pinky_3sec",
            19:	"close_ring_3sec",
            20:	"close_middle_3sec",
            21:	"close_index_3sec",
            22:	"close_thumb_3sec",
            23:	"open_pinky_3sec",
            24:	"open_ring_3sec",
            25:	"open_middle_3sec",
            26:	"open_index_3sec",
            27:	"open_thumb_3sec",
        },
        code_pos = 3
    ),
    target_code = TargetMovementCode(
        {
            "grasp": [3, 4, 5, 6, 8, 15, 16],
            "wrist": [11, 12, 13, 14],
            "pinch": [18, 19, 20, 21, 22]
        }
    )
)
