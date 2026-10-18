# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ORCA v2 dexterous hand.
the control values are copied from ORCA v1b and should be revised for v2.
the thumb joints are mapped by position in the chain, not by name: v1b mcp/abd/pip/dip -> v2 cmc/abd/mcp/dip.
the USD file is in data/orca_v2
"""

from faive_lab import FAIVE_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

ORCA_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FAIVE_ASSETS_DATA_DIR}/orca_v2/orcahand_right.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            # was 3.0 against finger-squeeze blow-ups; 1000 (as v1b) trains better
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_wrist",
                "right_thumb_cmc",
                "right_thumb_abd",
                "right_thumb_mcp",
                "right_thumb_dip",
                "right_index_abd",
                "right_index_mcp",
                "right_index_pip",
                "right_middle_abd",
                "right_middle_mcp",
                "right_middle_pip",
                "right_ring_abd",
                "right_ring_mcp",
                "right_ring_pip",
                "right_pinky_abd",
                "right_pinky_mcp",
                "right_pinky_pip",
            ],
            stiffness={
                "^right_(index|middle|ring|pinky)_abd$": 9.7,
                "^right_(index|middle|ring|pinky)_mcp$": 11.8,
                "^right_(index|middle|ring|pinky)_pip$": 14.4,
                "right_thumb_cmc": 14.1,
                "right_thumb_abd": 15.,
                "right_thumb_mcp": 3.5,
                # v1b: characterization response for thumb_dip was not that good, just copy the pip values
                "right_thumb_dip": 3.5,
                "right_wrist": 12.0,
            },
            damping={
                "^right_(index|middle|ring|pinky)_abd$": 0.74,
                "^right_(index|middle|ring|pinky)_mcp$": 1.2,
                "^right_(index|middle|ring|pinky)_pip$": 2.05,
                "right_thumb_cmc": 1.76,
                "right_thumb_abd": 1.37,
                "right_thumb_mcp": 0.42,
                "right_thumb_dip": 0.42,
                "right_wrist": 0.85,
            },
            # v1b's sysid values halved: PhysX scales joint friction with joint load, and with the full values v2's joints
            # tracked much worse than v1b's (abduction ~half the motion, pinky_abd stuck). Provisional until v2 is characterised.
            friction={
                "^right_(index|middle|ring|pinky)_abd$": 0.039,
                "^right_(index|middle|ring|pinky)_mcp$": 0.003,
                "^right_(index|middle|ring|pinky)_pip$": 0.00035,
                "right_thumb_cmc": 0.,
                "right_thumb_abd": 0.0155,
                "right_thumb_mcp": 0.,
                "right_thumb_dip": 0.,
                "right_wrist": 0.105,
            },
            armature={
                "^right_(index|middle|ring|pinky)_abd$": 0.,
                "^right_(index|middle|ring|pinky)_mcp$": 0.,
                "^right_(index|middle|ring|pinky)_pip$": 0.,
                "right_thumb_cmc": 0.,
                "right_thumb_abd": 0.,
                "right_thumb_mcp": 0.,
                "right_thumb_dip": 0.,
                "right_wrist": 0.0,
            },
            effort_limit_sim={
                "^right_(index|middle|ring|pinky)_abd$": 0.66,
                "^right_(index|middle|ring|pinky)_mcp$": 0.78,
                "^right_(index|middle|ring|pinky)_pip$": 1.29,
                "right_thumb_cmc": 1.09,
                "right_thumb_abd": 0.58,
                "right_thumb_mcp": 1.95,
                "right_thumb_dip": 1.95,
                "right_wrist": 1.95,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Orca v2 Hand robot."""
