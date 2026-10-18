# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for continuous rotation with given angular velocity."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os 
from datetime import datetime 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import ContinuousRotationCommandCfg
class ContinuousRotationCommand(CommandTerm):
    """Command term that generates continuous rotation commands with given angular velocity.

    This command term generates angular velocity commands for continuous rotation of the object
    in the robot's hand. The command includes target position and target angular velocity.

    Command format: [pos_x, pos_y, pos_z, ang_vel_x, ang_vel_y, ang_vel_z] = 6 dimensions

    Design philosophy:
    - Position command: tells the robot where the object should stay
    - Angular velocity command: tells the robot how fast the object should rotate
    - Let the robot learn how to achieve the target angular velocity autonomously
    """
    cfg: ContinuousRotationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: ContinuousRotationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the continuous rotation command generator.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Get the object to manipulate
        self.object: RigidObject = env.scene[cfg.asset_name]

        # === Position command: keep object at fixed position ===
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command = self.object.data.default_root_state[:, :3] + init_pos_offset
        
        # === Angular velocity command: target rotation speed ===
        self.angular_vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        
        # === Performance metrics ===
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["current_angular_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_angular_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_velocity_magnitude_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_velocity_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_velocity_alignment"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["total_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vertical_drift"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["linear_speed"] = torch.zeros(self.num_envs, device=self.device)
        

        # # Add: Used to control printing frequency
        # self._log_counter = 0
        # self._log_interval = 200 # Print once every 200 updates
        # self._setup_logging()


    def __str__(self) -> str:
        """Return string description of the command generator."""
        msg = "ContinuousRotationCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tAngular velocity range: {self.cfg.angular_vel_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose and angular velocity in the environment frame. Shape is (num_envs, 6).
        
        Command format: [pos_x, pos_y, pos_z, ang_vel_x, ang_vel_y, ang_vel_z]
        - [0:3]: target position (x, y, z)
        - [3:6]: target angular velocity (wx, wy, wz)
        """
        return torch.cat((self.pos_command, self.angular_vel_command), dim=-1)
    
    def _update_metrics(self):
        """Update performance metrics for logging and monitoring."""
        current_pos = self.object.data.root_pos_w - self._env.scene.env_origins
        current_ang_vel = self.object.data.root_ang_vel_w
        current_lin_vel = self.object.data.root_lin_vel_w
        target_ang_vel = self.angular_vel_command
        # 1. position_error
        self.metrics["position_error"] = torch.norm(
            current_pos - self.pos_command, dim=1
        )
        # 2. angular_velocity_magnitude_error
        current_magnitude = torch.norm(current_ang_vel, p=2, dim=-1)
        target_magnitude = torch.norm(target_ang_vel, p=2, dim=-1)
        # self.metrics["angular_velocity_magnitude_error"] = torch.abs(
        #     current_magnitude - target_magnitude
        # )
        self.metrics["angular_velocity_magnitude_error"] = current_magnitude - target_magnitude
        # 3.Angular velocity direction alignment
        cosine_sim = torch.nn.functional.cosine_similarity(
            current_ang_vel,
            target_ang_vel,
            dim=-1,
            eps=1e-8
        )
        self.metrics["angular_velocity_alignment"] = cosine_sim
        
        # 4. Total angular velocity error (L2 norm)
        self.metrics["angular_velocity_error"] = torch.norm(
            current_ang_vel - target_ang_vel, p=2, dim=-1
        )
        # 5. For monitoring - current and target speed
        self.metrics["current_angular_speed"] = current_magnitude
        self.metrics["target_angular_speed"] = target_magnitude
        # 6.overall success
        position_success = self.metrics["position_error"] < self.cfg.position_success_threshold
        magnitude_success = self.metrics["angular_velocity_magnitude_error"] < self.cfg.angular_vel_magnitude_threshold
        direction_success = self.metrics["angular_velocity_alignment"] > self.cfg.angular_vel_direction_threshold
        overall_success = position_success & magnitude_success & direction_success
        self.metrics["consecutive_success"] = torch.where(
            overall_success,
            self.metrics["consecutive_success"] + 1,
            torch.zeros_like(self.metrics["consecutive_success"])
        )
        # 7.Total number of successes (for success rate calculation)
        if not hasattr(self, '_total_success_initialized'):
            self.metrics["total_success"] = torch.zeros(
                self._env.num_envs, device=self.device
            )
            self._total_success_initialized = True
        
        self.metrics["total_success"] = torch.where(
            overall_success,
            self.metrics["total_success"] + 1,
            self.metrics["total_success"]  # 失败时保持不变
        )
        # 8. vertical deviation
        self.metrics["vertical_drift"] = torch.abs((current_pos - self.pos_command)[:, 2])
        # 9. linear velocity
        lin_speed = torch.norm(current_lin_vel, p=2, dim=-1)
        self.metrics["linear_speed"] = lin_speed
        
        # # === Logging (NEW) ===
        # self._log_counter += 1
        # if self._log_counter >= self._log_interval:
        #     self._log_counter = 0
        #     self._print_and_save_angular_velocity_info(current_ang_vel)
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample angular velocity commands for specified environments.
        
        Args:
            env_ids: List of environment IDs that need new angular velocity targets.
        """
        # Speed range: 60%-100% of maximum
        MIN_RATIO = 0.6
        MAX_RATIO = 1.0

        # Scale to configured velocity range
        vel_range = torch.tensor(self.cfg.angular_vel_range, device=self.device)
        # Generate random ratio within the range
        rand_ratio = torch.rand((len(env_ids), 3), device=self.device)
        rand_ratio = MIN_RATIO + rand_ratio * (MAX_RATIO - MIN_RATIO)
        # Calculate speed magnitude
        speed_magnitude = rand_ratio * vel_range
        # Random direction (+1 or -1)
        direction = torch.sign(torch.rand((len(env_ids), 3), device=self.device) - 0.5)
        self.angular_vel_command[env_ids] = speed_magnitude * direction
    
    def _update_command(self):
        """Update command state - simplified version relies on time-based resampling."""
        # This function is intentionally left empty for the simplified version.
        # Unlike the complex version that updates target orientation every frame,
        # this simplified version keeps the angular velocity target stable until
        # the parent class triggers resampling based on the configured time range.
        pass

    def _setup_logging(self):
        """Setup logging files for angular velocity monitoring"""
        # Create log directory
        log_dir = "logs/angular_velocity"
        os.makedirs(log_dir, exist_ok=True)
    
        # Generate log filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = os.path.join(log_dir, f"angular_vel_{timestamp}.txt")
        self._csv_file = os.path.join(log_dir, f"angular_vel_{timestamp}.csv")
    
        # Write text log header
        with open(self._log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Angular Velocity Monitoring Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environments: {self.num_envs}\n")
            f.write(f"Velocity range: {self.cfg.angular_vel_range}\n")
            f.write("="*80 + "\n\n")

        # Write CSV header - includes environment 0 and 1
        with open(self._csv_file, 'w') as f:
            header = "update,"
            # Environment 0
            header += "env0_tgt_tot,env0_tgt_x,env0_tgt_y,env0_tgt_z,"
            header += "env0_cur_tot,env0_cur_x,env0_cur_y,env0_cur_z,env0_err,"
            # Environment 1
            header += "env1_tgt_tot,env1_tgt_x,env1_tgt_y,env1_tgt_z,"
            header += "env1_cur_tot,env1_cur_x,env1_cur_y,env1_cur_z,env1_err,"
            # Statistics data
            header += "mean_tgt_tot,std_tgt_tot,"
            header += "mean_tgt_x,std_tgt_x,"
            header += "mean_tgt_y,std_tgt_y,"
            header += "mean_tgt_z,std_tgt_z,"
            header += "mean_cur_tot,std_cur_tot,"
            header += "mean_cur_x,std_cur_x,"
            header += "mean_cur_y,std_cur_y,"
            header += "mean_cur_z,std_cur_z,"
            header += "mean_err,std_err,success_rate\n"
            f.write(header)

        print(f"✅ Logging: {self._log_file}")
        print(f"✅ CSV: {self._csv_file}")
    
    def _print_and_save_angular_velocity_info(self, current_ang_vel: torch.Tensor):
        """Print and save angular velocity information to console and files
        
        Args:
            current_ang_vel: Current angular velocity tensor (num_envs, 3)
        """
        target_ang_vel = self.angular_vel_command
        
        # Calculate total magnitudes for statistics (use absolute values)
        target_total = torch.norm(target_ang_vel, dim=-1)
        target_x_abs = torch.abs(target_ang_vel[:, 0])
        target_y_abs = torch.abs(target_ang_vel[:, 1])
        target_z_abs = torch.abs(target_ang_vel[:, 2])
        
        current_total = torch.norm(current_ang_vel, dim=-1)
        current_x_abs = torch.abs(current_ang_vel[:, 0])
        current_y_abs = torch.abs(current_ang_vel[:, 1])
        current_z_abs = torch.abs(current_ang_vel[:, 2])
        
        error = self.metrics["angular_vel_error"]
        success_rate = (error < self.cfg.angular_vel_success_threshold).float().mean().item() * 100
        
        # Get real values (with sign) for environment 0 and 1
        # Environment 0
        env0_target_x = target_ang_vel[0, 0].item()
        env0_target_y = target_ang_vel[0, 1].item()
        env0_target_z = target_ang_vel[0, 2].item()
        env0_current_x = current_ang_vel[0, 0].item()
        env0_current_y = current_ang_vel[0, 1].item()
        env0_current_z = current_ang_vel[0, 2].item()
        
        # Environment 1
        env1_target_x = target_ang_vel[1, 0].item()
        env1_target_y = target_ang_vel[1, 1].item()
        env1_target_z = target_ang_vel[1, 2].item()
        env1_current_x = current_ang_vel[1, 0].item()
        env1_current_y = current_ang_vel[1, 1].item()
        env1_current_z = current_ang_vel[1, 2].item()
        
        # Save to CSV file
        with open(self._csv_file, 'a') as f:
            f.write(f"{self._log_interval},")
            
            # Environment 0 data (real values with sign)
            f.write(f"{target_total[0]:.6f},")
            f.write(f"{env0_target_x:.6f},{env0_target_y:.6f},{env0_target_z:.6f},")
            f.write(f"{current_total[0]:.6f},")
            f.write(f"{env0_current_x:.6f},{env0_current_y:.6f},{env0_current_z:.6f},")
            f.write(f"{error[0]:.6f},")
            
            # Environment 1 data (real values with sign)
            f.write(f"{target_total[1]:.6f},")
            f.write(f"{env1_target_x:.6f},{env1_target_y:.6f},{env1_target_z:.6f},")
            f.write(f"{current_total[1]:.6f},")
            f.write(f"{env1_current_x:.6f},{env1_current_y:.6f},{env1_current_z:.6f},")
            f.write(f"{error[1]:.6f},")
            
            # Statistics data (absolute values)
            f.write(f"{target_total.mean():.6f},{target_total.std():.6f},")
            f.write(f"{target_x_abs.mean():.6f},{target_x_abs.std():.6f},")
            f.write(f"{target_y_abs.mean():.6f},{target_y_abs.std():.6f},")
            f.write(f"{target_z_abs.mean():.6f},{target_z_abs.std():.6f},")
            
            f.write(f"{current_total.mean():.6f},{current_total.std():.6f},")
            f.write(f"{current_x_abs.mean():.6f},{current_x_abs.std():.6f},")
            f.write(f"{current_y_abs.mean():.6f},{current_y_abs.std():.6f},")
            f.write(f"{current_z_abs.mean():.6f},{current_z_abs.std():.6f},")
            
            f.write(f"{error.mean():.6f},{error.std():.6f},{success_rate:.2f}\n")
        
        # Format text output
        text = "\n" + "="*80 + "\n"
        text += f"📊 Angular Velocity Monitor - Update #{self._log_interval}\n"
        text += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += "="*80 + "\n"
        
        # Environment 0 details (with sign to show direction)
        text += f"\n🔍 Environment 0:\n"
        text += f"   🎯 Target: Total={target_total[0]:.4f} | "
        text += f"X={env0_target_x:+.4f} | Y={env0_target_y:+.4f} | Z={env0_target_z:+.4f} rad/s\n"
        text += f"   📈 Current: Total={current_total[0]:.4f} | "
        text += f"X={env0_current_x:+.4f} | Y={env0_current_y:+.4f} | Z={env0_current_z:+.4f} rad/s\n"
        text += f"   ❌ Error: {error[0]:.4f} rad/s\n"
        
        # Environment 1 details (with sign to show direction)
        text += f"\n🔍 Environment 1:\n"
        text += f"   🎯 Target: Total={target_total[1]:.4f} | "
        text += f"X={env1_target_x:+.4f} | Y={env1_target_y:+.4f} | Z={env1_target_z:+.4f} rad/s\n"
        text += f"   📈 Current: Total={current_total[1]:.4f} | "
        text += f"X={env1_current_x:+.4f} | Y={env1_current_y:+.4f} | Z={env1_current_z:+.4f} rad/s\n"
        text += f"   ❌ Error: {error[1]:.4f} rad/s\n"
        
        # Statistics across all environments (absolute values)
        text += f"\n📊 Statistics (all {self.num_envs} envs):\n"
        text += f"   Target  Total → {target_total.mean():.4f}±{target_total.std():.4f} | "
        text += f"X={target_x_abs.mean():.4f}±{target_x_abs.std():.4f} | "
        text += f"Y={target_y_abs.mean():.4f}±{target_y_abs.std():.4f} | "
        text += f"Z={target_z_abs.mean():.4f}±{target_z_abs.std():.4f}\n"
        
        text += f"   Current Total → {current_total.mean():.4f}±{current_total.std():.4f} | "
        text += f"X={current_x_abs.mean():.4f}±{current_x_abs.std():.4f} | "
        text += f"Y={current_y_abs.mean():.4f}±{current_y_abs.std():.4f} | "
        text += f"Z={current_z_abs.mean():.4f}±{current_z_abs.std():.4f}\n"
        
        text += f"   Error → {error.mean():.4f}±{error.std():.4f} rad/s\n"
        text += f"   Success Rate → {success_rate:.1f}%\n"
        text += "="*80 + "\n\n"
        
        # Save to text file and print to console
        with open(self._log_file, 'a') as f:
            f.write(text)
        print(text, end='')
    
    # === Debug visualization methods ===

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization implementation - single marker type for both arrows.
        
        Args:
            debug_vis: Whether to enable visualization.
        """
        if debug_vis:
            if not hasattr(self, "target_arrow_visualizer"):
                target_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/target_arrows",
                    markers={
                        "arrow": sim_utils.ConeCfg(
                            radius=0.06,
                            height=0.12,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                        )
                    },
                )
                self.target_arrow_visualizer = VisualizationMarkers(target_cfg)
            if not hasattr(self, "current_arrow_visualizer"):
                current_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/current_arrows",
                    markers={
                        "arrow": sim_utils.ConeCfg(
                            radius=0.06,
                            height=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                        )
                    },
                )
                self.current_arrow_visualizer = VisualizationMarkers(current_cfg)
            if not hasattr(self, "frame_visualizer"):
                frame_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/frames",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.02, 0.02, 0.02),
                        ),
                    },
                )
                self.frame_visualizer = VisualizationMarkers(frame_cfg)
            self.target_arrow_visualizer.set_visibility(True)
            self.current_arrow_visualizer.set_visibility(True)
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_arrow_visualizer"):
                self.target_arrow_visualizer.set_visibility(False)
            if hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer.set_visibility(False)
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        """Simplified visualization: show all arrows with one marker type.
    
        Args:
            event: Render event object (usually not used directly).
        """
        if not hasattr(self, "target_arrow_visualizer"):
            return
    
        # Get object position and angular velocities
        target_pos = self.pos_command + self._env.scene.env_origins
        current_ang_vel = self.object.data.root_ang_vel_w
        target_ang_vel = self.angular_vel_command
    
        # Calculate magnitudes
        target_magnitude = torch.norm(target_ang_vel, dim=-1, keepdim=True)
        current_magnitude = torch.norm(current_ang_vel, dim=-1, keepdim=True)
    
        # Arrow base position
        # show the marker in a static position above the object
        offset = torch.tensor([0.0, 0.0, 0.15], device=self.device)

        target_base_pos = target_pos + offset
        current_base_pos = target_pos + offset
    
        # Convert angular velocities to arrow orientations
        target_orientations = self._angular_velocity_to_orientation(target_ang_vel)
        current_orientations = self._angular_velocity_to_orientation(current_ang_vel)

        # Calculate arrow lengths
        min_length = 0.03
        scale = 0.05
        target_length = min_length + target_magnitude * scale
        current_length = min_length + current_magnitude * scale

        target_scales = torch.cat([
            torch.ones_like(target_magnitude) * 0.04,  # width (x)
            torch.ones_like(target_magnitude) * 0.04,  # width (y)
            target_length,  # length (z)
        ], dim=-1)
    
        current_scales = torch.cat([
            torch.ones_like(current_magnitude) * 0.04,  # width (x)
            torch.ones_like(current_magnitude) * 0.04,  # width (y)
            current_length,  # length (z)
        ], dim=-1)
    
        self.target_arrow_visualizer.visualize(
            translations=target_base_pos,
            orientations=target_orientations,
            scales=target_scales,
        )
        self.current_arrow_visualizer.visualize(
            translations=current_base_pos,
            orientations=current_orientations,
            scales=current_scales,
        )
        self.frame_visualizer.visualize(
            translations=target_pos,
        )


    def _angular_velocity_to_orientation(self, angular_velocity):
        """Convert angular velocity vector to arrow orientation.
    
        Args:
            angular_velocity: Angular velocity vector (num_envs, 3)
        
        Returns:
            orientations: Arrow quaternion orientations (num_envs, 4) [x, y, z, w]
        """
        # Get angular velocity magnitude
        ang_vel_norm = torch.norm(angular_velocity, dim=-1, keepdim=True)
    
        # For zero angular velocity, use default upward direction
        zero_vel_mask = ang_vel_norm.squeeze(-1) < 1e-6
    
        # Normalize angular velocity vector (rotation axis direction)
        ang_vel_unit = torch.where(
            zero_vel_mask.unsqueeze(-1),
            torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1),
            angular_velocity / torch.clamp(ang_vel_norm, min=1e-6)
        )
    
        # Default arrow direction (pointing up)
        default_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
    
        # Calculate rotation axis (cross product)
        rotation_axis = torch.cross(default_dir, ang_vel_unit, dim=-1)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True)
    
        # Calculate rotation angle (dot product)
        cos_angle = torch.sum(default_dir * ang_vel_unit, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
        # Handle parallel vectors
        parallel_mask = rotation_axis_norm.squeeze(-1) < 1e-6
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
    
        # Calculate rotation quaternion for non-parallel vectors
        angle = torch.acos(cos_angle)
        half_angle = angle / 2.0
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)
    
        rotation_axis_normalized = rotation_axis / torch.clamp(rotation_axis_norm, min=1e-6)
    
        quat_xyz = rotation_axis_normalized * sin_half_angle
        quat_w = cos_half_angle
    
        calculated_quat = torch.cat([quat_xyz, quat_w], dim=-1)
    
        # Use identity quaternion for parallel vectors
        result_quat = torch.where(
            parallel_mask.unsqueeze(-1),
            identity_quat,
            calculated_quat
        )
    
        return result_quat

