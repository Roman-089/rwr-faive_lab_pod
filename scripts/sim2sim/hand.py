"""MuJoCo stand-in for the real ORCA v2 hand (+ cube): joint targets in, joint states out. No policy logic here.

The interface is what a hand driver offers the policy node: ``set_joint_targets`` (rad, JOINT_NAMES order) and
``joint_pos`` / ``joint_vel``. Time only advances in ``step_physics`` (one Isaac physics step, 1/120 s); the caller
decides the control rate (run_sim2sim.py: 6 physics steps per 20 Hz control step, as decimation 6 in Isaac). A ROS fake
driver wraps the same object: subscribe to the target topic -> set_joint_targets, timer -> step_physics, publish
joint_pos / joint_vel (docs/ablation/sim2sim.md, "Fake ROS hand driver").

The cube (pose, velocity, reset) is accessible for scoring and resets only, as an external tracker would be.
"""

from __future__ import annotations

import numpy as np

import mujoco

import orca_task as task
from scene import SceneCfg, build_mjcf, load_usd_model


class OrcaHandSim:
    def __init__(self, scene_cfg: SceneCfg | None = None, render: bool = False):
        self.cfg = scene_cfg or SceneCfg()
        self.joint_params = task.read_joint_params()
        self.usd = load_usd_model()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg, self.usd, self.joint_params, render=render))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = task.JOINT_NAMES
        jids = [m.joint(n).id for n in task.JOINT_NAMES]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self._vadr = np.array([m.jnt_dofadr[j] for j in jids])
        self._act = np.array([m.actuator(n).id for n in task.JOINT_NAMES])
        self._jids = np.array(jids)
        cube = m.joint("cube")
        self._cube_q, self._cube_v = m.jnt_qposadr[cube.id], m.jnt_dofadr[cube.id]
        self._cube_body = m.body("cube").id
        self._cube_geom = m.geom("cube").id
        self._hand_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, self._cube_body)
                                     and m.geom_contype[g] > 0])
        self._root = m.body("right_forearm").id
        self.n_sub = int(round(task.PHYSICS_DT / m.opt.timestep))
        if not np.isclose(self.n_sub * m.opt.timestep, task.PHYSICS_DT):
            raise ValueError(f"MuJoCo timestep {m.opt.timestep} does not divide the Isaac physics step")
        self.links = list(self.usd["links"])
        self._link_bodies = np.array([m.body(n).id for n in self.links])
        # nominal values, restored by set_physics(PhysicsParams())
        self._nominal = {
            "body_mass": m.body_mass.copy(), "body_inertia": m.body_inertia.copy(), "jnt_range": m.jnt_range.copy(),
            "root_quat": m.body_quat[self._root].copy(),
        }  # fmt: skip
        self.physics = task.PhysicsParams()
        self.set_physics(self.physics)

    # --- hand driver interface ------------------------------------------------------------------------------------

    @property
    def joint_limits(self) -> np.ndarray:
        """(17, 2) current physical limits (rad); DR may move them, the policy side keeps the nominal ones."""
        return self.model.jnt_range[self._jids].copy()

    def set_joint_targets(self, targets: np.ndarray) -> None:
        """PD position targets (rad, JOINT_NAMES order); held until the next call (MuJoCo clamps to ctrlrange)."""
        self.data.ctrl[self._act] = targets

    def step_physics(self) -> None:
        """Advance one Isaac physics step (1/120 s) in ``n_sub`` MuJoCo steps."""
        mujoco.mj_step(self.model, self.data, nstep=self.n_sub)

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self._vadr].copy()

    # --- world / scoring ------------------------------------------------------------------------------------------

    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_q : self._cube_q + 3].copy()

    def cube_quat(self) -> np.ndarray:
        """w x y z, world frame."""
        return self.data.qpos[self._cube_q + 3 : self._cube_q + 7].copy()

    def cube_lin_vel_w(self) -> np.ndarray:
        """Linear velocity of the cube center in the world frame (free joint: qvel[0:3] is already world-frame)."""
        return self.data.qvel[self._cube_v : self._cube_v + 3].copy()

    def cube_ang_vel_w(self) -> np.ndarray:
        """Angular velocity in the WORLD frame (Isaac's root_ang_vel_w). MuJoCo's free-joint qvel[3:6] is in the body
        frame and must be rotated: using it directly would score rotation about the cube's own z axis."""
        r = self.data.xmat[self._cube_body].reshape(3, 3)
        return r @ self.data.qvel[self._cube_v + 3 : self._cube_v + 6]

    def link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """World positions / quaternions (w x y z) of the USD links, in the order of orca_v2_right_usd.json."""
        return self.data.xpos[self._link_bodies].copy(), self.data.xquat[self._link_bodies].copy()

    def reset(self, state: task.ResetState) -> None:
        """Teleport like Isaac's reset events: joints at rest at ``state.joint_pos`` with that as the PD target, cube at
        its pose with its linear velocity, hand mount at ``state.hand_quat``."""
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        m.body_quat[self._root] = state.hand_quat
        d.qpos[self._qadr] = state.joint_pos
        d.ctrl[self._act] = state.joint_pos
        d.qpos[self._cube_q : self._cube_q + 3] = state.cube_pos
        d.qpos[self._cube_q + 3 : self._cube_q + 7] = state.cube_quat
        d.qvel[self._cube_v : self._cube_v + 3] = state.cube_lin_vel
        mujoco.mj_forward(m, d)

    def set_physics(self, p: task.PhysicsParams) -> None:
        """Apply one trial's physics (nominal or DR-sampled) to the model; takes effect from the next step."""
        m, nom = self.model, self._nominal
        self.physics = p
        jp = self.joint_params
        k = jp.stiffness * p.stiffness_scale
        d = jp.damping * p.damping_scale
        m.actuator_gainprm[self._act, 0] = k
        m.actuator_biasprm[self._act, 1] = -k
        m.actuator_biasprm[self._act, 2] = -d
        lim = nom["jnt_range"][self._jids] + p.limit_offset
        m.jnt_range[self._jids] = lim
        m.actuator_ctrlrange[self._act] = lim
        scale = np.ones(len(self.links)) if p.link_mass_scale is None else p.link_mass_scale
        m.body_mass[self._link_bodies] = nom["body_mass"][self._link_bodies] * scale
        m.body_inertia[self._link_bodies] = nom["body_inertia"][self._link_bodies] * scale[:, None]
        s = p.cube_mass / task.CUBE_MASS
        m.body_mass[self._cube_body] = nom["body_mass"][self._cube_body] * s
        m.body_inertia[self._cube_body] = nom["body_inertia"][self._cube_body] * s
        # the cube has priority 1, so its friction is used for every hand-cube contact; hand-hand contacts use the hand's
        m.geom_friction[self._hand_geoms, 0] = p.hand_friction
        m.geom_friction[self._cube_geom, 0] = p.contact_friction
        m.geom_friction[self._cube_geom, 1] = p.contact_friction * self.cfg.cube_torsional_radius
        # derived constants (e.g. the solver's mean inertia) at the nominal mount, so no state leaks between trials
        m.body_quat[self._root] = nom["root_quat"]
        mujoco.mj_setConst(m, mujoco.MjData(m))
