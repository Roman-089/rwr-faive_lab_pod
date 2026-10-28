"""Chirp recordings and simulated joint parameters for the servo actuator net.

A recording is a CSV with ``timestamp`` (s), ``commanded_position`` (deg) and ``measured_position`` (deg), one file per
joint, named ``chirp_<joint>_<YYYYMMDD>_<HHMMSS>_ctrl<rate>Hz_...csv`` (written by faive_system
experiments/orca/chirp_characterization.py for ORCA v1/v1b and by rwr_faive_system experiments/orca/chirp.py for v2).
Any control rate works: the command is held between its timestamps (zero-order hold) on the 120 Hz physics grid, and
the simulated position is read at the measured samples' own timestamps, so 40 Hz (v1/v1b) and 20 Hz (v2) data and
jittery loops are all handled the same way.

Joint names: v2 recordings use the v2 names (wrist, thumb_cmc/abd/mcp/dip, index_abd/mcp/pip). The v1/v1b reference
recordings in scripts/motor_param_characterization/2025*/ name the thumb by v1b position (mcp/abd/pip/dip), which
orca_v2.py maps by position in the chain to v2 cmc/abd/mcp/dip; ``naming="auto"`` detects this per folder from the
joint names present (thumb_pip only exists in v1b naming, thumb_cmc only in v2 naming).

The simulated parameters of each joint type (stiffness, damping, friction, armature, effort limit) are read from
source/faive_lab/faive_lab/assets/orca_v2.py without importing Isaac Lab, so a refit of that file is picked up by the
next training run.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCA_V2_PY = REPO_ROOT / "source/faive_lab/faive_lab/assets/orca_v2.py"
CHIRP_DATA_ROOT = REPO_ROOT / "scripts/motor_param_characterization"

PHYSICS_DT = 1.0 / 120.0
"""Physics step of the RL env (OrcaV2ReposeEnvCfg.sim.dt); the actuator runs once per physics step."""

JOINT_TYPES = ("wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip", "index_abd", "index_mcp", "index_pip")
"""The 8 joint types recorded on the hand (RWR26 Task 1, section 7.1). Middle/ring/pinky use the index model."""

V1B_TO_V2 = {"thumb_mcp": "thumb_cmc", "thumb_pip": "thumb_mcp"}
"""v1b thumb names -> v2 (orca_v2.py: v1b mcp/abd/pip/dip -> v2 cmc/abd/mcp/dip; abd and dip keep their names)."""

LINK_INERTIA = {
    "wrist": 1.484e-3,
    "thumb_cmc": 6.747e-5,
    "thumb_abd": 5.192e-5,
    "thumb_mcp": 1.651e-5,
    "thumb_dip": 1.766e-6,
    "index_abd": 6.362e-5,
    "index_mcp": 3.266e-5,
    "index_pip": 4.799e-6,
}
"""Joint-space inertia (kg m^2) of the v2 joint at the default pose: all child links about the joint axis, from the
mass properties in assets/data/orca_v2/orcahand_right.usd (read with the pxr bundled in Isaac Sim 5.1). Only the wrist
is heavy enough for this to matter next to the drive damping; the fingers are effectively massless at 120 Hz."""

# optional tag between joint and date: servo P gain (index_mcp_p1200) or wrist orientation (wrist_horizontal)
_NAME_RE = re.compile(r"chirp_(?P<joint>[a-z_]+?)(?:_(?P<tag>p\d+|horizontal|upright))?_(?P<date>\d{8})_\d{6}")


@dataclass
class Recording:
    """One chirp on one joint, in radians, with the static offset removed."""

    name: str
    path: Path
    joint_type: str
    time: np.ndarray
    """Sample timestamps (s), starting at 0."""
    commanded: np.ndarray
    """Commanded joint position (rad)."""
    measured: np.ndarray
    """Measured joint position (rad), shifted by ``offset`` so that its mean equals the commanded mean."""
    offset: float
    """Removed static offset mean(measured - commanded) (rad): calibration, gravity, tendon pre-tension."""
    ctrl_hz: float

    @property
    def tag(self) -> str:
        return f"{self.joint_type}:{self.path.parent.name[:6]}:{self.path.stem.split('_ctrl')[0][6:]}"


def joint_name_from_file(path: Path) -> tuple[str, str | None]:
    """``chirp_index_mcp_p1200_20250613_...`` -> (``index_mcp``, ``p1200``)."""
    m = _NAME_RE.match(path.name)
    if m is None:
        raise ValueError(f"not a chirp recording name: {path.name}")
    return m.group("joint"), m.group("tag")


def resolve_naming(joint_names: list[str], naming: str) -> str:
    if naming != "auto":
        return naming
    if "thumb_pip" in joint_names:
        return "v1b"
    return "v2"


def load_recording(path: Path, naming: str = "v2") -> Recording:
    """Load one CSV. ``naming`` is ``"v2"`` or ``"v1b"`` (see module docstring)."""
    joint, _ = joint_name_from_file(path)
    joint_type = V1B_TO_V2.get(joint, joint) if naming == "v1b" else joint
    if joint_type not in JOINT_TYPES:
        raise ValueError(f"{path.name}: joint '{joint}' ({naming} naming) is not one of {JOINT_TYPES}")
    df = pd.read_csv(path)
    t = df["timestamp"].to_numpy(float)
    cmd = np.deg2rad(df["commanded_position"].to_numpy(float))
    meas = np.deg2rad(df["measured_position"].to_numpy(float))
    order = np.argsort(t, kind="stable")
    t, cmd, meas = t[order] - t[order][0], cmd[order], meas[order]
    offset = float(np.mean(meas - cmd))
    if abs(offset) > np.deg2rad(20.0):
        print(f"[WARN] {path.name}: static offset {np.rad2deg(offset):.1f} deg removed (encoder zero / wrap?)")
    return Recording(
        name=path.stem,
        path=path,
        joint_type=joint_type,
        time=t,
        commanded=cmd,
        measured=meas - offset,
        offset=offset,
        ctrl_hz=float(1.0 / np.median(np.diff(t))),
    )


def find_recordings(paths: list[str | Path], naming: str = "auto", exclude: tuple[str, ...] = ()) -> list[Recording]:
    """All chirp CSVs in the given folders/files (naming resolved per folder), minus names containing ``exclude``."""
    recordings = []
    for p in map(Path, paths):
        if not p.is_absolute() and not p.exists():
            p = CHIRP_DATA_ROOT / p
        files = sorted(p.glob("chirp_*.csv")) if p.is_dir() else [p]
        files = [f for f in files if not any(x in f.name for x in exclude)]
        if not files:
            raise FileNotFoundError(f"no chirp_*.csv in {p}")
        folder_naming = resolve_naming([joint_name_from_file(f)[0] for f in files], naming)
        recordings += [load_recording(f, folder_naming) for f in files]
    return recordings


def physics_grid(rec: Recording, dt: float = PHYSICS_DT) -> tuple[np.ndarray, np.ndarray]:
    """Command per physics tick (zero-order hold by timestamp) and the tick position of every measured sample.

    Tick n starts at n * dt and applies the last command sent at or before that time. The measured sample k at t_k is
    compared with the simulated position at t_k (position t_k / dt on the tick axis), i.e. before command k acts.
    """
    n_ticks = int(np.floor(rec.time[-1] / dt)) + 1
    idx = np.searchsorted(rec.time, np.arange(n_ticks) * dt, side="right") - 1
    return rec.commanded[np.clip(idx, 0, None)], rec.time / dt


# --- simulated joint parameters (orca_v2.py) ----------------------------------------------------------------------


def _resolve(values: dict | float, joint: str) -> float:
    """Value of a joint in an Isaac Lab per-joint dict (keys are regexes matched with re.fullmatch)."""
    if not isinstance(values, dict):
        return float(values)
    hits = [v for k, v in values.items() if re.fullmatch(k, joint)]
    if len(hits) != 1:
        raise ValueError(f"{joint}: {len(hits)} matches in {list(values)}")
    return float(hits[0])


def read_orca_v2_params(path: Path = ORCA_V2_PY) -> dict[str, dict[str, float]]:
    """Per joint type: stiffness, damping, friction, armature, effort_limit and link inertia of the simulated joint.

    Parses the ImplicitActuatorCfg literals in orca_v2.py (no Isaac Lab import). The index values stand for all fingers.
    """
    tree = ast.parse(Path(path).read_text())
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ImplicitActuatorCfg"
    ]
    if len(calls) != 1:
        raise ValueError(f"expected one ImplicitActuatorCfg in {path}, found {len(calls)}")
    kw = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg != "joint_names_expr"}
    params = {}
    for jt in JOINT_TYPES:
        joint = f"right_{jt}"
        params[jt] = {
            "stiffness": _resolve(kw["stiffness"], joint),
            "damping": _resolve(kw["damping"], joint),
            "friction": _resolve(kw.get("friction", 0.0), joint),
            "armature": _resolve(kw.get("armature", 0.0), joint),
            "effort_limit": _resolve(kw["effort_limit_sim"], joint),
            "inertia": LINK_INERTIA[jt],
        }
    return params
