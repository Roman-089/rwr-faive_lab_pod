# Subtask 2: exteroception (cube pose from the camera tracker)

Task sheet §2.3 / §8.2–8.4: add the cube position and orientation to the policy observation, make the simulated values
look like the tracker's (noise, lag, lost detections), and feed the same terms to the policy node.

Relation to the team's scaffold: `orca_v2_repose_cube_env_cfg.py` (task `-cube-v0`, owner E, 2026-10-26) appends
`object_pos` (3) and `object_quat` (4) with placeholder noise to S0's default task, with the tracker model left as TODO.
We keep its order (74 proprio, then position, then quaternion) and add the tracked flag last, on the 06 / rank-1 recipe
tasks (our ablation cfg); the scaffold file is not edited.

## Observation (policy 74 → 82)

`tracker_obs.TrackedCubePose`, appended by `add_cube_tracker_obs` (orca_v2_repose_ablation_env_cfg.py):

| dims | term | real side (policy node) |
|---|---|---|
| 3 | cube position, env frame (as `mdp.root_pos_w`) | `obs_T_hand @ hand_T_cube`, already in the node |
| 4 | cube quaternion (w, x, y, z), symmetry-canonical | `canonical_cube_quat` of the node's quaternion |
| 1 | tracked: last detection ≤ 0.5 s (10 steps) ago | the node's `cube_pose_timeout_s` = 0.5 |

Not tracked: the last pose is held and the flag is 0; before the first detection the pose is the command's target
position with the identity orientation.

**Cube symmetry.** The tracker solves PnP from 8 corner keypoints of a plain cube, so its orientation is defined only
up to the cube's 24 rotations and can jump by 90° when the corner labels flip. The observation uses the equivalent
closest to the identity (w ≥ 0), in sim and on the robot. The detector's slerp smoothing should compare each new
measurement with the symmetric equivalent closest to its current estimate (`closest_symmetric`), as the sim does.

## Tracker model (randomised per env and episode; `TRACKER_DEFAULTS`)

| effect | range | why |
|---|---|---|
| latency | 0–3 steps (0–150 ms) | camera + Mask R-CNN + PnP + TF |
| calibration offset | ±4 mm, ±3° per episode | AprilTag + hand-to-tag measurement |
| per-step noise | σ 0–3 mm, 0–5° | keypoint jitter through PnP |
| missed frames | 0–30 % per step | detector misses, < 20 Hz processing |
| lost bursts | enter 0–5 % / leave 10–50 % per step | fingers covering the cube |
| smoothing | position EMA gain 0.3–1, orientation slerp gain 0.2–0.6 | Kalman filter / slerp α 0.25 in the detector |

Placeholders until the tracker is measured on the hand (cube still for 30 s, then turned by hand): replace with the
measured jitter, latency and dropout statistics.

## Runs (pod, 2 slots)

- **A, fine-tune:** the rank-1 family (`06_oneshot_servowide` seed 42 m5999) widened to 82 inputs with
  `scripts/export/widen_actor_input.py` (new first-layer columns zero: the widened policy acts exactly like the old
  one, checked to 0.0), trained on `-realfric-wristl1-servofast-cube-v0` (amax 1–3×, calm recipe).
- **B, from scratch:** `-onedir-droppen-cube-v0` (tracking 45) → `-realfric-wristl1-cube-v0` → servowide on the cube
  task: does the policy use the pose more when it learns with it from the start?

Evals: `-upright-cube-v0` at 512 / 2048 envs with and without the servo model; tracker off
(`env.observations.policy.cube_tracker.params.force_lost=true`); tracker ×2 noise and lag. Sim vs real check: compare
`play.py --record-csv` observation distributions with the policy node's CSV log.
