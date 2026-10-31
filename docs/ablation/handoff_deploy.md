# Handoff: deploy the +z cube-rotation policies on the real ORCA v2 hand (2026-10-31)

From the DTU training session (`dtu-ablation-study-rwr`) to the session at the hand (`deploy-rwr`, MacBook). Goal: run the
trained policies on the real ORCA v2 hand with the course's `rwr_system` stack, safely, and send the trial logs back.

## 1. What you deploy

W&B artifact **`rwr-fast/orca_v2_repose/orca_v2_zrot_deploy:latest`** (v0). The W&B instance is CoreWeave's, API host
`https://api.forge.coreweave.com` (web UI `https://forge.coreweave.com/wandb/rwr-fast/orca_v2_repose`). The user logs
in themselves (`wandb login --host https://api.forge.coreweave.com`); never ask for, store or echo an API key.

```bash
wandb artifact get rwr-fast/orca_v2_repose/orca_v2_zrot_deploy:latest --root ~/rwr_deploy/orca_v2_zrot_deploy
```

Contents: `1_tilt_pos45_adr/`, `2_tilt_adr/`, `3_pos45_adr/` (each `policy.pt` TorchScript + `policy.json`),
`run_policy.sh`, `README.md`, `manifest.json` (sha256 per policy: check them after download).

| # | policy | Isaac sim | notes |
|---|---|---|---|
| 1 | tilt penalty + cube position ×3 + ADR | 14.2 RPM, 98 % drop-free; 72 % with 50–100 ms delay + encoder offset | fastest |
| 2 | tilt penalty + ADR | 11.6 RPM, 91 % / 83 % | most conservative: flattest cube, almost still wrist |
| 3 | cube position ×3 + ADR | 12.3 RPM, 95 % / 84 % | tilts the cube more, more wrist motion |

All three: proprioception only, memoryless MLPs, **+z rotation only**, 20 Hz, same 74-dim observation and 17-action
interface as the course's default `policy.pt`. They run in the **unchanged** course node
`rwr_system/experiments/rl_inference/policy_inference_cube_spin_v2.py` (checked at rwr_system commit f0f8cf6; on DTU
the unchanged node drove a MuJoCo hand for 16 × 60 s trials without drops after spawn, and its observations / commands
matched the training code to float32 precision).

## 2. Must-haves (each one breaks the run if missed)

1. **`goal_ang_vel:=1.5`** (the node's default 1.0 makes the policies hold the cube without turning it), **flip period
   0** (+z only). `run_policy.sh` sets both.
2. **Hand pose like in training:** palm up, **forearm ~30° below horizontal, fingertips lower than the wrist** (palm
   normal 30° from vertical, tilted toward the fingertips). Training randomized only 27–33°.
3. **Cube:** 45 mm (size was not randomized), ideally ~110 g (training covered 55–165 g; weigh it). Place it in the
   middle of the palm.
4. A **calibrated** hand (`calibration.yaml` in the hand_control install, see `rwr_system/src/hand_control/README.md`),
   joint order from `config.yaml` (the node maps it; don't edit the node's tables).

## 3. Procedure

```bash
# terminal 1: hand controller (rwr_system README)
ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right launch_rviz:=true

# terminal 2: dry run first (computes and logs actions, does not move the hand)
cd ~/rwr_deploy/orca_v2_zrot_deploy && ./run_policy.sh 2 -p dry_run:=true
```

Dry-run check: no "Waiting for hand state", the logged normalized joint positions lie in about [−1, 1], `|action| max`
is finite. Then live, one 60 s trial at a time (the node commands the hand as soon as it starts; Ctrl+C stops it and
the hand holds its last pose; emergency stop: `ros2 service call /right/hand/estop std_srvs/srv/SetBool "{data: true}"`):

```bash
mkdir -p ~/rwr_trials
./run_policy.sh 2 -p csv_log_path:=$HOME/rwr_trials/p2_t1.csv     # policy 2 first: check everything works
./run_policy.sh 1 -p csv_log_path:=$HOME/rwr_trials/p1_t1.csv     # then the fastest
./run_policy.sh 3 -p csv_log_path:=$HOME/rwr_trials/p3_t1.csv
```

Restart the node for every trial (fresh episode state). Per trial note: policy, cube mass, hand angle, full rotations
in 60 s (count from video), time of the first drop (or none), anything odd (jitter, a joint at its limit, overheating).
Film the trials if possible.

## 4. Report back

- Upload the CSVs (+ videos, notes) as a W&B artifact: `WANDB_ENTITY=rwr-fast wandb artifact put -n
  orca_v2_repose/hand_trials_2026-10-31 -t dataset ~/rwr_trials` (or `wandb.Artifact(...)` in Python).
- Message the DTU session `dtu-ablation-study-rwr` (SendMessage) with a short table per trial and the artifact name.
  The CSV logs (observation, joint commands, time) let it estimate the real action delay and joint offsets; they set
  the ADR ranges of the next training round.

## 5. If something is off

- Policy holds but does not turn → `goal_ang_vel` is not 1.5, or the hand angle is far from 30°.
- `size of tensor a (74) must match ... (81)` → wrong node / policy pairing (these policies take 74 values).
- Cube flies off at the start → the cube was not resting in the palm when the node started; place it, then start.
- Fallback that turns both ways (transformer student, 10-step history): in `rwr-fast/orca_v2_repose/deployable_policies_2026-10-31:latest`, folder `fallback_student_pt_rotlin_bidirectional`; it **needs one node change**: `self.policy.reset()` right after the warm-up call `self.policy(torch.zeros(1, OBS_DIM))`, and the default `goal_ang_vel`.

## 6. Optional: dry-run against a simulated hand on the Mac

Branch `ablation-study` of `rwr-fast/rwr_faive_lab` has `scripts/sim2sim/ros_sil/orca_mujoco_driver.py`: a ROS 2 node
with `hand_control_node`'s topics, backed by a MuJoCo model of the hand + cube. With ROS 2 sourced, `pip install mujoco`,
the repo cloned, and the **real hand controller not running** (same topics):

```bash
python scripts/sim2sim/ros_sil/orca_mujoco_driver.py <rwr_system checkout> --state_hz 100
./run_policy.sh 1        # in another terminal
```

It has only run under an in-process ROS stand-in on DTU (no ROS there), never under real ROS 2: treat failures as
bugs in the driver, not the policies, and report them.

## Rules from the user

- Commits: `git -c user.name=Roman-089 -c user.email=209470582+Roman-089@users.noreply.github.com commit ...`, **no
  Co-Authored-By / Claude attribution lines**.
- Never handle W&B API keys or GitHub tokens; the user logs in.
- Don't change the policies; ask before changing the course node beyond the documented `reset()` line.
