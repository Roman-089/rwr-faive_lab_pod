# Sim2sim: Isaac-trained policies on a MuJoCo ORCA v2 hand (CPU)

[Index](README.md) · related: [eval protocol](eval_protocol.md), [export](export.md), [sim2real](sim2real.md),
[actuator net](actuator_net.md)

> **Role (user decision, 2026-10-29):** sim2sim results are informational. Design decisions for new ablations are
> based solely on the Isaac eval ([eval protocol](eval_protocol.md)); the harness stays the interface / pre-deploy check.

## Purpose

Run the exported policies (`policy_history.pt`, the file the robot loads) in MuJoCo on the CPU, with the observation /
action interface the real policy node will use, and score them with the [eval](eval_protocol.md) metrics. It has two
jobs:

1. **Interface check before the hardware.** It checks joint order, limits, obs order and scales, action rescaling, the
   EMA filter, 20 Hz timing, and history handling plus `reset()` of exported history policies. A bug there looks
   like a bad policy on the robot; here a laptop catches it.
2. **Second physics engine.** Policies that exploit PhysX quirks degrade in MuJoCo. Large drops from Isaac to MuJoCo
   are expected. What matters is the ranking and gross failures.

Step 2 (when `rwr_faive_system` is released): wrap the MuJoCo hand as a fake ROS hand driver so the course's policy
node runs unchanged ([plan below](#plan-fake-ros-hand-driver)).

## Results (2026-10-29)

Eval protocol as Isaac's ([eval protocol](eval_protocol.md)): |cmd| 1.5 rad/s about z, sign alternating, 60 s
trials, respawn after a drop, obs noise on. MuJoCo **nominal** = centers of Isaac's DR ranges, 32 trials;
**+ Isaac DR** = Isaac's DR terms sampled per trial, 16 trials. Isaac = the nominal eval in the [README](README.md)
table (64 envs, DR + noise), with per-direction RPM from its rollout. Login node, one core, ~12 s per trial (student
~16 s).

| Policy (checkpoint) | Isaac RPM (+z / −z) | Isaac RA / DFSR / DC | MuJoCo nominal RPM ± SE (+z / −z) | RA / DFSR / DC | MuJoCo + Isaac DR RPM ± SE (+z / −z) | RA / DFSR / DC | MuJoCo / Isaac RPM (nominal / DR) |
|---|---|---|---|---|---|---|---|
| `abl_B_rotlin` (model_4999) | **10.41** (9.53 / 11.30) | 74 / 84 / 0.22 | **5.39** ± 0.23 (4.29 / 6.50) | 64 / 88 / 0.16 | **5.04** ± 0.59 (3.40 / 6.67) | 61 / 94 / 0.06 | 0.52 / 0.48 |
| `abl_B_droppen` (model_4999) | **7.94** (7.34 / 8.53) | 80 / 83 / 0.27 | **4.39** ± 0.16 (5.14 / 3.63) | 87 / 88 / 0.12 | **4.79** ± 0.36 (5.19 / 4.39) | 88 / 100 / 0.00 | 0.55 / 0.60 |
| `abl_B_asym_ent5e-3` (model_4999) | **7.33** (7.94 / 6.71) | 79 / 88 / 0.12 | **5.38** ± 0.27 (6.77 / 3.99) | 86 / 81 / 0.19 | **4.99** ± 0.38 (6.02 / 3.96) | 85 / 100 / 0.00 | 0.73 / 0.68 |
| `student_pt` (model_1999) | **6.89** (7.44 / 6.33) | 75 / 95 / 0.05 | **2.79** ± 0.21 (3.69 / 1.89) | 57 / 72 / 0.38 | **2.68** ± 0.41 (3.93 / 1.43) | 55 / 88 / 0.31 | 0.41 / 0.39 |
| `abl_B_rotlin_ent5e-3` (model_4999) | **9.50** | 76 / 81 / 0.27 | **4.00** (64 trials) | 62 / 77 / 0.27 | **3.89** (64) | 60 / 83 / 0.19 | 0.42 / 0.41 |
| `abl_B_onedir_clip3_ent5e-3` (model_4999, +z only) | **15.19** | 79 / 44 / 1.30 | **5.42** (64) | 64 / 31 / 1.31 | **5.47** (64) | 58 / 17 / 1.83 | 0.36 / 0.36 |
| `abl_B_rotlin_droppen_ent5e-3` (model_4999) | **8.29** | 71 / 89 / 0.17 | **4.77** (64) | 72 / 92 / 0.08 | **4.44** (64) | 67 / 92 / 0.08 | 0.58 / 0.54 |

Rows added 2026-10-29 23:10 (job 29851207, CPU queue `hpc`, 64 trials each; informational, decisions use Isaac). The
fastest Isaac policy (onedir, 15.2 RPM) keeps only 36 % of its speed and drops most; the drop-penalty run keeps the
highest share among the rotlin variants (0.58) and is 92 % drop-free in MuJoCo too.

Cube height while held (mean z), Isaac → MuJoCo: rotlin −49.1 → −48.1 mm, droppen −26.7 → −24.3, ent5e-3 −27.3 →
−25.5, student_pt −27.4 → **−36.3** mm. Drops within 5 s of a (re)spawn (reset artifacts, see
[eval protocol](eval_protocol.md#drop-timing-are-better-starting-grasps-needed-2026-10-28-scriptsevaldrop_timingpy-cpu)),
MuJoCo nominal: 4 of 5 (rotlin), 4 of 4 (droppen), 5 of 6 (ent5e-3), but only 6 of 12 for student_pt. With DR,
student_pt's 5 drops all come later than 5 s.

**Reading:**

- **No interface bug.** All four policies grasp the cube and turn it in the commanded direction, with RA 55–88 %.
  The joint-level gait matches Isaac's (check iv). The consistency checks below found no mismatch in joint order,
  signs, limits, scales, EMA, timing or history handling.
- **The expected engine gap.** MLP policies keep 48–73 % of their Isaac speed, the transformer student ~40 %. Isaac's
  DR switched on in MuJoCo changes RPM by ≤ 0.4 (within SE) and lowers drops. The gap is not a missing randomization;
  it comes from the contact model (sensitivity table below: the cube's torsional friction alone moves RPM by 2×).
- **Ranking changes.** Isaac: rotlin ≫ droppen > ent5e-3 > student_pt. MuJoCo nominal: rotlin ≈ ent5e-3 (5.4) >
  droppen (4.4) ≫ student_pt. With DR the three MLPs tie (4.8–5.0, within SE).
  - **rotlin's +42 % speed over ent5e-3 does not carry over.** It keeps the least of its speed among the MLPs, has the
    lowest RA (61–64 %) and the most off-axis wobble (1.28 vs 0.49–0.90 rad/s). It holds the cube deep in the hand
    (−48 mm) and seems to use a PhysX-specific gait.
  - **droppen and ent5e-3 transfer best.** RA 85–88 % (higher than in Isaac) and 100 % drop-free with DR.
  - Direction preference holds for ent5e-3 (+z faster) and rotlin (−z faster), and flips for droppen.
- **student_pt is the worst in MuJoCo** (2.7–2.8 RPM, RA 55–57 %), while the README ranks it the best deployable
  policy by drop-free rate (95 %). It holds the cube 9 mm lower than in Isaac. It is the only policy with many
  steady-state drops, and its gait differs most (wrist std 14.5° vs 5.2° in Isaac). The harness drives its history
  exactly as specified (test: `flat_history()` == the last 10 harness obs, filled at reset). So the likely reason is
  that the transformer reads PhysX-specific contact signatures from the joint history, and they change with the
  engine. Before choosing it over a B-family MLP for the hardware demo, weigh this against its Isaac numbers.

The relative ranking from one engine to the other is the useful signal, not absolute numbers. Neither engine is the
real hand.

MP4s (first 2 trials, 20 s each, nominal):
`logs/rsl_rl/orca_v2_repose/{2026-10-28_21-11-45_abl_B_rotlin,2026-10-28_22-01-33_student_pt}/sim2sim/sim2sim_drnone_2trials_20s.mp4`.
Per-trial numbers: `<run>/sim2sim/metrics_dr{none,isaac}.json`, rollouts `rollout_*.npz` (`logs/` is gitignored).
64-trial runs: `scripts/sim2sim/sim2sim_job.sh` (CPU queue `hpc`, ~25 min, not submitted).

## Design

| File (`scripts/sim2sim/`) | What |
|---|---|
| `orca_task.py` | Everything copied from the Isaac task, each with its source: joint order, physics / control rate, mount pose, cube, obs scales and noise, command, reset and DR ranges. It parses `orca_v2.py`'s `ImplicitActuatorCfg` (no Isaac import) and samples resets / DR like Isaac's event terms. |
| `extract_usd_model.py` → `orca_v2_right_usd.json` | Extracts, from the hand USD, what PhysX simulates: 19 links (mass, COM, principal inertia), 19 joints (both joint frames, axis, limits) and 53 convex collision hulls (64 vertices each, link frame). Also stores the sha256 of the USD layers. 243 KB, committed, so the harness needs no pxr / Isaac Sim. Rerun only when the USD changes (needs pxr from Isaac Sim's bundled libs, see its docstring). |
| `scene.py` | Builds the MJCF from the JSON + `orca_v2.py` + task constants. `SceneCfg` holds the MuJoCo-only choices. `python scene.py --out x.xml` dumps it for `python -m mujoco.viewer`. |
| `hand.py` | `OrcaHandSim`, the **hand API**: `set_joint_targets(q*)`, `step_physics()` (one Isaac physics step, 1/120 s), `joint_pos()`, `joint_vel()` in the policy's joint order. Also the cube state (world-frame angular velocity), `reset(ResetState)` and `set_physics(PhysicsParams)` (DR). No policy logic. |
| `policy_io.py` | `PolicyIO`, **the policy node's side**: the 74-dim observation (+ optional Isaac obs noise) and the EMA action processing, with the node's nominal joint limits. No physics. |
| `run_sim2sim.py` | Eval CLI: loads the TorchScript file only (no repo / rsl_rl classes), runs N trials × 60 s, writes metrics JSON + rollout NPZ, optional MP4. |
| `consistency_checks.py` | `joints` (chirp vs Isaac's implicit PD) and `kinematics` (vs Isaac eval rollouts and the vendor MJCF). |
| `sim2sim_job.sh` | LSF CPU job (queue `hpc`): 64 trials × 4 policies × {nominal, DR}. |

One control step of `run_sim2sim.py` is the deploy loop:

```
obs = PolicyIO.observation(hand.joint_pos(), hand.joint_vel(), command)   # + N(0, .1) / N(0, .05) like Isaac
a   = policy(obs[None])                  # exported TorchScript, batch 1, exactly once per 50 ms tick
q*  = PolicyIO.process_action(a)         # clip [-1, 1] -> rescale to limits -> EMA 0.2 -> clip to limits
hand.set_joint_targets(q*); 6 x hand.step_physics()                    # 6 x 1/120 s (10 MuJoCo steps each)
if cube z < -0.2 m: count drop, respawn (Isaac's reset distribution), PolicyIO.reset(q), policy.reset()
log cube w_z (world frame) -> metrics(), a verbatim copy of scripts/eval/eval_policy.py's
```

Protocol as in `eval_policy.py`: |command| 1.5 rad/s about world z, sign alternating over trials (trial i = env i),
`+z` only for `-onedir` tasks (read from the export's task id), no resampling, 1200 steps, respawn after a drop.
`--dr none` (default) uses the centers of Isaac's DR ranges. `--dr isaac` samples Isaac's DR terms per trial: startup
terms once, reset terms at the spawn and again on a respawn ≥ 720 steps later, like `EventManager`. Obs noise is on as
in Isaac's eval (`--no_obs_noise` turns it off). Trials are seeded per trial index, so runs are reproducible
bit for bit (tested) and different policies see the same spawns.

## Model sources and fixes

**Vendor model.** [orcahand/orcahand_description](https://github.com/orcahand/orcahand_description), MIT license,
commit `b9b349a` (2026-07-31). Per its README, the v2 hand ("newer Fusion360 export") is `v2/`
(`v2/scene_right.xml` → `v2/models/mjcf/orcahand_right{,_body}.mjcf`) and `v1/` is the legacy hand. The top-level
`assets/mjcf/` holds meshes (`right/` with `collision/` + `visual/`), `options.xml` and the `scene.xml` with the 50 mm
cube, but no hand MJCF. Used here only as an independent
kinematics check, from a shallow, blobless, sparse clone in a scratch dir. The v2 right-hand STLs are 59 MB, too big
for the repo, and the harness does not need them:

```bash
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/orcahand/orcahand_description.git && cd orcahand_description
git sparse-checkout init --no-cone
printf '/LICENSE\n/v2/*.xml\n/v2/models/mjcf/*\n/v2/models/assets/*.xml\n/v2/models/assets/right/*\n' > .git/info/sparse-checkout
git checkout HEAD
```

**What the vendor MJCF gets right, and why the harness does not simulate with it.** The vendor MJCF's joints are
identical to our USD's. Over 50 random configurations, every link pose relative to ours stays one constant offset per
link, to 0.1 µm / 1e-4°. The offsets are pure translations: the vendor puts body frames at the mesh origins, e.g.
carpals (−5.0, −1.9, 49.5) mm, ap links (−17.75…−19.25, 0, −1) mm. But as a physics model it differs from what Isaac
trains on:

1. **No `<inertial>` at all.** MuJoCo then infers mass from every mesh at 1000 kg/m³, and the overlapping skin and bone
   meshes both count. Result: 1.5–4× the USD masses (fingertip 13.4 vs 8.1 g, ap 8.9 vs 3.5 g, carpals 280 vs 71 g,
   1.76 kg in total).
2. **Issue #19** ("mass/inertia inconsistencies in v2 URDFs", open) is about the **v2 URDF**, not the MJCF. The forearm
   is 7.42 kg (expected ~0.9). All inertias are ~10⁴× too large (exported as kg·cm² instead of kg·m²), e.g. M-PP
   Iyy 0.065. Left/right carpals mismatch (0.135 vs 0.854 kg), addressed by commit `c18735e` per the issue page. The
   report does not mention a steel density. Our USD has none of these errors: forearm 0.659 kg (fixed to the world
   anyway), finger links 3–10 g with 1e-7..2e-6 kg m² inertias. **Fix in our copy:** every mass, COM and inertia comes
   from the USD.
3. **Collision.** MuJoCo collides the convex hull of each mesh, so a whole-link carpals mesh fills the palm cup the cube
   sits in. Isaac collides with 53 convex pieces, 21 of them for the palm (`R_Carpals_heel`, `_center_lo`, ...).
   **Fix:** the harness uses exactly those pieces. They are already 64-vertex convex hulls, so MuJoCo's hull of the
   same vertices is the same shape PhysX uses.
4. **Actuators and joints** in the vendor file: kp 2, ±1 N·m, joint damping 0.1, armature 0.001. They have nothing to
   do with `orca_v2.py`. Joint names also differ (`right_i-abd`, ...; the thumb's last joint is `t-pip` = our
   `thumb_dip`).

So the MuJoCo hand is **rebuilt from what Isaac simulates**: kinematics, masses and collision hulls from the USD
(`orca_v2_right_usd.json`), drives from `orca_v2.py`, scene from the env cfg. The vendor files are only a cross-check.

**Cube.** Isaac's `DexCube/dex_cube_instanceable.usd` (Isaac 5.1 Nucleus, checked by download): the collider is a
`Cube` primitive (sharp box, 60 mm scaled to 45 mm), and its mass 0.216 kg is overridden to 0.11 kg. It carries
**PhysX torsional patch friction** (`torsionalPatchRadius` 0.1, `minTorsionalPatchRadius` 0.008 m). The bound
`CubeMaterial` prim does not exist, so the material comes from Isaac's DR with the default *average* combine mode.

## How the scene matches Isaac

| Item | Isaac (source) | MuJoCo | Status |
|---|---|---|---|
| Joint order (obs / action) | articulation order, logged as "Joint names:" in every training log: wrist, index/middle/pinky/ring abd, thumb_cmc, 4× mcp, thumb_abd, 4× pip, thumb_mcp, thumb_dip | same (`JOINT_NAMES`) | tested against 4 run logs + action-term log |
| Kinematics | USD joint frames | built from the same frames | < 1 µm vs Isaac rollouts (below) |
| Masses / inertia | USD `MassAPI` | copied | exact |
| Collision geometry | 53 convex hulls (64 vertices) | same vertices | exact |
| Self-collision | on (`orca_v2.py`); PhysX skips joint-connected links | explicit `<exclude>` for the 18 joined pairs (MuJoCo's parent filter misses the wrist: its parent is world-welded) | same pairs |
| Hand gravity | `disable_gravity=True` | `gravcomp="1"` on all hand bodies | exact |
| Mount | root `right_forearm` fixed at (0.0004, 0.1512, −0.0187), quat (−0.2607, 0.9654, 0, 0) | same (quaternion normalized, the cfg's has norm 0.99998) | exact |
| Drive | PhysX implicit PD, k / d / effort per joint (`orca_v2.py`) | position actuator kp = k, kv = d, forcerange = effort, integrator `implicitfast` | chirp check below |
| Armature | 0 | 2·d·dt (7e-4..3.4e-3 kg m²) | numerical fix, see below |
| Joint friction | Isaac Sim 5 static friction effort 0–0.105 N·m; dynamic and viscous 0 (log table) | `frictionloss` (Coulomb, also while moving) | approximate |
| Physics step | 1/120 s, TGS, 8 position iterations | 1/1200 s, 10 MuJoCo steps per Isaac step | sensitivity below |
| Control | 20 Hz (decimation 6), EMA α 0.2, rescale to limits | same (`PolicyIO`) | tested against Isaac Lab's class |
| Cube | box 45 mm, 0.11 kg, I = m s²/6 | same | exact |
| Hand–cube friction | PhysX average of hand (0.8–2.0) and cube (dynamic 0.4–1.0) | cube geom has priority, μ = average (nominal 1.05) | same rule; Isaac's static μ unused |
| Torsional friction | DexCube min torsional patch radius 8 mm | condim 4, torsional μ·8 mm | approximate (PhysX's grows with penetration) |
| Restitution | cube 0–0.2 | none (critically damped soft contact) | not modeled |
| Contact | rigid, contactOffset 1 mm, restOffset 0 | soft: solref 4 ms, solimp 0.95/0.99, elliptic cone, impratio 10 | engine difference |
| Reset | joints U(0.2 lo, 0.2 hi) (`reset_robot_joints`), cube ±5 mm / z −19…−21 mm / Euler U(±3.145) / lin. vel ±1 mm/s, EMA state = joint pos, last action 0 | same sampler, Isaac's `quat_from_euler_xyz` | tested |
| DR (`--dr isaac`) | hand friction per body bucket, link mass ×0.95–1.05, gains ×U_log(0.75–1.5 / 0.3–3), limits +N(0, 0.01), cube friction / mass ×0.5–1.5, hand roll ±0.05, gravity noise | same ranges; one hand friction per trial; limit offsets on the physical joints only (the node keeps nominal limits); gravity noise not modeled (Isaac applies it after 24–360 s) | close |
| Obs noise | N(0, 0.10) joint_pos (normalized), N(0, 0.05) joint_vel before ×0.2 | same | tested |
| Command | (0, 0, −0.05, 0, 0, ±1.5), env frame = world | same | Isaac's eval shows a resampled command on the first obs after a respawn (pinned from the next step); MuJoCo pins it always |

**The armature fix.** Without it the hand flew apart: a 0.11 kg cube was flung at > 20 rad/s within 0.2 s, and
joints jumped 100° in one step. MuJoCo integrates `kv` implicitly but **drops that term while the actuator force is
clamped at `forcerange`**. A clamped finger joint (I ≈ 5e-6 kg m²) then integrates 1.3 N·m / I explicitly and slams
between its limits. PhysX keeps the drive implicit inside the clamp. An armature of 2·d·dt (dt = MuJoCo step)
bounds the velocity change of one clamped step to F / (2d), inside the damping band, so the clamp engages smoothly. It
adds a 2·dt = 1.7 ms mechanical lag to a 70–140 ms first-order joint response (chirp check: invisible). The real
servo's reflected rotor inertia is probably larger (not measured). Doubling it changes RPM within noise (sensitivity table).

**Sensitivity to MuJoCo-only choices** (`abl_B_rotlin`, 8 trials × 60 s, nominal physics, ± = SE over trials; first
sweep with the cube at condim 3, before the DexCube torsional friction was found):

| Variant | RPM | RA % | DFSR % | DC | s / trial |
|---|---|---|---|---|---|
| *cube condim 3 (no torsional friction):* 1/1200 s, solref 4 ms, impratio 10, armature 2·d·dt | 6.11 ± 0.34 | 61.8 | 50 | 0.50 | 10.0 |
| timestep 1/2400 s | 5.86 ± 0.25 | 59.1 | 50 | 0.62 | 17.9 |
| timestep 1/600 s | 6.61 ± 0.26 | 63.1 | 62.5 | 0.38 | 5.6 |
| contact solref 20 ms (MuJoCo default, softer) | 6.85 ± 0.26 | 64.9 | 75 | 0.25 | 10.9 |
| impratio 1 (softer friction) | 6.73 ± 0.23 | 64.0 | 62.5 | 0.38 | 8.7 |
| armature ×2 | 6.54 ± 0.27 | 64.2 | 62.5 | 0.50 | 9.6 |
| **default:** + condim 4, torsional friction μ·8 mm (DexCube min patch radius) | **5.69 ± 0.38** | 63.8 | 62.5 | 0.50 | 11.6 |
| torsional friction μ·30 mm | 3.51 ± 0.25 | 53.4 | 75 | 0.25 | 11.1 |
| default without obs noise | 4.57 ± 0.62 | 60.5 | 62.5 | 0.38 | 11.5 |

Timestep (1/600 → 1/2400 s), contact softness, friction-cone hardness and armature move RPM by ≤ 0.7, about two SE
of an 8-trial run. Every variant stays far below Isaac's 10.4. 1/1200 s is converged against 1/2400 s within the SE
at 60 % of its cost. The **torsional patch friction is the most sensitive parameter**: 8 mm → 30 mm halves the
speed. PhysX's effective radius is ≥ 8 mm and grows with penetration, so its exact value is the largest physics
uncertainty of the cube model. Without obs noise the policy is not faster (it was trained with it).

## Consistency checks

**(i) Joints** (`consistency_checks.py joints`). Every joint of the full MuJoCo hand runs the chirp of
`motor_param_characterization/chirp_characterization.py`: 20 s, 0.1–5 Hz logarithmic, ±10° about mid-range, 20 Hz
commands, joint read before each command acts, 10 s settling, other joints held at 0, no cube. The reference is the
1-DoF replica of PhysX's implicit drive (`actuator_net/servo_model.py` `pd_step`: implicit Euler of
I q̈ = k (q* − q) − d q̇ − friction) with each joint's USD inertia. It runs with 1 implicit step per 1/120 s
(**PhysX-1**) and with 8 sub-steps (**PhysX-8**, TGS-like), which brackets the unknown TGS integration. FRF via
`signal_processing.py`.

| Joint (type) | I [kg m²] | RMSE vs PhysX-1 / PhysX-8 [°] | MuJoCo gain dB / phase ° at 0.5 · 1 · 2 · 4 Hz | worst Δ vs PhysX-1 | worst Δ vs PhysX-8 |
|---|---|---|---|---|---|
| wrist | 1.5e-03 | 0.16 / 0.07 | −0.6 / −20 · −0.9 / −36 · −2.6 / −64 · −6.3 / −104 | 0.5 dB / 1° | 0.2 dB / 1° |
| index_abd | 6.4e-05 | 0.14 / 0.05 | −0.5 / −19 · −0.9 / −36 · −2.8 / −65 · −6.6 / −104 | 0.4 dB / 1° | 0.1 dB / 1° |
| thumb_cmc | 6.7e-05 | 0.09 / 0.04 | −0.8 / −25 · −2.0 / −46 · −5.1 / −77 · −10.0 / −111 | 0.3 dB / 1° | 0.1 dB / 1° |
| index_mcp | 3.3e-05 | 0.11 / 0.04 | −0.6 / −22 · −1.4 / −41 · −4.0 / −71 · −8.5 / −108 | 0.3 dB / 1° | 0.1 dB / 2° |
| thumb_abd | 5.2e-05 | 0.12 / 0.07 | −0.5 / −21 · −1.2 / −39 · −3.5 / −69 · −7.9 / −108 | 0.3 dB / 2° | 0.1 dB / 3° |
| index_pip | 4.8e-06 | 0.08 / 0.03 | −1.0 / −28 · −2.4 / −50 · −6.0 / −80 · −11.1 / −113 | 0.3 dB / 1° | 0.1 dB / 1° |
| thumb_mcp | 1.7e-05 | 0.09 / 0.03 | −0.8 / −25 · −1.8 / −45 · −4.9 / −76 · −9.7 / −110 | 0.3 dB / 1° | 0.1 dB / 1° |
| thumb_dip | 1.8e-06 | 0.09 / 0.03 | −0.8 / −25 · −1.8 / −45 · −4.9 / −76 · −9.7 / −110 | 0.3 dB / 1° | 0.1 dB / 1° |

Middle / ring / pinky equal index (inertia does not matter next to the damping). Over all 17 joints: RMSE ≤ 0.16°
(max error 0.49°) vs PhysX-1 and ≤ 0.07° vs PhysX-8 on a ±10° chirp; gain within 0.5 / 0.16 dB, phase within 2 / 3°.
MuJoCo sits between the two references, closer to PhysX-8. The PD gains, units (N·m/rad, N·m·s/rad: Isaac Lab writes
rad-based drive values, and the log table shows the same numbers) and effort limits carry over. The joint-level
dynamics are not where the engines differ.
Not covered: the drives under load (contact forces, effort clamp), and PhysX's static-only joint friction (the replica
and MuJoCo both use Coulomb friction; ≤ 0.039 N·m except the wrist, 0.105).

**(ii) Kinematics** (`consistency_checks.py kinematics`). Isaac's eval rollouts store all 19 link poses (4 envs ×
1200 steps), not the joint angles. The angles are recovered from parent / child link rotations about each USD joint
axis. MuJoCo is then posed there, with the root at Isaac's (roll-randomized) root pose:

- `abl_B_rotlin` model_4999.npz, 960 configurations: link positions within **0.94 µm** (mean 0.64), orientations
  within 1e-4°. Kinematic tree, joint frames, mount pose and link frames are identical.
- **Joint sign / zero:** the recovered angles lie within the (asymmetric) limits ±3° in 100 % of samples. With every
  axis flipped it would be 51 %. So the joint angle MuJoCo reports is Isaac's `joint_pos`, sign and zero included.
  Isaac exceeds its nominal limits by up to 1.9° (DR on the limits plus PhysX's soft limit).
- Vendor MJCF: identical kinematics (above).
- **Not verified:** that the joint angles Isaac *observed* equal these reconstructed ones (Isaac stores no joint
  angles; a PhysX joint-zero offset would show here as a limit violation, and there is none).

**(iii) Interface** (CPU tests, `/tmp/claude-322334/-zhome-b3-e-222334/e98b2254-0dd6-46ce-be5a-af9ce9954bee/scratchpad/sim2simtest/test_sim2sim.py`,
`run_tests.sh`, output `test_output.txt`, not in the repo): **47 / 47 pass.**

- EMA action processing equals **Isaac Lab's real `EMAJointPositionToLimitsAction`** (imported with stubbed bases, as
  the action-holding test does) over 200 steps with 3 resets: targets within 3e-7 rad, `last_joint_pos_cmd` =
  `_prev_applied_actions`, `last_action` = unclipped `raw_actions`.
- joint_pos normalization = `isaaclab.utils.math.scale_transform`, exactly −1 / 0 / +1 at lower / center / upper.
  joint_vel ×0.2. Command (0, 0, −0.05, 0, 0, ±1.5). First obs after a reset: last_action 0, last_joint_pos_cmd = joint
  positions in rad.
- Obs noise: measured std 0.100 on joint_pos and 0.0100 on the joint_vel obs (0.05 before the 0.2 scale), 0 on the
  rest.
- The 4 exported JSONs (obs layout, starts, dims, scales, train noise, action term) match the harness.
- `JOINT_NAMES` equals the articulation and action-term joint order in all 4 runs' Isaac logs. Stiffness, damping,
  armature, friction, limits and effort limits as the harness reads them equal Isaac's printed joint table.
- 20 task constants found verbatim in the env cfg (dt, decimation, α, scales, noise, mount, cube, drop height, command,
  reset and DR ranges, hand gravity off).
- `metrics()` has the same code and values as `scripts/eval/eval_policy.py`'s.
- Cube angular velocity is world-frame (= `mj_objectVelocity`, ≠ MuJoCo's body-frame free-joint qvel).
- Hand API joint / actuator order, gravity compensation, reset distribution and Isaac's Euler convention all match.
  A trial with DR and noise is deterministic.
- History policy (`student_pt`): over two episodes, the module's `flat_history()` equals the last 10 observations the
  harness fed it, term-major, oldest first, filled with copies of the first obs after `reset()`.

**(iv) Closed loop: same gait.** The joint angles Isaac's eval rollout recovers (4 envs) are compared with MuJoCo's
(32 trials), first 2 s after the spawn skipped (`abl_B_rotlin`, script `gait_compare.py` in the scratch test dir). Per
joint, means agree within 0–7° except index_pip (10°) and ring_mcp (12°), and spreads within a few degrees. Mean
joint speed is 1.05 vs 1.23 rad/s.
The cube is held at the same height (−48.1 vs −49.1 mm; the other policies hold it at −27 mm in Isaac). So the policy
sees in-distribution observations and moves the fingers as in Isaac. The speed gap is in how the finger motion
transfers to the cube (contact dynamics), not in the interface.

The exported modules' own history and `reset()` semantics are verified in [export](export.md). Here `reset()` is
called at every (re)spawn, as the deploy contract says.

## Findings for the policy node / deploy

1. **Joint order.** The policy's order is the articulation order (wrist, index/middle/pinky/ring abd, thumb_cmc,
   4× mcp, thumb_abd, 4× pip, thumb_mcp, thumb_dip), **not** the order of the joint list in `orca_v2.py` (wrist, thumb ×4,
   index ×3, ...). The exported JSON records only `joint_names: [".*"]`. The node must hard-code the first order;
   `orca_task.JOINT_NAMES` is the reference. Adding the resolved joint names and limits to the export JSON would make
   this checkable on the robot (not done: export script not changed).
2. `last_action` is the **raw, unclipped** policy output. In MuJoCo, 58 % of `abl_B_rotlin`'s action entries are
   outside [−1, 1] (p99 |a| 4.6, max 7.7): the policy is bang-bang. A node that clips before storing it feeds the
   policy out-of-distribution input.
3. joint_vel noise is added before the 0.2 scale (std 0.01 in obs units). Irrelevant on deploy (no noise there), but it
   matters for any sim-side replica.
4. `reset_robot_joints` samples U(0.2·lower, 0.2·upper) per joint, e.g. mcp −5°…+20° and pip −3°…+21°. That is not
   "effectively none" ([sim2real](sim2real.md#domain-randomization-ranges) says so). It is what the code does.
5. Scoring needs the cube angular velocity in the **world** frame: MuJoCo's free-joint `qvel[3:6]` is body-frame.

## Limitations

- **MuJoCo is not the real hand.** It has the same idealized drives as Isaac: no servo delay, backlash, tendon
  stretch, current limit dynamics, or soft fingertips. A policy that works here can still fail on hardware. One that
  fails here for a physics reason likely relies on PhysX specifics (contact patches, rigid contacts, torsional
  patch friction).
- Contacts differ by design (soft vs rigid, emulated patch / torsional friction, no restitution). The joint check covers
  free motion only. There is no step-by-step comparison of hand–cube dynamics with Isaac (no Isaac on CPU, and the
  rollouts store no joint targets).
- Numerical armature (above); Coulomb instead of static-only joint friction; one hand friction per trial; no gravity DR.
- Trials run sequentially, one env each: 32 trials give an SE of ~0.3 RPM, same order as Isaac's 64-env eval (0.2).
- The harness copies the task constants. A cfg change must be mirrored in `orca_task.py`; the CPU test fails on a
  mismatch of the checked constants. The export JSON is checked: dims 74 / 17, EMA α 0.2, rescale to limits.
  `-relact` (other action term) and `-wristfix` (16 actions) are rejected. `-delay` / `-acthold` (EMA subclasses) run
  with a warning, without their training-time delay / holding, as the deploy node would run them. `-jointbias` runs
  without its training-time bias. Changed reward or command ranges (`-onedir`: +z only, read from the task id) need
  nothing else.

## How to run

```bash
source ~/rwr_faive_lab/activate_rwr.sh                # conda env rwr: torch, mujoco 3.17, imageio; CPU is enough
# 1. export (once per run; CPU, ~10 s)
python scripts/export/export_history_policy.py --run logs/rsl_rl/orca_v2_repose/<run>
# 2. sim2sim eval: ~12 s per 60 s trial (MLP, student ~16 s) on one core; --dr isaac for Isaac's DR; --video: MP4 (EGL)
python scripts/sim2sim/run_sim2sim.py --policy logs/rsl_rl/orca_v2_repose/<run> --trials 32 [--dr isaac] \
    [--video sim2sim.mp4 --video_trials 2]
#    -> <run>/sim2sim/metrics_dr{none,isaac}.json (per-trial numbers, settings), rollout_*.npz
# 3. 64 trials x 4 policies x {nominal, DR} as a CPU job, ~30 min (edit RUNS in the script; not submitted yet;
#    overwrites the 32 / 16-trial metrics_dr*.json from the login node)
bsub < scripts/sim2sim/sim2sim_job.sh
# consistency checks (CPU, ~1.5 min / seconds)
python scripts/sim2sim/consistency_checks.py joints
python scripts/sim2sim/consistency_checks.py kinematics --rollout logs/rsl_rl/orca_v2_repose/<run>/eval/model_X.npz \
    [--vendor <orcahand_description checkout>]
# inspect the scene
python scripts/sim2sim/scene.py --out /tmp/orca_sim2sim.xml && python -m mujoco.viewer --mjcf=/tmp/orca_sim2sim.xml
```

`--policy` takes any TorchScript module with `forward(obs[1, 74]) -> [1, 17]` (and `reset()` if it has one); the JSON
embedded by `export_history_policy.py` is checked (dims, EMA α 0.2 action term). After a USD change, rerun
`extract_usd_model.py` (docstring has the pxr environment).

## Plan: fake ROS hand driver

When `rwr_faive_system` is released, the course's node `experiments/rl_inference/policy_inference_cube_spin_v2.py`
should run against MuJoCo unchanged, so its own joint mapping, scaling, EMA, timing and model loading are what gets
tested:

1. **Read the real driver's interface first:** topic names / message types, joint naming and order, units (the ORCA
   stack may use degrees and motor-side values), rate, and the command → servo path (does the node send EMA targets or
   raw actions?).
2. **`orca_mujoco_driver` node** around `OrcaHandSim` (no change to the class: it already has the driver interface).
   Subscribe to the hand command topic → convert units / order exactly as the real driver does →
   `set_joint_targets`. A physics timer at 120 Hz → `step_physics()`. Publish joint states at the real driver's rate
   and names. Publish `/clock` and run the node with `use_sim_time` for lock-step, deterministic runs; a real-time mode
   tests 20 Hz jitter.
3. **World side:** cube drop detection + respawn as here (`orca_task.sample_reset`), cube pose / angular velocity on a
   separate topic for scoring only, rosbag recording, scoring offline with `run_sim2sim.metrics`.
4. **Acceptance test:** same policy, seed and spawns through `run_sim2sim.py` and through node + fake driver. With
   obs noise off and lock-step, the first observation must be bit-identical and the trajectories must match. Any
   difference is an interface bug in the node (joint order, limits, scales, EMA, `reset()`, ticks).
5. Then add what the real system has: the fixed one-tick actuation delay ([sim2real](sim2real.md#fixed-action-delay-sim--deploy)),
   dropped / late commands, and the actuator-net servo model on the MuJoCo side.
