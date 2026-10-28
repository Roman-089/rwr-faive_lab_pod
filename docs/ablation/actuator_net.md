# Actuator network (Phase 2, sim2real)

[Index](README.md) · [sim2real](sim2real.md) · code: `scripts/actuator_net/`,
`source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net.py`

**Status: implemented, not yet trained in RL — Phase 2 (sim2real).** Fitted offline (CPU) to the ORCA **v1 / v1b**
reference chirps, CPU-tested against the real Isaac Lab actuator classes, **not run in Isaac Sim yet**. The deployed
weights are placeholders until the ORCA v2 chirps are recorded: retrain on those first
([below](#retrain-on-the-orca-v2-chirps)).

## TL;DR

- A learned servo model for all 17 joints, fitted to commanded vs measured **position** only (no force sensor):
  per joint type a **command delay**, a **servo play** (backlash / stiction band) and a **PD time constant**, plus a
  small **residual MLP** shared by all joints. It plugs into Isaac Lab as an `ImplicitActuator` subclass that reshapes
  the PhysX PD target every physics step; stiffness, effort limit and friction stay those of `orca_v2.py`.
- Held-out test (fit on the v1 hand, test on the v1b hand, 8 joint types): position RMSE **2.20 → 1.38 deg**, Bode
  phase error **18.6 → 8.0 deg**, gain error **3.8 → 1.7 dB** versus the current implicit PD. Most of that comes from
  the 3-parameter physical prior (1.47 deg / 9.1 deg / 1.6 dB); the net adds ~6 % RMSE and ~12 % phase on top.
- What the chirps say about the real servo: the joint **stops ~2-3 deg short of every reversal** and then follows with
  that offset (this is why the chirp gain looks like a flat -1.5 dB); it **lags by ~25-50 ms plus a ~20-35 ms first-order
  lag**; and the play **differs between hands** (thumb: ±3.3 deg on v1, almost none on v1b). The current PD
  (70-140 ms time constants) loses 7-12 dB at 4 Hz where the real joints lose ~3 dB (0.4-6.5 dB).
- New task id to register (snippets [below](#enable-it)): `orca-v2-inhand-repose-asym-actnet-v0` = ablation B with the
  servo model. Policy observation unchanged, so the policy deploys like B.

## Motivation

The sim drives every joint with an ideal implicit PD (`orca_v2.py`), fitted per joint type by Bayesian optimisation
of stiffness / damping / friction / effort limit against the chirp FRF (task PDF §7). A PD is a first-order lag: it
cannot have a flat gain and a large phase lag at the same time, and it has no latency and no play. The real servos
have both (see the plots in `scripts/actuator_net/results/`). An actuator network learns the command-to-motion map
from data instead, so the policy trains against servo behaviour close to the real hand's.

## Papers

### Yuryev & Hughes 2026, tendon force modeling (arXiv 2603.04351)

- **Problem.** Tendon hands use position-controlled servos (Dynamixel XC330-T288-T here); an "ideal force source"
  `F = P (θd − θ)` (P = 4.2 N/rad, i.e. a P law like our PD) misses motor friction, tendon slack, latency and
  compliance, so RL policies trained on it transfer badly.
- **Inputs / outputs.** History of H = 30 steps at 20 Hz (1.5 s) of `[θd, θ, θ̇]` (desired motor position, measured
  motor position, measured motor velocity) → one scalar, the tendon force in N. They argue slow servos need a much
  longer context than the 3-step history of Hwangbo et al. Normalisation statistics are stored inside the exported model.
- **Architectures** (similar parameter counts, 0.1-0.2 MB): MLP 256-128-64 ReLU on the flattened history; RNN with a
  64-d hidden state; Transformer encoder (linear projection to 16-d, sinusoidal positions, 2 layers × 4 heads, causal
  mask, last step → 3-layer MLP). The Transformer generalises best across loads; the RNN drifts; the MLP is close on
  the finger but oscillates on the springs.
- **Data.** A force test bench: servo → tendon → inline load cell → weak spring / strong spring / TPU finger, partly
  blocked by hand for contact; random steps, sinusoids, ramps and stairs; ~36 min at 80 Hz (load cell 20 Hz
  upsampled).
- **Integration.** Vsim (GPU tendon-force simulator): the force is computed at the 20 Hz control rate and applied for
  the following physics steps; motor position / velocity come from the simulated tendon length (θ = l / r).
- **Results.** Force RMSE 0.61 N (2.9 % of the 21 N max) across setups; replaying a sinusoid in sim, fingertip RMSE
  8.6 vs 14.6 mm for the ideal source (−41 %); an RL fingertip-tracking policy trained with the model tracks with
  12 vs 24 mm on the real finger (−50 %), mainly because the ideal-source policy overshoots when opening.
- **What we take:** a history-based servo model trained on servo signals only at inference time; the ideal P law as the
  baseline to beat; normalisation inside the exported file; the warning that slow servos need context (we tested a
  0.5 s history, [below](#offline-results)). **What we cannot copy:** their training target is a measured force. We have
  no load cell, only position (next section).

### Hwangbo et al. 2019, actuator net (arXiv 1901.08652)

- MLP, 3 hidden layers × 32 units, softsign; input = position error (target − actual) and velocity at t, t − 10 ms and
  t − 20 ms; output = joint torque. One net shared by the 12 identical ANYmal SEAs (actuators assumed independent).
- Data: < 4 min at 400 Hz (> 1 M samples) of sine foot trajectories at 1-25 Hz with manual disturbances, measured SEA
  torque, 90 / 10 split. Validation RMS 0.74 Nm vs 3.55 Nm for the ideal actuator (test 0.97 vs 5.74 Nm).
- Rules we follow: the history must be longer than the delays plus the mechanical response time, tuned on validation
  error; the excitation must cover a wide frequency band, otherwise the model makes the simulated robot oscillate.

### Isaac Lab 2.3.2 actuators (`isaaclab.actuators`)

- `ImplicitActuator`: stiffness / damping are written into PhysX and the PD is integrated implicitly by the solver
  (stable for any gains); `compute()` only estimates the torque for logging / rewards and passes the targets through.
- Explicit actuators (`IdealPDActuator`, `DCMotor`, ...): `compute()` returns joint efforts every physics step; sim
  stiffness / damping are set to 0 and the efforts are applied with `set_dof_actuation_forces`.
- `ActuatorNetMLP` (Hwangbo): per-physics-step history buffers of position error and velocity, `input_idx` picks the
  taps, `pos_scale / vel_scale / torque_scale`, TorchScript file, history zeroed on reset. `ActuatorNetLSTM` (Rudin
  et al., legged_gym): (position error, velocity) per step with an LSTM state. Both subclass `DCMotor` (explicit).
  `GO1_ACTUATOR_CFG` uses the walk-these-ways MLP with taps `[0, 1, 2]`; `ANYDRIVE_3_LSTM_ACTUATOR_CFG` the ANYdrive LSTM.
- `Articulation.write_data_to_sim()` calls every actuator's `compute()` once per physics step (120 Hz here) with the
  current joint state; `reset()` is called per env on episode reset. `randomize_actuator_gains` rewrites
  `actuator.stiffness / damping` (and writes them to PhysX for implicit actuators).

### Public code

- arXiv 2603.04351: none found (arXiv page, HTML / PDF, GitHub search).
- Hwangbo et al.: no official training code. Isaac Lab ships the `ActuatorNetMLP / LSTM` classes and pretrained
  ANYdrive / Go1 nets. Improbable-AI/walk-these-ways has `scripts/actuator_net/` (MLP 6 → 32 → 32 → 1, softsign, on 3
  position errors + 3 velocities, trained on the Go1's estimated motor torque) — the source of Isaac Lab's Go1 net.

## Design

### The constraint: position data only

The chirps give commanded and measured joint position (motor encoder through the tendon calibration) at 40 Hz (v1 /
v1b) or 20 Hz (v2), on a free-moving joint. That identifies the **command → motion** dynamics, not forces: in free
motion the servo torque only balances the link's inertia and friction, so the torque scale (and the stall / contact
force) is not observable. The teammate's PD fit has the same limit — on these nearly massless links only the ratio
damping / stiffness shapes the chirp, which is likely why the example fit ran to the stiffness bound (15).

### Why not an explicit torque net

An explicit actuator computes the torque once per physics step from the current state; PhysX then integrates it
explicitly. The finger links have 2e-6 to 7e-5 kg m² about their joints (wrist 1.5e-3; from the v2 USD). Explicit
damping d is stable only for `dt · d / I < 2`, i.e. d < 0.02 Nms/rad on the fingers, while the drives need 0.25-2
Nms/rad (index_mcp: `dt · d / I` ≈ 100 with the fitted damping, ≈ 300 with `orca_v2.py`'s). Any torque net that
reproduces the servo's stiffness / damping would blow up at 120 Hz unless ~1e-2 kg m² of armature is added to every
joint (fingertip inertia then ~3 kg against a 0.11 kg cube, a much bigger change to the sim than the actuator itself),
and its torque scale would only be anchored by that made-up armature.

### Options considered

| Option | Verdict |
|---|---|
| (a) net predicts the next position / velocity, inverted through the identified joint dynamics into a torque | Inverting a near-massless first-order joint in one 8 ms step needs a gain of ~(d + k dt) / (k dt) ≈ 13 on the predicted increment: a very stiff position servo whose contact force is set by extrapolating a free-motion predictor. Rejected |
| (b) net outputs torque, trained by backpropagating through a differentiable 1-DoF joint model | The right training principle (we use it), but with an explicit torque output it is unstable at 120 Hz (above) |
| **(c) chosen: (b) with the stiff part kept in PhysX** | Delay + play + learned correction shape the **target** of the PhysX implicit PD every physics step; the net is trained by backpropagating through a replica of that implicit PD. Its correction `delta` acts like a torque `k · delta`, but PhysX keeps it stable for any gains |

### The model (per physics step and joint)

```
q*_L   = joint target L physics steps ago              servo / bus latency                    per joint type
y      = clip(y, q*_L - b, q*_L + b)                   servo play (stops short at reversals)  per joint type
delta  = net(history of q*_L - q and dq/dt)            |delta| < 5 deg, shared by all joints
target = y + delta                                     -> PhysX implicit PD: stiffness k (orca_v2.py), damping k * tau
```

- **Physical prior** (3 numbers per joint type, fitted first): delay L, play half-width b, time constant tau. The
  play acts on the target, upstream of the drive: inside the band the target holds still and the joint is as stiff as
  the PD (a friction-locked servo), and there is no feedback from the joint state into the play.
- **Force scale** from `orca_v2.py`: stiffness, effort limit, friction, armature are copied from the implicit cfg it
  replaces; damping is `stiffness · tau`. Free motion fixes tau, not k, so the fitted motion stays the same when the
  teammate changes stiffness, and contact forces stay those of the current sim (blocked joint: `k · (|q* − q| − b)`,
  clipped at the effort limit).
- **Residual net**: MLP 20 → 32 → 32 → 1, softsign, on 10 taps (0-200 ms) of the error to the delayed target and of
  the joint velocity (Hwangbo-style inputs; the delay is explicit, so the history only needs the servo's own response
  time), output bounded by `5° · tanh(·/5°)`. Bounded and with a stable drive underneath, it cannot make the joint
  blow up: the target stays within ±5° of the play output.
- Only signals the servo already has: no input uses the current (undelayed) command.

### Training (CPU, ~10 min per run)

1. Load the chirps, remove each recording's static offset (mean measured − commanded: calibration / pre-tension, not
   dynamics; one v1b file has a −1148° encoder offset), convert to rad.
2. **Rate mismatch**: the command is zero-order held by its timestamps on the 120 Hz physics grid (3 steps per command
   at 40 Hz, 6 at 20 Hz, jitter handled), the model is simulated at 120 Hz, and the simulated position is read at each
   measured sample's own timestamp (before that sample's command acts). The same code handles the v2 20 Hz chirps,
   whose command steps match the RL policy's.
3. **Replica of the sim joint**: implicit Euler of `I q̈ = k (target − q) − d q̇ − friction` (PhysX treats drives
   implicitly), I from the v2 USD, k / friction / effort limit read from `orca_v2.py`.
4. **Prior fit** per joint type: exhaustive grid over delay (0-24 steps) × tau × play, then a finer grid, minimising
   the free-run position MSE.
5. **Net**: full free-run rollouts of all training chirps from rest, loss = position MSE at the measured samples +
   0.01 · mean(delta²), truncated BPTT over 1 s segments, Adam 3e-3 → 3e-4 (cosine), 120 epochs; last layer
   initialised to zero (starts as the prior). Input normalisation from a prior-only rollout, stored in the file.
6. Export one TorchScript file (`servo_net.pt`, 24 kB) with the per joint type fit in its `meta.json`.

## Data

- `scripts/motor_param_characterization/202506_collected_realworld_data/` (ORCA v1, 16 chirps, 2 per joint type) and
  `202510_orcav1b_realworld_data/` (ORCA v1b, 8 chirps, 1 per type): 20 s log chirps 0.1-5 Hz, ±10°, 40 Hz control.
- v1b thumb names are mapped by position in the chain as in `orca_v2.py` (v1b thumb mcp / abd / pip / dip → v2 cmc /
  abd / mcp / dip); the index model stands for all four fingers.
- Excluded by default: `chirp_index_mcp_p1200_*` (servo P gain changed to 1200, not the default setting).
- None of this is the v2 hand. The task PDF says to record new v2 chirps at 20 Hz.

## Offline results

Free-run simulation of each chirp from rest with the same 1-DoF replica for all models; Bode gain / phase from
`signal_processing.py` (as `optimization.py` / `plot_chirp_utils.py`) over 0.1-4.5 Hz, mean |error| against the
measured FRF. **PD** = the current sim (`orca_v2.py` stiffness / damping, no delay); **prior** = delay + play + tau;
**net** = prior + residual net (what the actuator runs).

**Held out: fit on v1 (15 chirps), test on v1b (8 chirps; a different hand and session).** The PD baseline uses
`orca_v2.py`'s values, which were copied from the v1b fit, so the baseline is in-sample here and the net is not.

| joint type | RMSE deg: PD / prior / net | gain err dB: PD / prior / net | phase err deg: PD / prior / net |
|---|---|---|---|
| wrist | 2.04 / 1.48 / 1.39 | 2.18 / 1.54 / 1.52 | 26.7 / 8.2 / 7.4 |
| thumb_cmc | 2.37 / 2.20 / 2.09 | 5.48 / 2.37 / 2.59 | 12.9 / 12.1 / 10.9 |
| thumb_abd | 2.47 / 1.72 / 1.59 | 2.23 / 1.92 / 1.71 | 24.0 / 10.9 / 10.4 |
| thumb_mcp | 2.16 / 1.66 / 1.58 | 4.80 / 1.61 / 1.78 | 15.3 / 11.8 / 8.7 |
| thumb_dip | 2.25 / 1.08 / 0.94 | 3.48 / 1.27 / 1.06 | 20.2 / 6.3 / 5.9 |
| index_abd | 2.03 / 1.09 / 1.01 | 2.46 / 1.18 / 1.34 | 21.1 / 9.7 / 8.1 |
| index_mcp | 2.05 / 1.44 / 1.38 | 4.18 / 1.68 / 1.89 | 13.2 / 9.0 / 8.6 |
| index_pip | 2.25 / 1.06 / 1.05 | 5.38 / 1.40 / 1.60 | 15.1 / 5.1 / 3.9 |
| **mean** | **2.20 / 1.47 / 1.38** | **3.77 / 1.62 / 1.68** | **18.6 / 9.1 / 8.0** |

In-sample on the 15 v1 chirps: RMSE 2.09 / 1.19 / 1.04 deg, gain 3.34 / 1.25 / 1.20 dB, phase 18.5 / 7.6 / 6.7 deg.

Prior fitted on v1:

| joint type | delay | tau | play | | joint type | delay | tau | play |
|---|---|---|---|---|---|---|---|---|
| wrist | 50 ms | 21 ms | 0.0° | | thumb_dip | 42 ms | 17 ms | ±2.7° |
| thumb_cmc | 42 ms | 19 ms | ±3.3° | | index_abd | 33 ms | 30 ms | ±2.6° |
| thumb_abd | 33 ms | 25 ms | ±1.9° | | index_mcp | 33 ms | 36 ms | ±1.9° |
| thumb_mcp | 25 ms | 19 ms | ±3.1° | | index_pip | 33 ms | 30 ms | ±2.8° |

(`orca_v2.py`'s PD time constants are 71-142 ms.)

**History length.** A 0.5 s history (13 taps up to 60 physics steps; 2603.04351 argues that slow servos need long
context) gives the same held-out numbers as the default 0.2 s (RMSE 1.38 deg, gain 1.67 dB, phase 8.0 deg;
`results/v1_to_v1b_hist500ms/`), so the default stays at 0.2 s. Presumably the explicit delay and the play state
already carry the long-memory part that their Transformer has to learn.

Reading the numbers:

- The prior is most of the improvement; the net's gain is small and consistent in RMSE / phase, not in gain. With
  20-40 s of chirp per joint type and one excitation amplitude, a bigger net has little to learn from.
- The worst joint, thumb_cmc, is a hand-to-hand difference, not a model error: the v1 thumb had ±3.3° of play, the v1b
  thumb almost none (`results/v1_to_v1b/test/bode_thumb_cmc.png`). Play depends on tendon tension, so it must be fitted
  on the hand that is deployed, and it is worth randomising (`play_scale_range`).
- The current PD fit misses mostly in phase (no latency) and at high frequency (too slow: −7 to −12 dB at 4 Hz).

**Deployed weights** (`actuator_net_weights/servo_net.pt`): same recipe fitted on all 23 v1 + v1b chirps
(`results/v1_v1b_all/`). In-sample: RMSE 2.13 / 1.22 / 1.10 deg, gain 3.51 / 1.27 / 1.25 dB, phase 18.4 / 7.6 / 7.0 deg.
Its prior:

| joint type | delay | tau | play | | joint type | delay | tau | play |
|---|---|---|---|---|---|---|---|---|
| wrist | 50 ms | 23 ms | ±0.2° | | thumb_dip | 42 ms | 17 ms | ±3.0° |
| thumb_cmc | 42 ms | 21 ms | ±2.4° | | index_abd | 33 ms | 29 ms | ±2.4° |
| thumb_abd | 33 ms | 23 ms | ±2.6° | | index_mcp | 42 ms | 21 ms | ±1.6° |
| thumb_mcp | 33 ms | 21 ms | ±2.3° | | index_pip | 42 ms | 20 ms | ±2.6° |

Together, delay + play + lag give the ~90-110 ms of apparent latency that the chirp phase shows (−40° at 1 Hz,
−135° at 4 Hz).

Plots (measured / PD / prior / net; Bode + low-frequency and end-of-chirp traces): `scripts/actuator_net/results/
v1_to_v1b/test/bode_<joint type>.png`, per-recording metrics in `metrics.csv` next to them.

## Retrain on the ORCA v2 chirps

1. Record the 8 joint types on the v2 hand (task PDF §7.1). Ideally record **two** chirps per joint type (or a second
   one at ±5°) so one can be held out, plus some larger / faster moves (steps, multi-sine) if the recorder allows: the
   model only knows ±10° chirps up to 5 Hz.
2. Check the result on the old model first (how far v2 is from v1 / v1b):
   `python scripts/actuator_net/evaluate.py --model source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net_weights/servo_net.pt --data <v2 dir> --run_name v1b_model_on_v2`
3. Train. With two chirps per type: `--train <dir A> --test <dir B>`. With one, hold out the last 25 % of each chirp:
   `python scripts/actuator_net/train.py --train <v2 dir> --holdout_tail 0.25 --run_name v2_tail`. That is a hard,
   pessimistic test: the tail is the only part above 1.9 Hz, so the fit sees no high frequencies and cannot separate
   delay from lag (smoke test on one v1b chirp per type: the wrist fit drops its delay to 0 ms and the tail RMSE barely
   beats the PD). Hence the advice to record two chirps. Then refit on everything and deploy:
   `python scripts/actuator_net/train.py --train <v2 dirs> --run_name v2_all --deploy`
4. v2 files use v2 joint names (`thumb_cmc`, ...); `--naming auto` detects this (v1b naming has `thumb_pip`).
5. Retrain after `orca_v2.py` changes (the teammate's fit): the prior keeps the fitted time constant when stiffness
   changes, but friction / stiffness and the wrist inertia / stiffness enter the fit. The actuator warns when the
   stiffness differs from the one in the weights.
6. Run `scripts/actuator_net/replay_chirp_isaac.py` on a GPU node ([below](#check-in-isaac-sim-before-rl)).

## Enable it

New files only; the two snippets below go into the ablation env cfg and `__init__.py` (not applied).

`orca_v2_repose_ablation_env_cfg.py`:

```python
from .actuator_net import servo_net_actuators


@configclass
class OrcaV2ReposeAsymActNetEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with the chirp-fitted servo model on all 17 joints (actuator_net.py, docs/ablation/actuator_net.md): per joint
    type command delay + servo play + PD time constant + bounded learned target correction on the implicit PD drive.
    Stiffness / effort limit / friction stay orca_v2.py's. Policy obs unchanged. Not to be stacked with -delay (the
    servo delay is already in the actuator)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = servo_net_actuators(self.scene.robot.actuators["fingers"])
```

`__init__.py`, in the ablation loop: `("orca-v2-inhand-repose-asym-actnet-v0", "OrcaV2ReposeAsymActNetEnvCfg"),`

Knobs (keyword arguments of `servo_net_actuators`, fields of `ServoActuatorNetCfg`):

| Knob | Default | Meaning |
|---|---|---|
| `joints` | `".*"` | regex of the joints that get the servo model; the others keep the implicit PD group |
| `network_file` | `actuator_net_weights/servo_net.pt` | weights + per joint type fit |
| `residual_scale` | 1.0 | 0 = physical prior only (delay + play + tau), an ablation of the net itself |
| `delay_jitter_steps` | 0 | per-episode, per-joint ± jitter of the delay in physics steps (DR; e.g. 2 = ±17 ms) |
| `play_scale_range` | (1, 1) | per-episode, per-joint scale of the play (DR; e.g. (0, 1.5) for re-tensioned hands) |

Interactions with the rest of the env:

- **Delay ablations.** `-delay` (`DelayedEMAJointPositionToLimitsActionCfg`, 50-100 ms) models the whole latency; the
  actuator already has the servo's. Do not stack them. The planned *fixed action delay* (next-tick actuation, one
  policy step) is a different part of the chain (policy node) and does stack.
- **Gain DR** (`randomize_actuator_gains`, stiffness ×0.75-1.5, damping ×0.3-3.0) keeps working: it scales the servo
  model's stiffness and damping (around k · tau) and writes them to PhysX.
- **`domain_randomization_params`** (critic of `-asym-dr`, RMA) reads joint stiffness / damping ratios: still valid
  (the actuator is implicit, damping default = k · tau).
- Physics dt must stay 1/120 s (delays and history taps are counted in physics steps).

## Check in Isaac Sim before RL

`scripts/actuator_net/replay_chirp_isaac.py` (GPU node, not run yet) replays a recorded chirp on one joint with the
implicit PD and with the servo model side by side and prints, for each, the RMSE against the measured joint and
against the offline replica. The second number tells whether the offline results carry over to PhysX; it should be
well below the model errors above (~0.1° or less). It also writes CSVs for `plot_chirp_utils.py`.

What can be checked offline already: `example_optimization_output/log.csv` holds Isaac Sim's FRF error (the
`optimization.py` metric) for two parameter sets on the v1 index_mcp chirp. The replica, run through the same synthetic
20 Hz chirp and metric:

| parameter set (Isaac Sim error) | replica, no friction | replica, friction as Coulomb effort |
|---|---|---|
| k 15, d 1.0, friction 0.50 (0.025) | 0.060 | 0.023 |
| k 6.8, d 1.06, friction 0.36, armature 0.025 (0.071) | 0.077 | 0.166 |

Sub-stepping the implicit drive (8 substeps instead of 1) changes these by < 10 %, so the drive integration is not
the issue; **joint friction is**: neither friction model matches both points (the log is from 2025-09, possibly an
Isaac Sim version where friction was a load-dependent coefficient). With `orca_v2.py`'s current friction (≤ 0.105 Nm, a ≤ 0.5° deadband) this
matters little for the fits above, but it will if the teammate's refit raises friction. The replay script settles it.

## Open risks

- **Not v2 data.** All fits are on v1 / v1b, 40 Hz. Delay, play and time constants must come from the v2 chirps
  (20 Hz, same rate as the policy). Until then the deployed weights are a placeholder for wiring and RL smoke tests.
- **Replica vs PhysX.** The offline model is a 1-DoF implicit-Euler replica: PhysX's TGS solver, joint friction
  semantics (Isaac Sim 5: static / dynamic friction efforts; the `orca_v2.py` comment reports load-dependent behaviour),
  link coupling and the effort limit are approximated. Run the replay script first.
- **Narrow excitation.** One amplitude (±10°), 0.1-5 Hz, free motion. RL commands bigger, faster moves and the
  fingers touch the cube; there the model extrapolates (bounded: ±5° of target correction on top of delay + play +
  PD). A blocked joint gets the PD's force minus `k · b`, which is plausible for tendon slack but not measured.
- **Force scale not identified.** Contact forces are those of `orca_v2.py`'s stiffness / effort limit, as in the
  current sim; only a force measurement (as in 2603.04351) would fix them.
- **Measured position = motor encoder.** The chirps see the motor side; tendon stretch under load is invisible.
- **Delay partly from the recorder.** The fitted delay includes whatever latency the recording loop adds between
  writing a command and reading the position; check `chirp.py`'s loop order when `rwr_faive_system` is released and
  compare with the policy node's loop.
- **Small net gain.** On held-out data the net adds little over the 3-parameter prior. If the v2 data shows the same,
  `residual_scale = 0` (prior only) is the simpler model to deploy; the RL ablation can compare both.

## Files

| File | What |
|---|---|
| `scripts/actuator_net/chirp_data.py` | CSV loading, joint naming, 120 Hz zero-order hold, `orca_v2.py` parameters, USD link inertias |
| `scripts/actuator_net/servo_model.py` | PhysX-style implicit PD replica, `ServoNet`, rollout, prior fit, TorchScript export |
| `scripts/actuator_net/train.py` | prior fit + net training + export + evaluation |
| `scripts/actuator_net/evaluate.py` | RMSE / Bode comparison vs the implicit PD, plots; evaluates a saved model on new chirps |
| `scripts/actuator_net/replay_chirp_isaac.py` | Isaac Sim replay check (GPU, not run yet) |
| `scripts/actuator_net/results/v1_to_v1b/` | held-out run: weights, metrics, Bode plots |
| `scripts/actuator_net/results/v1_v1b_all/` | run of the deployed weights (in-sample metrics, plots) |
| `scripts/actuator_net/results/v1_to_v1b_hist500ms/` | 0.5 s history variant (metrics only) |
| `source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net.py` | `ServoActuatorNet`, `ServoActuatorNetCfg`, `servo_net_actuators()` |
| `source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net_weights/servo_net.pt` | deployed weights (v1 + v1b) |
