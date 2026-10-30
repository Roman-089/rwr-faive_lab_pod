# Rewards (rotlin, rotsoft, drop penalty, planned variants)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

Our policies rotate at ~6 of 14 RPM. Is the reward the bottleneck (speed barely rewarded, wobble punished hard,
speed-independent survival bonus), and do drops come from a missing drop penalty? Training rewards of the variants are
not comparable with B's; compare by eval (RPM / RA / DFSR).

## Current reward (baseline)

Weights: direction 60, magnitude 30 (σ 0.2), position 15 (σ 0.02), ang. acc 1, joint_vel −1e-5, action −2e-4,
action_rate −0.075.

Terms, from the code (`mdp/rewards.py`): direction 60 · cos(ω, ω*) (**independent of speed**); magnitude
30 · exp(−| |ω| − |ω*| | / 0.2) (Laplacian, not Gaussian); position 15 · exp(−d / 0.02 m); ang. acc. 1 · exp(−|Δω| / 0.5);
small joint-vel / action / action-rate penalties. ω = finite difference of the cube quaternion. Commands: |ω*| sampled in
60–100 % of 1.5 rad/s about ±z, resampled every 8 s (so the eval's 1.5 rad/s is the top of the training range).

Early ideas (old notes §4):

| Idea | Values |
|---|---|
| Explicit drop penalty / termination reward | −50 … −200 on `object_dropped` |
| Position tracking σ | 0.02 → 0.03 (looser) |
| Action-rate penalty (smoothness for the real servos) | −0.075 → −0.15 |
| Magnitude vs direction weighting | 30/60 → 45/45 |

## Literature: reward structures for in-hand rotation (subagent, 2026-10-28)

| Work | Rotation reward | Stability / pose | Energy / smoothness | Drops | Curriculum | Source |
|---|---|---|---|---|---|---|
| HORA 2210.04887 | `clip(ω·k, ±0.5)` ×1 (speed up to a clip) | lin. vel −0.3‖v‖₁, pose −0.3 | torque −0.1‖τ‖², work −2.0 | reset, no penalty | – | code (public) |
| RotateIt / Vision+Touch 2309.09979 | same, + off-axis −0.1‖ω×k‖₁ | lin. vel −0.3, pose | work −2.0, torque −0.1 | reset | off-axis weight annealed 0 → −0.1 | paper |
| LEAP_Hand_Sim | `clip(ω_z, ±0.25)` ×1.25 | lin. vel −0.3, pose −0.1 | torque −0.1, work −1.0 | −10 per fallen step + reset | – | code |
| Aero Hand 2608.28578 / Playground `rotate_z` | unclipped 1·ω_z | none | action rate −1 | −100·dt on termination | – | code; only 0.51 rad/s in sim |
| ORCA PT 2605.21330 | **same as ours** (dir 60, mag 30, σ not given) | position 10 · exp(−d/0.02) | as ours | none | – | paper only; 11.8 RPM real |
| DLR Sievers 2022 2204.03698 | λ·Δθ, clipped ≈1 rad/s | −\|Δz\|, −\|Δxy\|, off-axis | action low-pass | reset | gravity 5→100 %, filter 0.12→0.5 s | paper |
| DLR 2024 2411.13148 | `min(Δθ, θ_clip)`, clip set by target speed | position progress, q⁴ | – | reset | horizon | paper: **"velocity-matching rewards failed"** |
| AnyRotate 2405.07391 | keypoint dist + 5·clip(Δθ·k, ±0.025/step) + bonus | pose −0.5, ω cap 0.6 | work −0.1, torque −0.05 | −50 | penalties scaled up with performance | paper |
| DeXtreme / IsaacLab reorient | 1/(\|rot err\|+0.1), +250 success | position −10·d | action −2e-4, rate −0.01 | 0 | ADR | code |

Other reward-related literature: AnyRotate: goal-increment keypoint reward beat angular-velocity reward; termination
penalty 50. Aero Hand: reward ω_z − 100·fall − ‖Δa‖². LEAP: rotation reward clipped ±0.25. Faive: reward from
finite-differenced orientation (sim ang. vel exploited). Ranked ideas (2) large drop penalty + clipped rotation reward
and (9) finite-difference angular velocity reward from the [in-hand rotation review](literature.md#in-hand-rotation-literature-review).
Drop-penalty code: AnyRotate not found; Aero Hand [TetherIA/aero-hand-open](https://github.com/TetherIA/aero-hand-open)
has `sim_rl/` (MuJoCo Playground), rotation task unverified.

**Diagnosis (why we under-rotate, ~6 of 14 RPM):** per step, any slow rotation in the right direction earns ~75 of
~105 (direction 60 + position 15). At 0.63 rad/s the magnitude term pays 30·e^(−0.87/0.2) ≈ 0.4 with a gradient of
~2 per rad/s, while 1 cm of position error costs ~6. Speed is barely rewarded, translational wobble from fast gaiting
is punished hard, and a speed-independent survival bonus makes holding on worth more than rotating. Every other work
except ORCA PT rewards ω·k **linearly** (constant gradient) and clips it; DLR 2024 reports that velocity-matching
rewards failed and clipping at the target speed worked. Prediction: the queued B + −2000 drop penalty (weight; −100
return after dt = 0.05) makes the policy slower and safer, not faster.

**Reward ablations (on top of B, ranked):**
1. `asym-rotlin` — direction + magnitude → 90 · clip(ω·k, −0.5, |ω*|) / |ω*| and −5 · off-axis ‖ω⊥‖₁ (HORA/LEAP
   form, DLR 2024 clip, RotateIt off-axis). ~30× the speed gradient below the target, nothing for overspeed. Risk: drops.
2. `asym-rotsoft` — keep the forms, magnitude σ 0.2 → 0.75 and weight 30 → 60, direction 60 → 30. Risk: precision.
3. (next) position σ 0.02 → 0.05, weight 15 → 5, + object linear-velocity penalty −20·‖v‖₁ (HORA): let the cube wobble
   but not slide.
4. (next) rotlin + moderate drop penalty −200 (−10 after dt; LEAP −10/step, Aero −5).
5. (if rotlin is jerky) HORA work / torque penalties, annealed in over the first 30 % (RotateIt, AnyRotate).

## Runs

Terms in `rewards_ablation.py`. Drop penalty: `is_terminated_term` on `object_dropped`, weight −2000 per step × dt 0.05
= −100 return.

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) |
|---|---|---|---|---|---|
| `abl_B_droppen` | `-asym-droppen-v0` | B + −100 return on drop (AnyRotate / Aero Hand) | 29835278 / A40 | 5k | queued |
| `abl_B_rotlin` | `-asym-rotlin-v0` | B, rotation reward linear in speed, clipped at the command (HORA/LEAP/DLR 2024) + off-axis penalty | 29838450 / L40S | 5k | done (2026-10-29 03:30) |
| `abl_B_rotsoft` | `-asym-rotsoft-v0` | B, magnitude kernel σ 0.2 → 0.75, weights mag 30 → 60 / dir 60 → 30 | 29838451 / A100 | 5k | done (2026-10-28 17:05) — **failed: reward hack** |
| planned: rotlin + drop penalty | – | ablation 4 above (−200 weight / −10 return) | – | 5k | planned |
| planned: position / velocity variant | – | ablation 3 above | – | 5k | planned |
| parked (wave 2) | `-droppen-v0` | −100 return on drop, on the baseline A | – | 5k | registered, parked ([sim2real](sim2real.md)) |
| TBD | | clipped rotation reward, finite-diff ω reward | | 5k | TBD |

## Results

**`abl_B_rotsoft`** (nominal / perturbed): the policy does not rotate the cube.

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 0.03 / −0.03 | 7.8 / 7.1 | 88 / 75 | 0.12 / 0.30 |
| 2000 | 0.17 / 0.14 | 11.1 / 7.0 | 97 / 73 | 0.03 / 0.47 |
| 3000 | −0.02 / 0.09 | 6.7 / 7.8 | 94 / 78 | 0.06 / 0.27 |
| 4000 | −0.06 / 0.06 | 7.5 / 7.5 | 92 / 72 | 0.09 / 0.36 |
| 4999 | −0.02 / −0.01 | 6.8 / 5.8 | 92 / 77 | 0.09 / 0.34 |

Training reward was high (~1480, drops 8 %), so the policy found a better-paying behaviour than rotating.

### Log

- **2026-10-28 02:00 — drop timing (A, model_9999, nominal eval, 116 drops):** time since spawn 0–2 s 12 %, 2–5 s 15 %,
  5–10 s 16 %, 10–20 s 23 %, 20–40 s 22 %, 40–61 s 12 %. Drops happen throughout the episode, not mainly while the cube
  settles after the reset, so **grasp-cache resets (HORA/LEAP) are deprioritized** (they target initial-grasp
  failures); drop penalty and entropy runs address sustained-grip drops. Re-check on B's rollout once its eval runs.
- **2026-10-28 09:30 — reward review** (literature table above): our direction term pays the same for any speed and the
  magnitude kernel is ~0 at our policies' speed, so speed is barely rewarded; the literature rewards speed linearly and
  clips it. Queued `abl_B_rotlin` (linear clipped progress) and `abl_B_rotsoft` (reweighted current terms).

- **2026-10-29 03:30 — `rotlin` wins the reward wave by a wide margin** (nominal / perturbed):

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 5.18 / 6.16 | 45.5 / 41.7 | 88 / 53 | 0.20 / 1.06 |
  | 2000 | 7.63 / 8.92 | 57.1 / 54.5 | 73 / 39 | 0.34 / 1.13 |
  | 3000 | 9.55 / 10.32 | 70.5 / 62.5 | 78 / 39 | 0.28 / 1.03 |
  | 4000 | 9.79 / 11.49 | 70.9 / 65.8 | 78 / 33 | 0.27 / 1.52 |
  | 4999 | **10.41 / 11.07** | 74.0 / 67.8 | 84 / 44 | 0.22 / 1.09 |

  - **+91 % RPM over B with the same everything else** (10.41 vs 5.45), +42 % over the best entropy run (7.33) and above
    the privileged teacher (7.44); still rising at 5k. The speed term finally pays for speed: mean ω_z in the commanded
    direction 1.09 rad/s (ent5e-3: 0.77), i.e. 73 % of the 1.5 rad/s command.
  - **Costs:** more drops (DFSR 84 % vs 88 %, DC 0.22 vs 0.12) and much less robust to untrained delay / joint offsets
    (perturbed DFSR 44 % vs 55 %, DC 1.09 vs 0.75). The cube also wobbles more: mean off-axis |ω_xy| 2.15 rad/s vs
    1.12, so the off-axis weight (−5) is too low to keep the rotation clean. Drop timing: 14 drops, 10 of them within
    5 s of spawn (reset artifacts, as for the others).
  - Direction: here −z is faster (1.18 vs 1.00 rad/s), unlike 11 of the 12 earlier policies (all on the old reward).
    The direction preference depends on the policy, not only on the hand; `-onedir` keeps +z.
  - Training reward is not comparable to B's (different terms).

- **2026-10-29 13:45 — the drop penalty makes B faster, not only safer** (`abl_B_droppen`, entropy 1e-4, one seed),
  nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 5.85 / 6.43 | 64.8 / 62.6 | 75 / 69 | 0.33 / 0.42 |
  | 2000 | 7.05 / 6.98 | 75.8 / 69.1 | 86 / 67 | 0.17 / 0.47 |
  | 3000 | **8.34** / 8.23 | **82.6** / 72.7 | **94** / 63 | **0.08** / 0.53 |
  | 4000 | 7.82 / 7.75 | 79.8 / 74.3 | 91 / 63 | 0.14 / 0.45 |
  | 4999 | 7.94 / 8.36 | 80.0 / 77.2 | 83 / 61 | 0.27 / 0.55 |

  vs B (5.45 RPM, 88 % DFSR, two seeds): **+2.5 RPM** (+2.9 at the best checkpoint, which is also 94 % drop-free), well
  beyond the ±0.7 RPM seed spread, and more robust under the perturbed eval (61 vs ~53 % DFSR, 8.4 vs ~6.2 RPM). The
  penalty only adds −100 to the return of a dropped episode, yet the policy ends up both firmer and faster:
  presumably a value function that sees drops as costly favours a secure grasp, which then supports a faster gait.
  The rotlin and onedir variants with the drop penalty are running / queued.

- **2026-10-29 14:50 — rotlin + entropy 5e-3** (`abl_B_rotlin_ent5e-3`), nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 8.90 / 9.62 | 69.4 / 62.9 | 72 / 44 | 0.39 / 1.70 |
  | 2000 | 9.32 / 9.34 | 70.6 / 61.6 | 75 / 31 | 0.31 / 1.69 |
  | 3000 | 8.99 / 9.88 | 71.7 / 64.3 | 81 / 48 | 0.22 / 1.05 |
  | 4000 | 9.34 / 9.92 | 73.6 / 66.8 | 83 / 61 | 0.27 / 0.67 |
  | 4999 | 9.50 / 10.43 | 76.1 / 70.8 | 81 / 58 | 0.27 / 1.00 |

  Entropy does not add speed on rotlin (9.50 vs 10.41 RPM at 5k, one seed each, within ~1.5 RPM of seed noise), but it
  learns much faster (8.9 vs 5.2 RPM at 1000 iterations) and holds up better under the perturbed eval (58 vs 44 %
  drop-free). It also confirms rotlin's level with a second, independent run (~9.5–10.4 RPM). W&B showed the run as
  "crashed" between the end of training and the eval upload: Isaac Sim closes the process before wandb's own shutdown,
  and the eval's upload then reopens and finishes the run (all completed runs went through this state).

- **2026-10-29 15:20 — rotlin + entropy 5e-3 + drop penalty** (`abl_B_rotlin_droppen_ent5e-3`), nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 7.54 / 8.08 | 64.8 / 59.1 | 89 / 58 | 0.13 / 0.61 |
  | 2000 | 8.37 / 9.02 | 69.9 / 65.4 | 92 / 69 | 0.11 / 0.56 |
  | 3000 | 8.03 / 8.39 | 69.7 / 62.9 | 89 / 69 | 0.22 / 0.73 |
  | 4000 | 8.09 / 8.61 | 69.8 / 63.5 | 88 / 48 | 0.16 / 0.86 |
  | 4999 | 8.29 / 8.17 | 70.8 / 60.8 | 89 / 44 | 0.17 / 0.92 |

  On rotlin the drop penalty **trades speed for safety** (vs `rotlin_ent5e-3`: −1.2 RPM, +8 pp drop-free nominal, 0.17
  vs 0.27 drops per trial), unlike on the old reward, where it gave both (+2.5 RPM). Plausibly the old reward barely paid
  for speed, so a firmer grasp was free speed there, while rotlin already pushes speed and the penalty now pulls toward
  caution. It does **not** help under the perturbed eval (44 % drop-free, vs 58 % without it). One seed each; the
  1.2 RPM gap is at the edge of seed noise. For the demo (long without a drop) the trade may still be worth it; the
  real hand decides.

- **2026-10-29 15:50 — off-axis penalty −15 kills the rotation** (`abl_B_rotlin_ent5e-3_offaxis15`): 0.19 RPM at 5k
  (RA 37 %, 91 % drop-free), every checkpoint below 0.3 RPM. The cube wobbles much less (mean off-axis |ω_xy| 0.39 vs
  1.74 rad/s with −5), because the policy just holds it: turning the cube always creates some off-axis angular velocity
  (contact switches, finite-difference ω), so at −15 that cost outweighs rotlin's reward (≤ 90 · 1 per step) and holding
  still is the safer optimum. Training logs looked healthy (long episodes, few drops), which is exactly what a holding
  policy produces: **check the rotation reward / commanded-axis speed in training, not only drops and episode length**.
  Keep −5; if wobble needs reducing, try −8 to −10 or penalize only off-axis speed above a dead band.

- **2026-10-30 03:00 — drop penalty, seed 1** (pod, resumed at it 2000), nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 2500 | 6.63 / 7.44 | 75.2 / 74.9 | 86 / 72 | 0.16 / 0.39 |
  | 3000 | 7.06 / 7.38 | 80.2 / 74.4 | 91 / 67 | 0.11 / 0.50 |
  | 3500 | 6.87 / 7.79 | 78.4 / 76.0 | 92 / 63 | 0.09 / 0.56 |
  | 4000 | 6.95 / 7.58 | 78.6 / 73.8 | 91 / 59 | 0.09 / 0.64 |
  | 4999 | 7.00 / 7.55 | 80.3 / 74.8 | 88 / 58 | 0.14 / 0.75 |

  **Replicated:** two-seed mean 7.47 RPM (7.94 / 7.00) vs B 5.45 (5.66 / 5.23) = **+2.0 RPM**, well beyond the ±0.7 seed
  spread, at the same or better drop-free rate (88 % / 83 % vs B 88 %) and more robust under perturbation (seed 1:
  58-72 % drop-free vs B ~53 %).

- **2026-10-30 03:40 — rotlin seed 1** (pod, resumed at it 2000), nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 2500 | 9.38 / 10.49 | 74.1 / 70.6 | 80 / 47 | 0.36 / 0.92 |
  | 3000 | 9.46 / 10.97 | 73.1 / 71.5 | 75 / 50 | 0.34 / 1.02 |
  | 3500 | 9.76 / 10.30 | 76.2 / 68.1 | 83 / 45 | 0.19 / 1.14 |
  | 4000 | 10.13 / 10.99 | 77.6 / 70.1 | 80 / 42 | 0.25 / 1.03 |
  | 4999 | **10.25** / 11.48 | 76.8 / 71.4 | 83 / 45 | 0.23 / 1.08 |

  **Replicated:** 10.41 / 10.25 RPM, mean 10.33 vs B 5.45 (+4.9 RPM, +90 %); drop-free 84 / 83 %. The weakness under
  perturbation replicates as well (44 / 45 % drop-free, ~1.1 drops per trial).

## Conclusions so far

- **`rotsoft` is a reward hack** (2026-10-28 17:10): ~0 RPM, RA ~7 %, while holding the cube safely (DFSR 92 %).
  The magnitude term rewards ‖ω‖ — the norm of the full 3-D angular velocity, **any axis and either sign** — and the
  direction term only the cosine. With the magnitude weight (60) above the direction weight (30) and a wide kernel
  (σ 0.75), wobbling / oscillating the cube in place pays more than rotating it about +z / −z. Lesson: a speed term must
  be measured *along the commanded axis and sign* (as `rotlin`'s ω·k is), never as ‖ω‖. The baseline weighting
  (direction 60 > magnitude 30, narrow σ 0.2) only avoided the hack by making speed nearly irrelevant.
- The training reward of a reward variant says nothing about rotation: always judge by eval.

## One direction, max speed (`-asym-onedir-v0`)

The task statement grades only the live demo: "more points if the hand turns the cube fast and for a long time without a
drop" (RWR26_Task1.pdf §2). Neither the commanded speed nor the direction switches are graded, and HORA / LEAP /
RotateIt all train one direction with a speed reward. Our policies are also slower (≤ 0.78 rad/s) than the slowest
training command (0.9 rad/s), so speed tracking never mattered in practice.

- **Direction:** +z. From the saved nominal rollouts (all 64 envs, sign alternating), +z is faster in 11 of 12 policies:

  | Policy | +z RPM | −z RPM |
  |---|---|---|
  | `abl_B_asym_ent5e-3` | 7.94 | 6.71 |
  | `teacher` | 7.65 | 7.24 |
  | `arch_rma_s1` / `s2` | 8.04 / 7.71 | 4.92 / 4.70 |
  | `abl_B_asym` / `seed1` | 6.34 / 6.80 | 4.98 / 3.66 |
  | `abl_C_asym_dr` | 6.29 | 5.86 |
  | `arch_asym_hist3` / `hist10` | 6.60 / 4.69 | 4.06 / 5.33 |
  | `arch_asym_lstm` | 5.83 | 4.41 |
  | `sanity_nodr` | 11.27 | 9.82 |

- **Command:** `commands_ablation.FixedDirectionRotationCommand`, always +1.5 rad/s about z. The observation layout is
  unchanged; deploy with `goal_ang_vel:=1.5 goal_flip_period_s:=0`.
- **Reward:** `rewards_ablation.rotation_progress_linear` = clip(ω·k, −0.5, 3.0) / 1.5, weight 90; off-axis −5; the
  rest as B. Clipped at 3 rad/s (2× the command, ~3× our best speed), so speed above 1.5 rad/s keeps paying.
- **The unclipped version was a reward hack** (`abl_B_onedir_ent5e-3`, killed 2026-10-29 at it 272): 99.9 % of
  episodes ended in a drop after ~45 steps (2.2 s) with the cube spinning at ~30 rad/s while it fell (magnitude error
  29 rad/s). A short fling paid more than holding the cube; the drop termination alone did not stop it. HORA's clip
  exists for this reason. Rerun with the 3 rad/s clip as `abl_B_onedir_clip3_ent5e-3`.
- **Base:** B (asymmetric critic, DR, single-observation MLP) + entropy 5e-3, the best deployable policy so far and
  exportable without the history wrapper. Run `abl_B_onedir_ent5e-3`; eval commands +z in every env.

### Result: `abl_B_onedir_clip3_ent5e-3` (2026-10-29 17:50), nominal / perturbed, +z only

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | **15.67** / 14.26 | 81.1 / 69.2 | 42 / 13 | 1.20 / 4.45 |
| 2000 | 14.90 / 13.69 | 79.0 / 66.9 | 44 / 22 | 1.50 / 3.55 |
| 3000 | 14.55 / 13.17 | 77.0 / 65.4 | 41 / 14 | 1.34 / 4.06 |
| 4000 | 15.17 / 14.59 | 79.7 / 69.5 | 52 / 22 | 1.00 / 3.27 |
| 4999 | 15.19 / 13.97 | 79.2 / 65.9 | 44 / 9 | 1.30 / 4.83 |

The fastest policy so far by far: 15.2 RPM, i.e. 1.59 rad/s about +z (above the 1.5 rad/s command; rotlin 1.09), with
the speed reward clipped at 3 rad/s instead of at the command. But it drops: 44 % drop-free and 1.3 drops per minute
(rotlin 84 % / 0.22), and almost always under the perturbed eval (9 % drop-free, 4.8 drops). The cube also tumbles
more (off-axis |ω_xy| 3.0 vs 2.2 rad/s). RPM counts through drops (respawn), so a fast-but-droppy policy scores well on
RPM and badly on the demo's "long without a drop". The drop-penalty version (`abl_B_onedir_droppen_ent5e-3`, running)
is the direct fix; a lower clip (2 rad/s) is the other knob.

### Result: `abl_B_onedir_droppen_ent5e-3` (2026-10-30 03:50), nominal / perturbed, +z only

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 13.61 / 12.97 | 74.5 / 63.9 | 73 / 33 | 0.34 / 1.67 |
| 2000 | 13.68 / 14.13 | 75.2 / 68.6 | 64 / 27 | 0.59 / 1.92 |
| 3000 | 15.20 / 15.06 | 78.1 / 68.9 | 61 / 23 | 0.70 / 3.59 |
| 4000 | 15.55 / 15.32 | 79.2 / 70.1 | 61 / 20 | 0.97 / 3.52 |
| 4999 | **16.41** / 15.07 | 82.8 / 69.9 | 63 / 19 | 0.64 / 2.94 |

Fastest policy so far (16.4 RPM = 1.72 rad/s). Versus the same onedir run without the drop penalty (15.19 RPM, 44 %
drop-free, 1.30 drops): the penalty halves the drops and is faster, as on B. Still too droppy for "long without a
drop" (63 % drop-free, 19 % perturbed), and drops rose during training (training drop rate 25 → 48 %). The 2 rad/s
clip version (pod, 14 % training drops) decides the demo setting.

## Next steps / open questions

- **`rotlin` is the new reward base.** Queued 2026-10-29: `abl_B_rotlin_ent5e-3` (+ entropy 5e-3), and the action-space
  runs moved onto it (`_alpha05`, `_relact`). The one-direction run (`-onedir`) already uses the unclipped version.
- Next on rotlin: a higher off-axis weight (−15) against the wobble; the teacher-student pipeline re-run with a rotlin
  teacher (the current teacher and students use the old reward).
- Then rotlin + drop penalty, position / velocity variant, HORA work / torque penalties if rotlin is jerky.
- Re-check drop timing on B's rollout.
- Grasp-cache resets (HORA `gen_grasp.py` + `cache/`; LEAP_Hand_Sim grasp cache per cube size): deprioritized, see log.

## Work penalty

`abl_B_rotlin_work` (`-asym-rotlin-work-v0`, 2026-10-30): rotlin plus HORA's work term (Qi et al. 2022,
arXiv 2210.04887), `rewards_ablation.joint_work_l1` = Σ|τᵢ·q̇ᵢ| over the 17 joints, weight −1.5. τ is the
implicit PD drive's `applied_torque` (≤ 0.66 / 0.78 / 1.29 Nm for abd / mcp / pip). HORA uses it, together with a
torque penalty, to get smooth finger gaits that transfer to hardware. Motivation: rotlin is our fastest deployable
reward (10.33 RPM, two seeds) but drops to ~45 % drop-free under the perturbed eval and its gait looks wobbly in the
renders. The weight is a first guess (aim: penalty ≈ 10–20 % of the rotation reward); check
`Episode_Reward/joint_work` against `rotation_progress` in the first iterations and rescale if it dominates or vanishes.
Expected result if it works: slightly lower nominal RPM, higher perturbed drop-free rate.

**Result at −1.5 (job 29851888, killed at iteration 467):** the term started at 2× the rotation reward (−3.9 vs 1.7
per episode-second) and the policy learned to let the cube drop: 100 % of episodes ended in a drop, rotation reward 1.7
where rotlin had 25 at iteration 500, and the work term shrank to −0.6 because the episodes got short. rotlin has no
drop penalty, so ending the episode early is the cheapest way to stop paying a per-step cost (the usual trap of
negative per-step rewards with early termination). Rerun at **−0.15** (`abl_B_rotlin_work015`, job 29851938): ~1/5 of
the rotation reward at the start, shrinking as the policy rotates. If that run still collapses, the next step is to
combine it with the drop penalty instead of shrinking it further.

## Speed cap: 2 vs 3 rad/s on onedir + drop penalty (2026-10-30)

Isaac eval, last checkpoint, nominal / perturbed (64 envs × 60 s):

| Run | RPM | DFSR % | DC |
|---|---|---|---|
| `abl_B_onedir_droppen_clip2_ent5e-3_r6k` (cap 2 rad/s, pod) | 10.09 / 9.68 | 91 / 41 | 0.11 / 1.36 |
| `abl_B_onedir_droppen_ent5e-3` (cap 3 rad/s) | **16.41** / **15.07** | 63 / 19 | 0.64 / 2.94 |
| `abl_B_rotlin_cont10k` (both directions, cap = command 1.5 rad/s) | 10.81 / 10.70 | 91 / 55 | 0.12 / 0.77 |
| `abl_B_rotlin` (5k, 2 seeds) | 10.33 / 11.3 | 84 / 45 | 0.22 / 1.08 |

- The 2 rad/s cap does not just clip the top speed: the policy settles at 10.1 RPM (1.06 rad/s, half the cap), the same
  speed as rotlin. With the 3 rad/s cap it settles at 1.72 rad/s. The cap sets how much extra speed is worth the
  extra drop risk, and the policy stops well below it in both cases.
- At 2 rad/s the onedir run is no better than rotlin 10k on any Isaac metric (equal nominally, worse perturbed:
  41 vs 55 % drop-free, DC 1.36 vs 0.77), and rotlin also turns both ways. **Decision: drop the 2 rad/s cap**;
  its second seed (A40 / A100) is cancelled before starting.
- The demo line stays at 3 rad/s, where drops are the problem (37 % of nominal and 81 % of perturbed episodes see a
  drop). Levers now queued: second seed (`abl_B_onedir_droppen_ent5e-3_seed1`), 10k training
  (`..._cont10k`), 3× drop penalty (`abl_B_onedir_droppen6k_ent5e-3`), and the privileged teacher → student path
  (`teacher_onedir_droppen_ent5e-3`).
- rotlin 10k vs 5k (same seed): +0.4 RPM (inside the ±0.7 seed spread), +6 points nominal and +10 perturbed
  drop-free, DC 0.22 → 0.12 nominal. Longer training mainly makes rotlin safer, not faster.
- Next point on the curve: cap 4 rad/s (`abl_B_onedir_droppen_clip4_ent5e-3`, A40).
- **Drop timing** (`scripts/eval/drop_timing.py`, nominal eval, drops per env-minute by time since (re)spawn):

  | Run | drops | 0–1 s | 1–2 s | 2–3 s | 3–5 s | 5–10 s | 10–20 s | 20–60 s |
  |---|---|---|---|---|---|---|---|---|
  | onedir + drop penalty (3 rad/s) | 41 | 2.94 | 3.76 | 1.96 | 1.01 | 1.16 | 0.42 | 0.30 |
  | onedir, no drop penalty (3 rad/s) | 83 | 3.35 | 0.88 | 0.92 | 2.46 | 2.31 | 0.96 | 0.90 |
  | rotlin 10k | 8 | 1.69 | 1.74 | 0.90 | 0 | 0 | 0.09 | 0.05 |
  | transformer student (rotlin) | 11 | 1.62 | 0 | 0 | 0 | 0 | 0.17 | 0.17 |

  The drop penalty cut the steady-state drop rate (≥ 10 s after spawn) by 2.5–3× (0.30–0.42 vs 0.90–0.96 per minute)
  but not the spin-up drops in the first 3 s (2–4 per minute, ~a third of its drops): the cube is lost while the
  hand accelerates it from rest to 1.7 rad/s. For a live demo a spin-up drop ends the attempt at once, so spin-up
  needs its own fix (teacher / student, or a reward that does not pay for speed before the grasp is stable).
- Two single-knob follow-ups on spin-up / position drops (2026-10-30 07:20). The base reward already has a cube
  smoothness term (`angular_acceleration`, exp(−|Δω/dt|/σ), σ 0.5, weight 1) and a position term
  (`object_position_tracking`, σ 2 cm, weight 15), both small next to the rotation term (90 × speed).
  `abl_B_onedir_droppen_acc10_ent5e-3` raises smoothness to 10 (should soften the spin-up; may cost steady-state speed if
  it also punishes the gait's wobble); `abl_B_onedir_droppen_pos45_ent5e-3` raises position to 45 (cube kept near the
  palm centre, fewer drops at any time).
- Spin-up ramp (2026-10-30 10:20): `abl_B_onedir_droppen_ramp_ent5e-3` (`-asym-onedir-droppen-ramp-v0`). The speed
  reward's cap rises linearly from 0.5 to 3 rad/s over the first 3 s of each episode (`rotation_progress_linear(...,
  ramp_s=3.0, ramp_start_vel=0.5)`, time from `episode_length_buf`), so a fast spin-up from rest pays no more than a
  gentle one; after 3 s the reward is the same as onedir-droppen. Tested on stubs: speed 2.5 rad/s at t = 0 / 1 / 2 / 10 s
  pays 0.5 / 1.33 / 2.17 / 2.5. The MLP has no clock, but the first seconds look different (hand near its reset pose,
  cube at rest), so it can learn a separate start-up behaviour.
