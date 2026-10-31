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

**Result at −0.15 (`abl_B_rotlin_work015`, job 29851938), nominal / perturbed:**

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 5.58 / 6.95 | 47.8 / 48.0 | 67 / 38 | 0.39 / 1.27 |
| 2000 | 6.57 / 7.80 | 53.6 / 52.9 | 86 / 44 | 0.16 / 0.80 |
| 3000 | 7.57 / 8.62 | 58.6 / 57.0 | 88 / 61 | 0.19 / 0.67 |
| 4000 | 10.12 / 9.88 | 69.1 / 61.1 | 88 / 61 | 0.16 / 0.73 |
| 4999 | 10.02 / 10.32 | 69.2 / 61.8 | 86 / 47 | 0.14 / 0.89 |
| `abl_B_rotlin` seed 0, 4999 | 10.41 / 11.07 | 74.0 / 67.8 | 84 / 44 | 0.22 / 1.09 |

Same nominal speed and drop rate (10.0 vs 10.3–10.4 RPM), slower to learn (7.6 vs 9.6 RPM at 3000), slightly slower
perturbed (10.3 vs 11.1–11.5) but more robust there: perturbed drop-free 61 / 61 / 47 % over the last three checkpoints
(mean 56 %) vs 39 / 33 / 44 % for rotlin seed 0, DC 0.67–0.89 vs 1.03–1.52. One seed, so suggestive. Since +z is the
line now and its weak point is the perturbed eval (19 % drop-free), the term is carried over:
`abl_B_onedir_droppen_work015_ent5e-3` (`-asym-onedir-droppen-work-v0`, L40S + pod).

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

## Decision: +z only (2026-10-30)

From 2026-10-30 every new ablation runs on the one-direction demo config (`-asym-onedir-*`, fixed +z command). The
bidirectional rotlin line is closed; its transformer student `student_pt_rotlin` (W&B p3dxn5pm) stays the fallback
deployable policy. Commanded one direction it gives +z 10.5 RPM / 91 % and −z 11.3 RPM / 81 % drop-free (nominal,
model_1999, 32 envs per direction; the drop-free gap is inside the ±7-point noise). Stopped for this: the bidirectional
α 0.5 run (A10, it ~850; re-queued on +z), `abl_B_rotlin_ent5e-3_offaxis8_r6k` (pod, it ~2000), and the held
re-distillation from the 10k rotlin teacher. +z was chosen from the per-direction eval of the old-reward policies (11 of 12
faster in +z); the rotlin-family policies are faster in −z (B 10k: 10.0 vs 11.6 RPM, teacher 10.7 vs 13.0), which is
noted but not acted on (a direction-specific policy can learn a direction-specific gait).

## Wrist motion (2026-10-30)

The user noticed in the onedir teacher's videos (742zn06f) that it moves the wrist a lot. Measured from the eval
rollouts (relative rotation forearm → carpals = the `right_wrist` joint, 4 rendered envs × 60 s; the tower is rigid to the
forearm):

| Policy | wrist range p5–p95 [rad] | mean \|q̇\| [rad/s] | p95 \|q̇\| [rad/s] | mean q̇² |
|---|---|---|---|---|
| `teacher_onedir_droppen_ent5e-3` model_5999 (742zn06f) | **1.24** | **2.19** | 4.45 | 6.37 |
| same, model_4500 | 1.11 | 1.89 | 4.11 | |
| `abl_B_onedir_droppen_ent5e-3` (80hm76fu) | 1.34 | 1.97 | 3.58 | |
| `student_pt_rotlin` (p3dxn5pm) | 0.48 | 0.23 | 0.78 | |
| `teacher_rotlin_ent5e-3` | 0.36 | 0.15 | 0.40 | 0.06 |
| `abl_B_rotlin_cont10k` | 0.40 | 0.47 | 1.13 | |

Every onedir policy swings the wrist ~10× faster than the rotlin ones: part of the extra onedir speed comes from rocking
the whole hand, which may not be reproducible on the real hand (the PT paper fixes the wrist at 0 on hardware) and may
explain why onedir keeps only 22 % of its speed in MuJoCo. Ablation: L2 penalty on the wrist joint velocity,
`add_wrist_vel_penalty` (−4·q̇², ~25 per second at the onedir motion vs ~95 per second of rotation reward, ~0.2 at
rotlin-like motion): teacher `teacher_onedir_droppen_wristpen_ent5e-3` (the requested ablation of 742zn06f; student task
`-student-pt-onedir-droppen-wristpen-v0` registered) and the deployable PPO version `abl_B_onedir_droppen_wristpen_ent5e-3`.
Stronger alternative if the penalty is not enough: the wrist held fixed (`-wristfix-v0` exists for the old reward).
- **3× drop penalty (`abl_B_onedir_droppen6k_ent5e-3_r6k`, pod, W&B 7ogu7mc5)**, nominal / perturbed: 1000 12.61 /
  12.49 RPM, 88 / 34 % · 2000 12.69 / 12.83, 72 / 17 % · 3000 14.30 / 13.37, 61 / 17 % · 4000 14.76 / 14.40, 47 / 14 % ·
  4999 14.38 / 14.12, 53 / 17 %, DC 1.14 / 3.94. Mean of the last three: 14.5 RPM / 54 % / DC 0.86 vs 15.7 / 61 % / 0.77
  for the −2000 baseline (seed 0): slower and not fewer drops. **Decision: keep −2000.** Its drop-free rate falls as
  training goes on (88 → 53 %) while the action-noise std climbs past 4 (pod log), the same pattern as the baseline
  (73 → 61–63 %): another pointer at the entropy-5e-3 noise (see [PPO / capacity](ppo_and_capacity.md#action-noise-under-the-entropy-bonus)).
  Queued 2026-10-30 15:30: the fixed-wrist version `abl_B_onedir_droppen_wristfix_ent5e-3` (`-asym-onedir-droppen-wristfix-v0`).
- **Baseline seed 1 (`abl_B_onedir_droppen_ent5e-3_seed1`, job 29852087)**, nominal / perturbed: 1000 12.77 / 13.01 RPM,
  67 / 27 % · 2000 13.17 / 14.06, 42 / 22 % · 3000 15.66 / 14.35, 50 / 20 % · 4000 13.88 / 13.17, 48 / 11 % · 4999 14.89 /
  14.40, 41 / 14 %. Last-3 mean 14.8 RPM / 46 % / DC 1.20 (seed 0: 15.7 / 61 % / 0.77). **Two-seed baseline: 15.3 RPM,
  54 % drop-free (seed spread ±8 points), DC 0.99, perturbed 18 %.** Seed 0's 16.4 RPM / 63 % was the lucky end. Against
  this: the 3× penalty (14.5 / 54 %) is inside the spread, and the onedir teacher (15.5 / 74 %) is ~20 points more
  drop-free. Seed 1's drop-free rate also falls during training (67 → 41 %) as the noise std rises, the third run with
  that pattern. Seed 2 (A10) is running.
- **Baseline seed 0 trained to 10k (`abl_B_onedir_droppen_ent5e-3_cont10k`, job 29852088)**, nominal / perturbed: 6000
  16.47 / 13.91 RPM, 59 / 14 % · 7000 16.52 / 14.12, 48 / 11 % · 8000 **17.05** / 13.30, 53 / 13 % · 9000 16.73 / 13.69,
  53 / 9 % · 9998 16.88 / 14.22, 59 / 14 %, DC 0.70 / 3.41. Last-3 mean 16.9 RPM / 55 % / DC 0.72 vs 15.7 / 61 % / 0.77
  for the same seed at 5k: +1.2 RPM (the fastest policy so far, 17.0 RPM at 8000), the same drop-free rate within seed
  noise, and more fragile under perturbation (12 vs 21 % drop-free). Unlike rotlin (10k: +0.4 RPM, +6 points
  drop-free) the onedir line gets faster, not safer, with more training. The wrist motion is unchanged (range 1.38 vs
  1.34 rad, mean |q̇| 2.06 vs 1.97 rad/s), so the extra speed is not more wrist swing.
- **Cap 4 rad/s (`abl_B_onedir_droppen_clip4_ent5e-3`, job 29852091)**, nominal / perturbed: 1000 14.59 / 14.75 RPM,
  59 / 19 % · 2000 13.56 / 14.54, 39 / 17 % · 3000 15.94 / 14.50, 55 / 8 % · 4000 15.97 / 14.98, 52 / 14 % · 4999 16.18 /
  15.47, 45 / 13 %. Last-3 mean 16.0 RPM / 50 % / DC 1.16 / perturbed 12 % vs the two-seed cap-3 baseline 15.3 / 54 % /
  0.99 / 18 %: inside the seed spread on speed and drops, a little worse perturbed. The steady-state cube speed (10–60 s
  after spawn) is 1.73 rad/s, as for cap 3 (1.56–1.75). Cap 2 → 3 moved it 1.06 → 1.72 rad/s; 3 → 4 adds nothing, so the
  speed is now limited by the hand / sim dynamics, not the reward. **Decision: keep cap 3.** The drop timing also shows
  the seed spread: steady-state drops 0.30 (seed 0) vs 0.90 (seed 1) per minute.
- **Cube smoothness ×10 (`abl_B_onedir_droppen_acc10_ent5e-3_r6k`, pod, W&B mau1bbmt)**, nominal / perturbed: 1000
  13.01 / 12.97 RPM, 55 / 13 % · 2000 14.20 / 14.02, 56 / 23 % · 3000 15.22 / 14.00, 69 / 20 % · 4000 14.70 / 14.08, 69 / 25 %
  · 4999 14.97 / 14.36, **72 / 33 %**, DC 0.53 / 1.86. Last-3 mean 15.0 RPM / **70 %** / DC 0.58, perturbed 26 % / DC 2.0
  vs the two-seed baseline 15.3 / 54 % / 0.99, perturbed 18 % / ~3.8. Same speed, +16 points drop-free (twice the ±8
  seed spread), perturbed drops halved. Its drop-free rate rises over training (55 → 72 %) while every baseline seed's
  falls, at a similar action-noise std (3.5): the smoothness term counteracts whatever drives the decline. **First clear
  win among the +z single-knob tests.** Follow-ups: seed 1 and weight 30 (dose).
- **Wrist motion and drop timing vs entropy (2026-10-30 18:45, model_4999 nominal rollouts; pod npz via remote-results):**

  | Policy | wrist range p5–p95 [rad] | mean \|q̇\| [rad/s] | drops (64 × 60 s) | drops / env-min at 0–1 / 1–2 / 2–3 / 3–5 / 5–10 / 10–20 / 20–60 s |
  |---|---|---|---|---|
  | `abl_B_onedir_droppen_ent1e-4_r6k` (default entropy) | **0.64** | **1.10** | **19** | 2.96 / 0.77 / 0.00 / 0.78 / 0.16 / 0.43 / **0.15** |
  | `abl_B_onedir_droppen_acc10_ent5e-3_r6k` | 1.11 | 1.90 | 34 | 3.11 / 1.31 / 0.00 / 1.00 / 0.71 / 0.56 / 0.33 |
  | `abl_B_onedir_droppen_ent5e-3` (seed 0) | 1.34 | 1.97 | 41 | 2.94 / 3.76 / 1.96 / 1.01 / 1.16 / 0.42 / 0.30 |
  | `abl_B_onedir_droppen_ent5e-3_seed1` | 1.33 | 2.04 | 71 | 4.16 / 2.46 / 1.02 / 1.07 / 1.03 / 0.99 / 0.90 |

  The default-entropy policy is faster (17.5 vs 15.3 RPM) with half the wrist swing and almost no spin-up drops
  (1–3 s after spawn); the remaining first-second drops (~3 per env-minute) are the spawn artifact every policy shows.
  So the large wrist swing seen in 742zn06f (entropy 5e-3) and most spin-up drops came largely from training under
  std 3–4 action noise, not from the onedir reward itself. Smoothness ×10 removed part of the spin-up drops on
  entropy 5e-3; on default entropy it trades ~2 RPM for +25 points perturbed drop-free ([below](#smoothness-on-default-entropy-2026-10-31)).

## Smoothness on default entropy (2026-10-31)

`abl_B_onedir_droppen_acc10_ent1e-4_r6k` (pod, W&B ps0bke4n): the cube smoothness term ×10 on the default-entropy +z
config. Isaac, +z, 64 envs × 60 s, nominal / perturbed (6–12 substep delay, ±0.05 rad offset):

| | 1000 | 2000 | 3000 | 4000 | 4999 | last-3 mean |
|---|---|---|---|---|---|---|
| RPM | 14.2 / 13.3 | 13.9 / 14.5 | 15.1 / 15.0 | 15.5 / 14.7 | 16.5 / 15.3 | 15.7 / 15.0 |
| DFSR % | 86 / 34 | 80 / 44 | 75 / 50 | 83 / 53 | 88 / 55 | 82 / **53** |

Against the same config without it (qn0pkisu, last-3 mean 17.5 RPM, 76 / 28 % drop-free) it gives up ~1.8 RPM for
+6 points nominal and **+25 points perturbed** drop-free (seed 1 of that config, cule0dej: 15.4 RPM, 71 / 27 %, so the seed spread is ~2 RPM and ~5 points): like the work penalty and the wrist penalty, a smoother
cube motion makes the policy much less sensitive to action delay and encoder offset. It does not fix the tumbling
(cube_motion at 4999: tilt 40° mean / 48° p90, 200 top-face changes / min, wrist 0.54 rad / 0.90 rad/s). Next:
`abl_B_onedir_droppen_upright_acc10_ent1e-4` (DTU L40S, 4000 it) adds it to the tilt penalty.

## Cube tumbling (2026-10-30)

The user noticed in the default-entropy +z video (qn0pkisu) that the cube's rotation axis does not stay vertical, and
that the wrist still moves too much. `scripts/eval/cube_motion.py` (nominal rollout of the last checkpoint):

| Policy | spin about z [rad/s] | off-axis spin [rad/s] | cube tilt mean / p90 [deg] | top-face changes / min | wrist range [rad] / mean \|q̇\| [rad/s] |
|---|---|---|---|---|---|
| +z default entropy (qn0pkisu) | 1.85 | **2.54** | **36.0 / 46.2** | **173** | 0.64 / 1.10 |
| +z smoothness ×10, entropy 5e-3 | 1.57 | 2.85 | 33.3 / 44.8 | 157 | 1.11 / 1.90 |
| +z baseline, entropy 5e-3 | 1.72 | 2.91 | 25.0 / 41.9 | 106 | 1.34 / 1.97 |
| +z teacher 742zn06f | 1.59 | 3.69 | 31.8 / 45.2 | 171 | 1.24 / 2.19 |
| rotlin student p3dxn5pm | 1.14 | 1.49 | **6.7 / 11.5** | **3.8** | 0.48 / 0.23 |
| rotlin teacher | 1.24 | 1.43 | 6.3 / 10.6 | 2.6 | 0.36 / 0.15 |
| +z tilt penalty (5oklh054, default entropy) | 1.18 | 1.52 | **7.4 / 14.7** | **2.7** | **0.28 / 0.09** |

The rotlin policies turn the cube flat (one face up, 6° tilt; their off-axis spin is rocking). Every +z policy tumbles
it: 25–36° mean tilt, the top face changes ~2–3 times per second, more spin about the horizontal axes than about z. RPM
only integrates the z-component, so this counted as speed; the 3 rad/s cap made it pay (rotlin's cap at the
commanded 1.5 rad/s did not). For the demo ("z-rotation") the +z speed numbers are therefore not comparable to rotlin's.

Fix: `rewards_ablation.cube_tilt` = 1 − max_i |(R e_i)·z|, zero while any face points up (so a clean z-rotation at any
yaw is free, unlike the off-axis angular-velocity penalty, which at −15 stopped rotlin rotating), weight −200
(`add_cube_tilt_penalty`: ~38 per second at 36° vs ~110 per second of rotation reward). Unit test: yaw only 0, 30° 0.134,
90° (another face up) 0, 45° 0.293. 2×2 on default entropy: baseline (qn0pkisu, done) / fixed wrist (queued) /
`abl_B_onedir_droppen_upright_ent1e-4` / `abl_B_onedir_droppen_upright_wristfix_ent1e-4`. The plain default-entropy
teacher was dropped; the next teacher uses the winning cell. From now on every +z result is also judged on cube tilt,
face changes and wrist motion (`cube_motion.py`), not only RPM.
- **Spin-up ramp (`abl_B_onedir_droppen_ramp_ent5e-3_r6k`, pod, W&B ty9c9smm)**, entropy 5e-3, nominal / perturbed:
  1000 12.26 / 12.38 RPM, 69 / 33 % · 2000 15.16 / 15.94, 64 / 33 % · 3000 16.02 / 15.79, 67 / 25 % · 4000 16.21 / 15.03,
  75 / 20 % · 4999 15.92 / 14.98, 66 / 27 %. Last-3 mean 16.1 RPM / 69 % / DC 0.42 / perturbed 24 % vs the entropy-5e-3
  baseline 15.3 / 54 % / 0.99 / 18 %: +15 points drop-free, like smoothness ×10 (70 %). Cube motion (model_4999): tilt
  31° mean, 129 face changes / min, wrist 1.00 rad / 1.64 rad/s: still tumbling. Not ported to default entropy for now:
  default entropy already removed most spin-up drops, and the tumbling fix (tilt penalty) comes first.
- **Cube position ×3 (`abl_B_onedir_droppen_pos45_ent5e-3`, job 29852803)**, entropy 5e-3, nominal / perturbed: 1000
  10.41 / 10.37 RPM, 72 / 38 % · 2000 12.32 / 12.07, 80 / 41 % · 3000 13.06 / 13.26, **91** / 48 % · 4000 12.73 / 14.01, 86 / 48 %
  · 4999 12.59 / 13.77, 86 / 50 %, DC 0.14 / 0.97. Last-3 mean **12.8 RPM, 88 % drop-free, DC 0.15, perturbed 49 % / DC
  1.07** vs the entropy-5e-3 baseline 15.3 / 54 % / 0.99 / 18 %. Cube motion (`cube_motion.py`, model_4999): tilt 16° mean /
  25° p90, **3.7 face changes / min** (baseline 106), wrist 0.42 rad / 0.75 rad/s (baseline 1.34 / 1.97). Keeping the cube
  at the palm centre (σ 2 cm, weight 45) removes the tumbling and most of the wrist swing on its own, and is by far the
  most robust +z policy under the perturbed eval (2.7× the baseline's drop-free rate), at −2.5 RPM. Against the clean
  bidirectional rotlin student (10.9 RPM, 86–91 %, perturbed 56–60 %, 7° tilt): faster at a similar drop-free rate.
  Follow-ups on default entropy: `abl_B_onedir_droppen_pos45_ent1e-4` and with the tilt penalty
  `abl_B_onedir_droppen_upright_pos45_ent1e-4` (L40S top of queue + pod first).
- **Work penalty on +z (`abl_B_onedir_droppen_work015_ent5e-3`, job 29853308)**, entropy 5e-3, nominal / perturbed: 1000
  13.65 / 13.11 RPM, 67 / 31 % · 2000 13.65 / 13.93, 64 / 36 % · 3000 14.78 / 14.47, 61 / 33 % · 4000 15.12 / 14.85, 64 / 59 % ·
  4999 14.35 / 13.77, 69 / 44 %. Last-3 mean 14.8 RPM / 65 % / DC 0.51, **perturbed 45 % / DC 1.17** vs the baseline
  15.3 / 54 % / 0.99 / 18 % / ~3.8: the perturbed drop-free rate ×2.5 (as on rotlin: 56 vs 39 %) at nearly the baseline
  speed. Cube motion: tilt 30°, 131 face changes / min, wrist 0.88 rad / 1.55 rad/s: still tumbling. Kept as a robustness
  add-on for the winning non-tumbling config (position ×3 / tilt penalty), not run alone again.
- **Wrist-penalty teacher (`teacher_onedir_droppen_wristpen_ent5e-3_r6k`, pod, W&B cbgi1o9d)**, entropy 5e-3, privileged,
  nominal / perturbed: 1000 13.84 / 11.61 RPM, 86 / 41 % · 2000 14.68 / 12.94, 80 / 52 % · 3500 14.87 / 13.34, 81 / 50 % ·
  4500 13.85 / 13.14, 84 / 64 % · 5999 14.18 / 12.50, **89** / 59 %, DC 0.11 / 0.59. Last-3 mean 14.3 RPM / **85 %** /
  DC 0.17, **perturbed 58 % / DC 0.60**, vs the same teacher without the penalty (742zn06f) 15.5 / 74 % / 0.34,
  perturbed 17 % / 2.9: −1.2 RPM, +11 points nominal and ×3.4 perturbed drop-free. A calm wrist makes the gait far less
  sensitive to action delay and encoder offset. Cube motion (model_5999): tilt 20° mean / 35° p90, **16 face changes /
  min** (742zn06f: 32° / 171), wrist 0.43 rad / 0.51 rad/s: the wrist penalty also mostly stops the tumbling, so the
  wrist swing was a main driver of it.
- **Tilt penalty on default entropy (`abl_B_onedir_droppen_upright_ent1e-4_r6k`, pod, W&B 5oklh054)**, nominal /
  perturbed: 1000 12.24 / 11.21 RPM, 83 / 58 % · 2000 11.90 / 11.20, 91 / 53 % · 3000 11.67 / 11.06, **95** / 53 % · 4000
  11.21 / 11.26, 94 / **63** % · 4999 11.24 / 10.54, 91 / 47 %, DC 0.09 / 0.80. Last-3 mean **11.4 RPM, 93 % drop-free,
  DC 0.07, perturbed 54 % / DC 0.83** vs the same config without it (qn0pkisu) 17.5 / 76 % / 0.30 / 28 %. Cube motion
  (model_4999): tilt **7.4° mean / 14.7° p90, 2.7 face changes / min** (qn0pkisu 36° / 173), wrist **0.28 rad /
  0.09 rad/s** (0.64 / 1.10): a flat z-rotation like the rotlin policies, the calmest wrist of any policy so far, and the
  most reliable +z policy; it gives up 6 RPM, i.e. most of the "speed" of the tumbling policies was off-axis tumbling.
  Pod decision rule → GOOD: teacher `teacher_onedir_droppen_upright_pos45` (6k) and the ADR fine-tune of this policy
  (`upright_adr`, 2k) started / queued on the pod. Dose: upright500 (stronger) and upright100 (weaker) queued on DTU.
  model_4000 (best nominal + perturbed pair, 94 / 63 %) exported bit-exact: `<run>/exported/policy_history.pt`
  (params from the DTU seed-1 run of the same config, `params/SOURCE.txt`).
- **Tilt penalty, seed 1 (`abl_B_onedir_droppen_upright_ent1e-4_seed1`, DTU A10, W&B 0p3fbvkd)**, nominal / perturbed:
  1000 6.76 / 5.86 RPM, 64 / 39 % · 2000 9.26 / 5.71, 73 / 33 % · 3000 9.44 / 5.76, 75 / 41 % · 4000 9.67 / 5.44, 61 / 33 % ·
  4999 9.63 / 5.39, 88 / 42 %. Last-3 mean **9.6 RPM, 74 % drop-free, perturbed 5.5 RPM / 39 %** vs seed 0 11.4 / 93 % /
  11.0 / 54 %. Cube motion (model_4999): tilt 7.8° / 15.2° p90, 3.2 face changes / min, wrist 0.31 rad / 0.07 rad/s. The
  flat rotation and the calm wrist replicate; speed and drop-free rate do not (its rotation reward plateaued at ~55 vs
  ~78 for seed 0): the tilt penalty makes the problem harder to optimize and seed-dependent, so seed 0 (5oklh054) is
  a good draw, not the expected result. Replicate of the combined config queued (`upright_pos45` seed 1, A10).
- **Position ×3 and tilt + position ×3 on default entropy (pod, 4000 it each; W&B kfuq4jwb, b0r506gz)**, nominal / perturbed:
  - position ×3 (`abl_B_onedir_droppen_pos45_ent1e-4_r6k`): 1500 14.03 / 13.75 RPM, 81 / 30 % · 2000 14.68 / 14.00, 84 / 52 % ·
    3000 15.52 / 14.40, 92 / 48 % · 3999 15.51 / 14.44, 92 / 42 %. Last-3 mean **15.2 RPM, 90 % drop-free, perturbed 47 %**
    (entropy 5e-3: 12.8 / 88 / 49). Cube motion (3999): tilt 16.8° / 28.8° p90, 3.8 face changes / min, wrist 0.46 rad /
    0.43 rad/s. The fastest policy that does not tumble (the tumbling baseline qn0pkisu: 17.5 RPM at 36° tilt).
  - tilt + position ×3 (`abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k`): 1500 13.64 / 12.22, 86 / 47 % · 2000 13.77 /
    12.45, 89 / 58 % · 3000 13.87 / 11.60, 94 / 45 % · 3999 14.54 / 11.53, 92 / 45 %. Last-3 mean **14.1 RPM, 92 %
    drop-free, perturbed 50 %**. Cube motion: tilt **10.4° / 17.7°**, 2.3 face changes / min, wrist 0.31 rad / 0.29 rad/s.
  Adding the position term to the tilt penalty buys +2.7 RPM (11.4 → 14.1) at the same drop-free rate and a still-flat
  cube (10° vs 7°): the centred grasp lets the hand turn faster without tilting. Its ADR fine-tune
  (`upright_pos45_adr`, pod) and the one of position ×3 (`pos45_ent1e-4_adr`, DTU) follow; their seed spread is open
  (`upright_pos45` seed 1 queued on A10).
- **Fixed wrist on default entropy (`abl_B_onedir_droppen_wristfix_ent1e-4`, DTU A40, W&B z49id6xq)**, nominal / perturbed:
  1000 14.25 / 13.99 RPM, 77 / 47 % · 2000 15.43 / 13.73, 78 / 66 % · 3000 15.59 / 12.62, 77 / 45 % · 4000 16.03 / 12.65,
  78 / 59 % · 4999 16.37 / 12.78, 70 / 47 %. Last-3 mean 16.0 RPM, 75 % drop-free, perturbed **51 %** (baseline qn0pkisu
  17.5 / 76 / 28). Cube motion (4999): **tilt 39.5° / 49.7°, 156 face changes / min** (baseline 36° / 173), wrist (passive)
  0.22 rad / 0.10 rad/s. Without a wrist the fingers tumble the cube just as much: on default entropy the wrist swing
  is not what causes the tumbling (on the entropy-5e-3 teacher the wrist penalty did reduce it, 171 → 16 / min), the
  tumbling is a finger gait the speed reward pays for, and the tilt penalty is the fix. A calm or fixed wrist does make
  the policy more robust to delay / offset (28 → 51 %), like the wrist penalty on the teacher (17 → 58 %).
- **Tilt + wrist penalty (`abl_B_onedir_droppen_upright_wristpen_ent1e-4`, DTU A100, W&B 1xxwpij9)**, nominal / perturbed:
  1000 10.55 / 9.63 RPM, 73 / 50 % · 2000 11.85 / 11.39, 78 / 47 % · 3000 13.06 / 11.95, 84 / 66 % · 4000 13.41 / 12.42,
  92 / 61 % · 4999 13.40 / 11.84, 91 / 63 %. Last-3 mean **13.3 RPM, 89 % drop-free, perturbed 63 %** (tilt alone: seed 0
  11.4 / 93 / 54, seed 1 9.6 / 74 / 39). Cube motion (4999): tilt **7.0° / 13.1°**, 2.3 face changes / min, wrist 0.33 rad /
  0.10 rad/s. Flat like the tilt penalty alone, faster than both tilt seeds and the most robust non-ADR policy so far:
  the wrist penalty adds robustness on top of the tilt penalty, as on the teacher (it is one seed; the tilt penalty's
  seed spread is ~2 RPM / ~20 points). ADR fine-tune queued (new task `-asym-onedir-droppen-upright-wristpen-adr-v0`).
- **Tilt penalty −500 (`abl_B_onedir_droppen_upright500_ent1e-4`, DTU L40S, W&B q96jruid)**, nominal / perturbed: 1000 9.48 /
  9.80 RPM, 88 / 50 % · 2000 11.73 / 11.02, 89 / 47 % · 3000 12.22 / 10.50, 80 / 42 % · 4000 12.05 / 10.23, 81 / 42 % · 4999
  12.17 / 10.14, 84 / 45 %. Last-3 mean 12.2 RPM, 82 % drop-free, perturbed 43 %; tilt 6.0° / 11.9°, 2.4 face changes / min,
  wrist 0.25 rad / 0.10 rad/s. Slightly flatter than −200 (7.4–7.8°); speed and drop-free rate inside the −200 seed spread
  (11.4 / 93 / 54 and 9.6 / 74 / 39). The weight is not critical between −200 and −500; −100 (queued) tests the low side.
- **Tilt penalty + smoothness ×10 (`abl_B_onedir_droppen_upright_acc10_ent1e-4_r6k`, pod, W&B x7i5siys)**, nominal / perturbed:
  1500 9.46 / 7.41 RPM, 75 / 25 % · 2000 10.84 / 9.32, 64 / 31 % · 3000 10.84 / 9.71, 72 / 27 % · 3999 10.96 / 10.54,
  67 / 31 %. Last-3 mean 10.9 RPM, 68 % drop-free, perturbed 30 %; cube motion (3999) tilt 9.1° / 17.0°, 6.0 face changes /
  min, wrist 0.32 rad / 0.19 rad/s. Worse than the tilt penalty alone on both seeds (11.4 / 93 / 54, 9.6 / 74 / 39):
  on the tumbling baseline smoothness ×10 doubled the perturbed drop-free rate, on top of the tilt penalty it hurts
  (two cube-motion terms competing with the speed reward?). Not combined further (`upright_pos45_acc10` cancelled).
