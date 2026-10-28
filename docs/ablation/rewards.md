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
| `abl_B_rotlin` | `-asym-rotlin-v0` | B, rotation reward linear in speed, clipped at the command (HORA/LEAP/DLR 2024) + off-axis penalty | 29838450 / L40S | 5k | queued |
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

## Conclusions so far

- **`rotsoft` is a reward hack** (2026-10-28 17:10): ~0 RPM, RA ~7 %, while holding the cube safely (DFSR 92 %).
  The magnitude term rewards ‖ω‖ — the norm of the full 3-D angular velocity, **any axis and either sign** — and the
  direction term only the cosine. With the magnitude weight (60) above the direction weight (30) and a wide kernel
  (σ 0.75), wobbling / oscillating the cube in place pays more than rotating it about +z / −z. Lesson: a speed term must
  be measured *along the commanded axis and sign* (as `rotlin`'s ω·k is), never as ‖ω‖. The baseline weighting
  (direction 60 > magnitude 30, narrow σ 0.2) only avoided the hack by making speed nearly irrelevant.
- The training reward of a reward variant says nothing about rotation: always judge by eval.

## Next steps / open questions

- `rotsoft` is out; `rotlin` (axis-projected, clipped) is the remaining candidate. The winner of the reward wave (B or `-rotlin`, decided by eval) is the base of the next architecture
  candidates ([action space](action_space.md), [PPO and capacity](ppo_and_capacity.md)).
- Then rotlin + drop penalty, position / velocity variant, HORA work / torque penalties if rotlin is jerky.
- Re-check drop timing on B's rollout.
- Grasp-cache resets (HORA `gen_grasp.py` + `cache/`; LEAP_Hand_Sim grasp cache per cube size): deprioritized, see log.
