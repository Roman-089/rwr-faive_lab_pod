# History and memory (hist3, hist10, asym LSTM)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

Can the (deployable) actor infer cube state from contacts if it sees a short proprioceptive history or has memory?
Does history make the policy more robust to the unmodeled delay (it can see its recent commands vs the joint response)?

## Literature

- HORA 2210.04887: MLP on long raw history collapses (OOD rot 141 @3 steps → 36 @30); 3-step joint pos + targets.
- Faive 2308.02453: 5-step history. PT 2605.21330: 10 steps; window 1 → 10 improved RPM by 31 % (averaged over sensor
  setups); window 3 with motor encoders dropped RA to 73.9 %.
- Dactyl 1808.00177: LSTM median 13 vs FF 3.5; LEAP: GRU (+ grasp cache).
- Ranked idea (1) from the [in-hand rotation review](literature.md#in-hand-rotation-literature-review): short history
  3–5 steps for an MLP, longer only behind a conv / transformer encoder.

Original idea table:

| Idea | Change | Notes |
|---|---|---|
| Proprioceptive history | stack last 3–10 obs, or GRU actor (`ActorCriticRecurrent`) | lets the policy infer cube state from contacts |
| History window 1 / 3 / 6 / 10 | config | paper: RPM +31 % from 1 → 10; window 3 with motor encoders dropped RA to 74 % |

## Runs

All on top of B (asymmetric critic). Deploy needs the history (hist3 = 222-dim, hist10 = 740-dim policy obs).
Reward = mean of last 100 iters.

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Drop % |
|---|---|---|---|---|---|---|---|
| `arch_asym_hist3` | `-asym-hist3-v0` | B + MLP on 3-step obs history | 29831099 / A10 | 5k | done (2026-10-28 14:25) | ~1663 | ~14 % |
| `arch_asym_hist10` | `-asym-hist10-v0` | B + MLP on 10-step obs history | 29831100 / A40 | 5k | done (2026-10-28 17:55) | | |
| `arch_asym_lstm` | `-asym-lstm-v0` | B + LSTM actor-critic (PPO) | 29831098 / A100 | 5k | done (2026-10-28 06:40) | ~1437 | ~13 % |

Cancelled (2026-10-27 15:20, before starting): A-based `-hist3` / `-hist10` / `-lstm` (jobs 29830927 / 29830928 /
29830930), re-based on B after the [interim asym-critic results](asymmetric_critic.md#log). The A-based
`-hist3-v0` task is still registered (sim2real wave 2: "3-step obs history (MLP)", 5k).

## Results (nominal / perturbed)

| Run | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| B (two-seed mean RPM 5.45) seed 42 / seed 1 | 5.66 / 6.55 · 5.23 / 5.92 | 66.3 / 62.0 · 62.6 / 61.1 | 86 / 52 · 91 / 55 | 0.20 / 0.98 · 0.11 / 0.86 |
| asym hist3 | 5.33 / 5.82 | 64.0 / 58.1 | 83 / 66 | 0.20 / 0.59 |
| asym LSTM | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
| asym hist10 | 5.01 / 5.72 | 68.8 / 67.6 | 88 / **70** | 0.14 / **0.44** |
| asym hist10 by ckpt 1000 / 2000 / 3000 / 4000 / 4999 (nominal RPM) | 3.74 · 4.64 · 5.04 · 4.98 · 5.01 | 49.8 · 61.9 · 66.8 · 67.8 · 68.8 | | |

### Log

- **2026-10-28 06:45** — asym LSTM eval (in the [asym-critic family table](asymmetric_critic.md#log)): **LSTM
  actor-critic ≈ B** nominal; slightly better under perturbation (DC 0.70 vs 0.86) but within seed noise. Memory alone
  does not close the gap at 5k iters (consistent with its lower training reward).
- **2026-10-28 14:30 — asym hist3 eval:** 5.33 / 5.82 RPM, RA 64.0 / 58.1 %, DFSR 83 / 66 %, DC 0.20 / 0.59 — **≈ B
  nominal** (B two-seed mean 5.45 RPM, RA 64 %), somewhat **more robust under perturbation** (DC 0.59 vs B 0.86 / 0.98,
  DFSR 66 vs 52 / 55 %; single seed). A 3-step history does not raise speed; it may help with the unmodeled delay,
  as expected for a policy that can see its recent commands vs joint response. hist10 pending.

- **2026-10-28 17:55 — asym hist10 eval:** 5.01 / 5.72 RPM, RA 68.8 / 67.6 %, DFSR 88 / 70 %, DC 0.14 / 0.44. Highest
  training reward of the B family (~1603) but no faster than B in eval; the most robust under perturbation so far.

## Conclusions so far

- **History / memory does not raise speed** over B at 5k iters (hist3 5.33, hist10 5.01, LSTM 5.12 vs B 5.45 RPM).
- **It does raise robustness to the unmodeled delay + offset, increasingly with window length:** perturbed DFSR / DC
  B 52–55 % / 0.86–0.98 → hist3 66 % / 0.59 → hist10 70 % / 0.44 (LSTM 59 % / 0.70). Single seed each, but monotonic.
  Consistent with the PT paper (history matters for motor-only sensing) and with HORA's caution only on speed.
- Training reward is not predictive of eval speed here (hist10 has the highest training reward, lowest nominal RPM).
- Implication: for deployment, history is worth its cost (needs the history kept inside the exported policy); for
  speed, look at entropy / reward / action space instead.

### History on the rotlin reward (2026-10-30)

`abl_B_rotlin_hist3` (`-asym-rotlin-hist3-v0`, job 29851782), nominal / perturbed:

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 7.62 / 7.65 | 54.0 / 50.3 | 66 / 38 | 0.47 / 1.31 |
| 2000 | 9.64 / 9.66 | 64.8 / 59.2 | 70 / 33 | 0.45 / 1.61 |
| 3000 | 10.65 / 10.45 | 70.5 / 61.3 | 89 / 28 | 0.11 / 1.50 |
| 4000 | 10.11 / 9.91 | 67.8 / 61.2 | 84 / 45 | 0.19 / 1.11 |
| 4999 | 10.49 / 10.48 | 71.2 / 59.6 | 81 / 34 | 0.27 / 1.30 |
| `abl_B_rotlin` (2 seeds), 4999 | 10.41, 10.25 / 11.07, 11.48 | 74.0, 76.8 / 67.8, 71.4 | 84, 83 / 44, 45 | 0.22, 0.23 / 1.09, 1.08 |

Same nominal speed and drop rate as rotlin, and no perturbed robustness gain (34 vs 44–45 % drop-free, DC 1.30 vs
1.08; within noise or slightly worse). The robustness gain hist3 showed on the old reward (perturbed DFSR 66 vs 52–55 %)
does not carry over to rotlin. **Decision: no 3-step history for the onedir line.** The memory that pays off is the
distilled transformer student (10-step history, perturbed 60 % drop-free), not a history MLP trained with PPO.

## Next steps / open questions

- ~~hist10 eval~~ done. Combine history with the new base (entropy 5e-3) once the replicate confirms it: speed from
  entropy, robustness from history.
- History behind a distilled student instead of PPO from scratch: [teacher-student](teacher_student.md), [RMA](rma.md).
- History must be initialized the same way at reset in sim (Isaac Lab fills the buffer with the first obs) and on deploy.

## hist3 on the calm + servo recipe (2026-11-01)

3-step actor history, staged like the shipped line (06-type base task → real friction → calm recipe → servo DR); 512 envs at the measured friction, history task. Calm: pod seed 42 best m5400 16.24 RPM / good-trial 85.5 % / 91.2 % after 2 s; DTU seed 1 m6997 15.71 / 87.1 / 92.2. Speed cap 1.6 rad/s: −0.4 to −1.1 RPM, good-trial unchanged (drops are not speed-driven). Servo DR (1.5–3×): full-servo good-trial 26 → 69–71 % within 500 it at −1 to −1.5 RPM without servo. Against the memoryless 06 line (rank 1: 12.7 RPM, good-trial 91.2 % / 79.3 % under the full servo model): +2.5–3.5 RPM but 4–9 points fewer good trials without servo and 8–12 under the full servo model. Not shipped. The history buys speed, not reliability; ROA and the LSTM actor (both training, staged) test whether a longer memory with explicit adaptation does better.

