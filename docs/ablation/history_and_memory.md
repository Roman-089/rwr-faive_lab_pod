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
| `arch_asym_hist10` | `-asym-hist10-v0` | B + MLP on 10-step obs history | 29831100 / A40 | 5k | queued | | |
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
| asym hist10 | pending | | | |

### Log

- **2026-10-28 06:45** — asym LSTM eval (in the [asym-critic family table](asymmetric_critic.md#log)): **LSTM
  actor-critic ≈ B** nominal; slightly better under perturbation (DC 0.70 vs 0.86) but within seed noise. Memory alone
  does not close the gap at 5k iters (consistent with its lower training reward).
- **2026-10-28 14:30 — asym hist3 eval:** 5.33 / 5.82 RPM, RA 64.0 / 58.1 %, DFSR 83 / 66 %, DC 0.20 / 0.59 — **≈ B
  nominal** (B two-seed mean 5.45 RPM, RA 64 %), somewhat **more robust under perturbation** (DC 0.59 vs B 0.86 / 0.98,
  DFSR 66 vs 52 / 55 %; single seed). A 3-step history does not raise speed; it may help with the unmodeled delay,
  as expected for a policy that can see its recent commands vs joint response. hist10 pending.

## Conclusions so far

- Neither a 3-step history nor an LSTM actor-critic raises speed over B at 5k iters.
- Both look somewhat more robust under perturbation (fewer drops), single seed each.

## Next steps / open questions

- hist10 eval (HORA predicts MLP + long raw history degrades; PT says 10 steps help behind a better encoder).
- History behind a distilled student instead of PPO from scratch: [teacher-student](teacher_student.md), [RMA](rma.md).
- History must be initialized the same way at reset in sim (Isaac Lab fills the buffer with the first obs) and on deploy.
