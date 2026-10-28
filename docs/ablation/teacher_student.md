# Teacher → student distillation

[Index](README.md) · metrics: [eval protocol](eval_protocol.md) · PPO fine-tuning of the student: [student_finetune.md](student_finetune.md)

## Question / hypothesis

The deployable gain is teacher → history student, not the critic: B/C only improve training of a policy that still
sees one noisy obs. Train a privileged teacher (teacher obs as *actor* input) and distill it into a student on joint
history (MLP-h10, LSTM, transformer + reconstruction loss), as in the PT paper. Originally (§3 of the old notes)
teacher–student via `DistillationRunner` was planned "only if the asymmetric critic and history stall"; the PT paper
takeaways moved it up.

## Literature

- ORCA PT paper 2605.21330: full takeaways and tables in [literature](literature.md#orca-proprioceptive-transformer-paper-takeaways).
  Student ablation with motor encoders only and a 10-step history (our case): MLP 5.08 RPM / 61.5 % DFSR / 2.0 drops;
  LSTM 7.00 / 100 % / 0; Transformer 9.33 / 98.1 % / 0.33; Transformer **without reconstruction loss** 4.75 / 35.0 % / 5.0.
- Reconstruction loss: L = L_BC + 0.5·(cube position) + 0.3·(clean joint pos) + 0.2·(clean joint vel).
- Student details (PT): per-step obs (noisy joint pos/vel, prev action, prev joint command, goal command) linearly
  projected to 256-dim tokens (10 history + 1 action + 1 command), sinusoidal pos. embedding, 3 layers, 4 heads, FF 1024,
  [CLS] queries → action + reconstruction heads. 1200 epochs (6000 steps), batch 256, lr 1e-5, wd 0.01. Distillation is
  L2 to teacher actions (DAgger vs fixed dataset not stated).
- PT code: **not found** (arXiv lists none; an author's post mentions code + data, link unverified).
- Dactyl 1808.00177 (LSTM 13 vs FF 3.5), LEAP GRU → student-lstm. SERNF 2602.09580 (ORCA, 45 mm cube): distilled sim
  student "fragile – most trajectories fail before a full rotation"; needed 105 min real online RL → 6.25 RPM.

## Plan (stages)

| Stage | What | Code | Status (old notes) |
|---|---|---|---|
| 1 | Privileged teacher: PPO, actor + critic read the noise-free 81-dim `teacher` group; env also exposes the noisy 74-dim `policy` group for the student | task `orca-v2-inhand-repose-teacher-v0` (`OrcaV2ReposeTeacherEnvCfg`, `rsl_rl_teacher_student_cfg.py:PPORunnerCfgOrcaV2Teacher`) | queued (10k iters) → done 2026-10-28 12:15 |
| 2a | MLP student on a 10-step history of the noisy policy obs (Isaac Lab `history_length=10`) | task `-student-mlp-h10-v0` (`DistillationRunner` + `StudentTeacher`, DAgger-style: student acts, teacher labels) | written, needs teacher |
| 2b | **LSTM student** (paper: 7.0 RPM, 100 % DFSR with motor encoders) | task `-student-lstm-v0` (`StudentTeacherRecurrent`) | written, needs teacher |
| 2c | Per-episode joint position bias in the student's obs noise | custom noise term (~30 lines) | TODO |
| 3 | Transformer student + reconstruction loss (paper's best, 9.3 RPM) | task `-student-pt-v0`: `transformer_student.py` (CLS token over 10 per-step tokens, 3 layers / 4 heads / d 256 / FF 1024; recon targets = first 37 dims of the teacher obs) | written, CPU unit-tested, needs teacher |
| 4 | Export | TorchScript wrapper that holds the history buffer / LSTM state inside `policy.pt` → still one 74-dim obs per call, **deploy node unchanged** | TODO |

Notes:
- The teacher is trained under full DR (sim only) so its labels cover the randomized dynamics the student must handle.
- Distillation loads the teacher checkpoint via `load_run` / `load_checkpoint` (train.py:170); `teacher_obs_normalization=True`
  so the PPO actor normalizer is reused. Submit with `EXTRA="--load_run <teacher run dir name> --checkpoint model_9999.pt"`
  ([infrastructure](infrastructure.md#submitting-runs)).
- History must be initialized the same way at reset in sim (Isaac Lab fills the buffer with the first obs) and on deploy.

Related ablation ideas from the paper:

| Idea | Why | Effort |
|---|---|---|
| Student MLP vs LSTM vs Transformer | MLP 5.1 / LSTM 7.0 / Transformer 9.3 RPM (motor encoders) | 2a/2b config, 3 code |
| Reconstruction loss on/off, weights (0.5 cube pos, 0.3 joint pos, 0.2 joint vel) | without it the transformer is the worst student (35 % DFSR) | with stage 3 |

## Runs

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Drop % |
|---|---|---|---|---|---|---|---|
| `teacher` | `-teacher-v0` | privileged teacher: `TeacherObsGroupCfg` as **actor + critic** input (sim only, not deployable) | 29830926 / L40S | 10k | done (2026-10-28 12:15) | ~1736 | ~6 % |
| `teacher_cont18k` | `-teacher-v0` | teacher resumed from model_9999 for 8k more iters (still improving at 10k) | 29840146 / L40S (+ A100 dup 29840147) | +8k | queued | | |
| `student_mlp_h10` | `-student-mlp-h10-v0` | distill teacher → MLP on 10-step noisy proprio history (PT paper's MLP student) | 29839014 / A100 | 2k | queued after teacher | | |
| `student_lstm` | `-student-lstm-v0` | distill teacher → LSTM student on noisy policy obs (stage 2b) | 29839015 / A40 | 2k | queued after teacher | | |
| `student_pt` | `-student-pt-v0` | distill teacher → transformer on 10-step motor-proprio history + reconstruction loss (stage 3; old plan: 6k steps) | 29839013 / L40S | 2k | queued after teacher | | |
| (TBD) | | + per-episode joint position bias in obs noise (stage 2c) | | 5k | TBD | | |

Students distill from the teacher's `model_9999`.

## Results

### Log

- **2026-10-28 12:20 — privileged teacher eval** (actor sees noise-free joints + cube pose; not deployable),
  nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 2000 | 4.00 / 4.99 | 46.7 / 47.0 | 89 / 69 | 0.16 / 0.89 |
  | 4000 | 3.81 / 4.59 | 45.1 / 47.7 | 84 / 61 | 0.22 / 1.58 |
  | 6000 | 4.40 / 4.60 | 50.4 / 46.9 | 83 / 53 | 0.19 / 1.66 |
  | 8000 | 6.09 / 6.58 | 72.5 / 68.8 | 92 / 80 | 0.09 / 0.22 |
  | 9999 | **7.44 / 7.63** | **81.4 / 74.4** | **94 / 73** | **0.06 / 0.31** |

  Best policy so far on every metric (vs RMA s1 6.48, C 6.08, B 5.45) and the most robust under perturbation
  (DC 0.31 vs 0.6–1.0). It is **still improving steeply at 10k** (6.1 → 7.4 RPM over the last 2k iters) — slower
  start than B (4.0 vs 4.3 RPM at 2k), but no plateau. Students distill from model_9999; a longer teacher (15–20k,
  resumed from 9999) is a cheap upgrade if the students track it (→ `teacher_cont18k`). Ceiling for the students: ~7.4 RPM.

## Conclusions so far

- The privileged teacher (7.44 RPM, 94 % DFSR) is the best policy so far and still improving at 10k.
- Student results pending.

## Next steps / open questions

- Student evals (MLP-h10 / LSTM / PT); do they track the teacher?
- Re-distill from `teacher_cont18k` if it improves further.
- Stage 2c (per-episode joint bias) and stage 4 (export with history inside `policy.pt`) are TODO.
- PPO fine-tuning of the best distilled student: planned, see [student_finetune.md](student_finetune.md) and the
  background below.

## Student PPO fine-tuning: background (from the old notes, §15)

Planned run "student fine-tune": PPO fine-tuning of the best distilled student, to close the gap to the teacher; known
to be unstable at the start → check existing code / stabilizers first (critic warm-up, KL-to-teacher anneal).

**Existing code and stabilizers** (subagent review 2026-10-28; [code] = checked in rsl_rl 3.1.2):
- No off-the-shelf "distill → PPO fine-tune" in rsl_rl / Isaac Lab [code]. Closest public code: TGRL (Improbable-AI/tgrl,
  RL + adaptive teacher-KL, teacher with privileged obs), DPPO (irom-princeton/dppo, critic warm-up, clamped std, low actor
  LR), residual PPO on a frozen policy (ResiP, Policy Decorator). Hora, Visual Dexterity, Extreme Parkour, DeXtreme stop
  at distillation. Wołczyk et al. (ICML 2024): early fine-tuning *forgets* skills in states not yet revisited; BC/KL
  regularization to the teacher (kickstarting) was the most reliable fix.
- Collapse causes specific to rsl_rl [code]: the distillation loss uses `act_inference` (mean only), so the student's std
  is never trained and PPO starts at `init_noise_std`; the adaptive LR schedule multiplies the LR by 1.5 whenever
  KL < desired_kl / 2, up to 1e-2; the critic starts random, and advantage normalization amplifies its noise.
- Recipe if we do it: actor ← student weights (`student.*` → `actor.*`, normalizer too), critic ← the teacher's trained
  privileged critic; actor frozen for 100–200 iters while the critic adapts; std = teacher's final std, entropy 0;
  LR 1e-4 fixed (or adaptive with desired_kl 0.003–0.005 and a clamped max), clip 0.1; + β·‖μ_student − μ_teacher‖²
  annealed 1 → 0 over ~1k iters; log RPM of the deterministic policy, MSE to teacher, explained variance; roll back if RPM
  drops > 20 %. Fallback: residual PPO on the frozen student.
