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
| `teacher_cont18k` | `-teacher-v0` | teacher resumed from model_9999 for 8k more iters (still improving at 10k) | 29840146 / L40S (+ A100 dup 29840147) | +8k | cancelled (superseded by `teacher_ent5e-3`) | | |
| `student_mlp_h10` | `-student-mlp-h10-v0` | distill teacher → MLP on 10-step noisy proprio history (PT paper's MLP student) | 29839014 / A100 | 2k | queued after teacher | | |
| `student_lstm` | `-student-lstm-v0` | distill teacher → LSTM student on noisy policy obs (stage 2b) | 29839015 / A40 | 2k | queued after teacher | | |
| `student_pt` | `-student-pt-v0` | distill teacher → transformer on 10-step motor-proprio history + reconstruction loss (stage 3; old plan: 6k steps); **4096 envs** (8192 ran out of memory on the L40S, job 29839013) | 29845112 / L40S | 2k | done (2026-10-29 01:00) | ~1740 | ~10 % |
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

- **2026-10-29 01:00 — transformer student `student_pt`** (deployable: 10-step noisy motor proprio only), nominal /
  perturbed; distilled from `teacher` model_9999 with 4096 envs:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 500 | 5.99 / 7.28 | 69.8 / 71.4 | 89 / 70 | 0.11 / 0.42 |
  | 1000 | 6.80 / 7.23 | 76.2 / 69.7 | 91 / 78 | 0.09 / 0.31 |
  | 1500 | 6.47 / 7.63 | 71.6 / 73.4 | 94 / 66 | 0.08 / 0.47 |
  | 1999 | **6.89 / 7.52** | **75.2 / 71.3** | **95 / 66** | **0.05 / 0.56** |

  Compared at the last checkpoint (nominal / perturbed):

  | Policy | RPM | DFSR % | DC |
  |---|---|---|---|
  | `teacher` (privileged) | 7.44 / 7.63 | 94 / 73 | 0.06 / 0.31 |
  | **`student_pt`** | 6.89 / 7.52 | **95** / 66 | **0.05** / 0.56 |
  | `abl_B_asym_ent5e-3` (best plain MLP) | 7.33 / 7.31 | 88 / 55 | 0.12 / 0.75 |
  | `arch_rma_s2` | 6.20 / 6.71 | 83 / 78 | 0.25 / 0.38 |
  | `arch_asym_hist10` | 5.01 / 5.72 | 88 / 70 | 0.14 / 0.44 |

  The student keeps 93 % of the teacher's speed and matches its drop-free rate (95 %, the best of any deployable
  policy). It is 0.4 RPM slower than ent5e-3 nominally, but drops less and is faster and more robust than it under the
  perturbed eval (7.5 vs 7.3 RPM, DFSR 66 vs 55 %). Drop timing: like the others, most of its drops are reset artifacts.
  RPM plateaus after 1000 iterations while the behavior loss still falls (0.12): the teacher, not the distillation length,
  limits it. Needs the history-in-`policy.pt` export to deploy.

- **2026-10-29 09:05 — teacher with entropy 5e-3 (`teacher_ent5e-3`)**, nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 2000 | 6.54 / 6.39 | 69.9 / 62.2 | 88 / 73 | 0.14 / 0.36 |
  | 4000 | 7.99 / 7.51 | 77.2 / 66.3 | 89 / 63 | 0.13 / 0.69 |
  | 6000 | 7.86 / 7.36 | 78.2 / 69.8 | 95 / 69 | 0.05 / 0.50 |
  | 8000 | **8.16** / 7.51 | 80.5 / 67.1 | 92 / 59 | 0.08 / 0.69 |
  | 9999 | 7.96 / 7.31 | 79.5 / 68.1 | 88 / 72 | 0.13 / 0.44 |

  Final +0.5 RPM over the entropy-1e-4 teacher (7.44; one seed each, within seed noise), but it **learns ~2.5× faster**:
  8 RPM at 4000 iterations, where the old teacher had 3.8 and needed all 10k. So the rotlin teacher is run with entropy
  5e-3 for 6k iterations (`teacher_rotlin_ent5e-3`). `student_pt` is not re-distilled from this teacher: the rotlin
  teacher is the one to distill from.

- **2026-10-29 19:10 — rotlin teacher (`teacher_rotlin_ent5e-3`)**, nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 9.67 / 9.40 | 70.9 / 59.2 | 91 / 38 | 0.16 / 1.61 |
  | 2000 | 9.95 / 10.16 | 73.2 / 67.1 | 81 / 63 | 0.25 / 0.67 |
  | 3500 | 10.70 / 10.42 | 77.9 / 67.1 | 84 / 55 | 0.17 / 0.80 |
  | 4500 | 11.27 / 10.73 | 79.8 / 67.5 | 88 / 58 | 0.14 / 0.67 |
  | 5999 | **11.85** / 10.52 | **83.2** / 67.2 | **95** / 69 | **0.05** / 0.56 |

  Best policy so far on every nominal metric and still improving at 6k: faster than any deployable policy (rotlin
  10.41) and as drop-free as the best student (95 %). The transformer / LSTM / MLP-h10 students distill from
  model_5999 (jobs 29848449 / 29850539 / 29850540).

### Demo-reward teacher (queued 2026-10-30)

`teacher_onedir_droppen_ent5e-3` (`-teacher-onedir-droppen-v0`): the rotlin teacher beat its deployable PPO
counterpart B-rotlin by 1.5 RPM and 11 points drop-free (11.85 / 95 % vs 10.33 / 84 %), and the onedir + drop penalty
B run is our fastest policy but still drops in 37 % of the 60 s episodes (16.41 RPM / 63 %). So the same privileged
teacher on the demo reward (+z only, linear speed reward clipped at 3 rad/s, −2000 drop penalty) should give the best
target to distill. Settings as `teacher_rotlin_ent5e-3` (entropy 5e-3, 6k iterations). The students reuse the
rotlin pipeline: `-student-pt-onedir-droppen-v0` / `-student-mlp-h10-onedir-droppen-v0` (10-step history env) and
`-student-lstm-onedir-droppen-v0`; which ones to run is decided from the rotlin students' Isaac eval.

## Conclusions so far

- The privileged teacher (7.44 RPM, 94 % DFSR) is the best policy so far and still improving at 10k.
- Transformer student = 93 % of the teacher's RPM, best drop-free rate of any deployable policy (95 %); a better teacher (`teacher_ent5e-3`) is the obvious next step.

## Next steps / open questions

- Student evals MLP-h10 / LSTM pending; PT done (tracks the teacher at 93 %).
- Re-distill `student_pt` from `teacher_ent5e-3` if that teacher beats 7.44 RPM.
- **rotlin pipeline** (2026-10-29): rotlin beats every old-reward policy incl. the teacher (10.4 vs 7.4 RPM), so the
  students should learn from a rotlin teacher: `teacher_rotlin` (`-teacher-rotlin-v0`, queued) → `-student-pt-rotlin-v0` /
  `-student-mlp-h10-rotlin-v0`.
- Re-distill from `teacher_cont18k` if it improves further.
- Stage 2c (per-episode joint bias) is TODO. Stage 4 (export with the history inside the module) is done and verified in sim:
  [export.md](export.md), `scripts/export/export_history_policy.py` → `exported/policy_history.pt`.
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

- **2026-10-28 16:25** — `teacher_cont18k` cancelled before it started: the entropy run (B + entropy 5e-3: 7.33 RPM vs
  B 5.45, [PPO / capacity](ppo_and_capacity.md)) shows our entropy 1e-4 suppresses exploration, and the teacher was
  trained with 1e-4. Replaced by a fresh 10k teacher with entropy 5e-3 (`teacher_ent5e-3`, jobs 29842059 L40S /
  29842060 A100 dup). The students queued from the 1e-4 teacher stay queued (they test distillation itself); if the
  new teacher is better, the best student recipe is re-distilled from it.
