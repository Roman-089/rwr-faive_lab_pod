# Ablation study: RWR26 Task 1, Subtask 1

Proprioceptive in-hand z-rotation of a 45 mm cube with the ORCA v2 hand in Isaac Lab
(task `orca-v2-inhand-repose-v0`, branch `ablation-study`). Demo hardware is a plain ORCA hand without joint sensors,
so the deployed policy sees motor-encoder proprio only (74-dim obs at 20 Hz).

## Phases

- **Phase 1 (now): architecture + reward study** (decided 2026-10-28: find the best architecture and reward first, then
  iterate on sim2real). All runs on the **baseline env (A)**, compared by a fixed eval: MLP (A) · asymmetric critic (B) ·
  critic + DR params (C) · history (hist3 / hist10) · LSTM actor-critic · RMA · privileged teacher → distilled students
  (MLP-h10, LSTM, transformer + recon) · reward variants · entropy. Next candidates, on the winner of the reward wave
  (B, `-rotlin` or `-rotsoft`, decided by eval): `alpha05`, `relact`, `big`, student fine-tune.
- **Phase 2: sim2real** (parked). Sim2real ablations change the training distribution and can only be judged on
  hardware. The first Phase 2 run is a **fixed, enforced action delay** matched in sim and on deploy
  ([sim2real](sim2real.md#fixed-action-delay-sim--deploy), `DelayedEMAJointPositionToLimitsActionCfg`).

## Key results so far

The asymmetric critic is the only large architectural effect: A's best checkpoint reaches 1.37 RPM, B 5.45 RPM (two-seed
mean) with ~3× the rotation accuracy and drop-free rate. C (+ cube velocity + DR params in the critic) adds a small,
not yet conclusive +0.4 RPM; RMA stage 1 (6.48 RPM) shows that knowing the DR sample buys little. A 3-step history and an
LSTM actor-critic match B nominally and drop less under perturbation. The privileged teacher is the best policy so far
(7.44 RPM, 94 % drop-free) and still improving at 10k, which sets the ceiling for the students. The no-DR sanity run
reaches 10.55 RPM, so DR + obs noise cost about half the speed; the reward review suggests speed is barely rewarded
(~6 of 14.3 RPM commanded). Every policy rotates faster but drops 3–8× more under the perturbed eval.

## All runs

Status as of 2026-10-28 ~15:00. Eval = nominal / perturbed at the last checkpoint ([eval protocol](eval_protocol.md));
"best" = best checkpoint if better. Task ids are `orca-v2-inhand-repose<suffix>`.

| Run | Thread | Task id | Change vs baseline | Job / GPU | Status | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|---|---|---|---|---|
| A `firstrun_l40s` | [asym critic](asymmetric_critic.md) | `-v0` | baseline, 10k iters | 29823035 / L40S | done | 0.61 / 1.16 (best 1.37 / 1.62 @4500) | 11.6 / 14.6 (19.1 / 19.0) | 22 / 20 (33 / 25) | 1.8 / 2.2 (1.3 / 2.0) |
| B `abl_B_asym` | [asym critic](asymmetric_critic.md) | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | done | 5.66 / 6.55 | 66.3 / 62.0 | 86 / 52 | 0.20 / 0.98 |
| C `abl_C_asym_dr` | [asym critic](asymmetric_critic.md) | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | done | 6.08 / 6.40 (best 6.19 / 6.77 @4000) | 72.2 / 63.2 | 84 / 69 | 0.20 / 0.44 |
| `abl_B_asym_seed1` | [asym critic](asymmetric_critic.md) | `-asym-v0` | B, seed 1 | 29832230 / L40S | done | 5.23 / 5.92 (best 5.48 / 6.16 @2000) | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
| `abl_C_asym_dr_seed1` | [asym critic](asymmetric_critic.md) | `-asym-dr-v0` | C, seed 1 | 29838635 / A10 | queued | | | | |
| sanity `sanity_nodr` | [sim2real](sim2real.md#sanity-runs-no-dr) | `-nodr-v0` | no DR, no obs noise (own no-DR eval) | 29829724 / A10 | done | 10.55 / 10.59 | 85.8 / 85.7 | 83 / 69 | 0.19 / 0.47 |
| `eval_sanity_nodr_on_dr` | [sim2real](sim2real.md#sanity-runs-no-dr) | eval on `-v0` | sanity `model_4999` evaluated with DR | 29836428 | queued | | | | |
| `arch_asym_hist3` | [history](history_and_memory.md) | `-asym-hist3-v0` | B + MLP on 3-step history | 29831099 / A10 | done | 5.33 / 5.82 | 64.0 / 58.1 | 83 / 66 | 0.20 / 0.59 |
| `arch_asym_hist10` | [history](history_and_memory.md) | `-asym-hist10-v0` | B + MLP on 10-step history | 29831100 / A40 | running | | | | |
| `arch_asym_lstm` | [history](history_and_memory.md) | `-asym-lstm-v0` | B + LSTM actor-critic | 29831098 / A100 | done | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
| `teacher` | [teacher-student](teacher_student.md) | `-teacher-v0` | privileged teacher (not deployable), 10k | 29830926 / L40S | done | 7.44 / 7.63 | 81.4 / 74.4 | 94 / 73 | 0.06 / 0.31 |
| `teacher_cont18k` | [teacher-student](teacher_student.md) | `-teacher-v0` | teacher resumed from model_9999, +8k | 29840146 / L40S (+ A100 dup 29840147) | queued | | | | |
| `student_mlp_h10` | [teacher-student](teacher_student.md) | `-student-mlp-h10-v0` | distill → MLP on 10-step history | 29839014 / A100 | queued | | | | |
| `student_lstm` | [teacher-student](teacher_student.md) | `-student-lstm-v0` | distill → LSTM | 29839015 / A40 | queued | | | | |
| `student_pt` | [teacher-student](teacher_student.md) | `-student-pt-v0` | distill → transformer + recon loss | 29839013 / L40S | queued | | | | |
| `arch_rma_s1` | [RMA](rma.md) | `-rma-teacher-v0` | B + RMA extrinsics encoder (not deployable) | 29835277 / L40S | done | 6.48 / 6.63 (best 6.55 / 7.03 @4000) | 74.5 / 69.3 | 89 / 66 | 0.16 / 0.61 |
| `arch_rma_s2` | [RMA](rma.md) | `-rma-student-v0` | RMA adaptation module, 30-step history, 2k | 29837730 / L40S | running | | | | |
| `abl_B_rotlin` | [rewards](rewards.md) | `-asym-rotlin-v0` | B, linear clipped rotation reward + off-axis | 29838450 / L40S | queued | | | | |
| `abl_B_rotsoft` | [rewards](rewards.md) | `-asym-rotsoft-v0` | B, magnitude σ 0.75, mag 60 / dir 30 | 29838451 / A100 | running | | | | |
| `abl_B_droppen` | [rewards](rewards.md) | `-asym-droppen-v0` | B + −100 return on drop | 29835278 / A40 | queued | | | | |
| `abl_B_asym_ent5e-3` | [PPO / capacity](ppo_and_capacity.md) | `-asym-v0` | B + entropy_coef 5e-3 | 29833758 / A100 | running | | | | |
| rotlin + drop penalty | [rewards](rewards.md) | – | rotlin + −200 drop weight (−10 return) | – | planned | | | | |
| position / velocity reward | [rewards](rewards.md) | – | position σ 0.05, weight 5, + cube lin-vel penalty | – | planned | | | | |
| `alpha05` | [action space](action_space.md) | – | EMA α 0.2 → 0.5 | – | planned | | | | |
| `relact` | [action space](action_space.md) | – | relative joint targets | – | planned | | | | |
| `big` | [PPO / capacity](ppo_and_capacity.md) | – | MLP 1024-1024-512-256 | – | planned | | | | |
| `student_mlp_h10_ft` | [student fine-tune](student_finetune.md) | `-student-mlp-h10-ft-v0` (fallback `-student-mlp-h10-res-v0`) | PPO fine-tuning of the MLP-h10 student: teacher critic, 150-iter critic warm-up, kickstarting β 1 → 0 | – | implemented + CPU-tested; launch after `student_mlp_h10` | | | | |
| fixed-delay run | [sim2real](sim2real.md#fixed-action-delay-sim--deploy) | – | fixed 1-step action delay + next-tick actuation (Phase 2) | – | planned | | | | |

Cancelled before starting (2026-10-27): A-based `-hist3` / `-hist10` / `-lstm` (29830927 / 28 / 30). Registered but
parked sim2real tasks (`-delay`, `-jointbias`, `-handgrav`, `-wristfix`, `-droppen`, `-hist3`): [sim2real](sim2real.md#registered--planned-sim2real-runs-parked).
Training rewards, episode lengths and drop rates per run are in the topic files.

## Topic files

| File | Contents |
|---|---|
| [eval_protocol.md](eval_protocol.md) | metrics (RPM, RA, DFSR, DC), nominal / perturbed eval, checkpoint selection, eval pipeline |
| [asymmetric_critic.md](asymmetric_critic.md) | A baseline, B, C, seeds |
| [history_and_memory.md](history_and_memory.md) | hist3, hist10, asym LSTM |
| [teacher_student.md](teacher_student.md) | privileged teacher, continuation, MLP-h10 / LSTM / PT students, distillation |
| [student_finetune.md](student_finetune.md) | PPO fine-tuning of the distilled student |
| [rma.md](rma.md) | RMA stage 1 / 2, HORA port |
| [rewards.md](rewards.md) | reward literature, diagnosis, rotlin, rotsoft, drop penalty, planned variants |
| [ppo_and_capacity.md](ppo_and_capacity.md) | entropy run, bigger network, PPO hyperparameter ideas |
| [action_space.md](action_space.md) | EMA α, relative actions, control frequency |
| [sim2real.md](sim2real.md) | Phase 2 (parked): delay, joint bias, gravity, wrist, DR ranges, actuator models, sanity runs, deploy facts |
| [literature.md](literature.md) | PT paper takeaways, general literature reviews, paper → run mapping, code availability |
| [infrastructure.md](infrastructure.md) | DTU LSF queues, limits, quota / pruning, hanging jobs, submit scripts |
| [changelog.md](changelog.md) | dated one-line timeline |

## Running and evaluating

Variants are task ids in `tasks/orca_inhand/orca_v2_repose_ablation_env_cfg.py` (seed 42, 8192 envs, baseline PPO cfg
unless noted). Submit with `hpc/submit_ablation.sh TASK RUN_NAME ITERS QUEUE WALLTIME [EXTRA]`; the job trains, then runs
the nominal + perturbed eval on 5 checkpoints, renders MuJoCo videos and uploads to W&B (`rwr-fast/orca_v2_repose`).
The job scripts are not in git. Details: [infrastructure](infrastructure.md), [eval protocol](eval_protocol.md).
