# Changelog

[Index](README.md). One line per dated entry from the old `ABLATION_RESULTS.md` log and `HYPERPARAM_NOTES.md`; the
full entries live in the linked topic files.

| When | What | Where |
|---|---|---|
| 2026-10-26 | Queue benchmarks; first runs submitted, only the L40S one kept (A, job 29823035) | [infrastructure](infrastructure.md) |
| 2026-10-27 | A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18 | [asymmetric critic](asymmetric_critic.md#baseline-a), [PPO](ppo_and_capacity.md#log) |
| 2026-10-27 | gpuv100 unusable (no sm_70 kernels) → 4 usable queues | [infrastructure](infrastructure.md#cluster-and-queues) |
| 2026-10-27 | Literature reviews: in-hand rotation; actuation / sensing / vision | [literature](literature.md) |
| 2026-10-27 15:20 | Interim matched-iteration results: asym critic ~3× reward; history / LSTM runs re-based on B | [asymmetric critic](asymmetric_critic.md#log), [history](history_and_memory.md#runs) |
| 2026-10-27 17:20 | B / C / sanity at matched iterations; their evals queued (29832523 / 25 / 26) | [asymmetric critic](asymmetric_critic.md#log) |
| 2026-10-27 19:45 | First eval (A): best @4500 1.37 RPM, degrades later; eval pipeline verified | [asymmetric critic](asymmetric_critic.md#log), [eval protocol](eval_protocol.md) |
| 2026-10-27 20:25 | Queued `abl_B_asym_ent5e-3` | [PPO and capacity](ppo_and_capacity.md#log) |
| 2026-10-28 01:30 | Literature → runs: RMA (HORA port) and B + drop penalty added | [RMA](rma.md#log), [literature](literature.md#paper--run-mapping-and-code-availability) |
| 2026-10-28 01:50 | Storage: quota 25.6 / 30 GB, prune script; checkpoint sort bug fixed | [infrastructure](infrastructure.md#storage-quota-and-pruning) |
| 2026-10-28 02:00 | Drop timing (A): drops throughout the episode → grasp-cache resets deprioritized | [rewards](rewards.md#log) |
| 2026-10-28 04:00 | Sanity (no-DR) eval: 10.55 / 10.59 RPM; `sanity_nodr_on_dr` queued (29836428) | [sim2real](sim2real.md#log) |
| 2026-10-28 06:45 | Evals of B seed 1, C, asym LSTM: asym critic ~4× A's best RPM | [asymmetric critic](asymmetric_critic.md#log), [history](history_and_memory.md#log) |
| 2026-10-28 07:30 | Jobs hang after finishing (orphaned W&B service); fixed in `train_job_ablation.sh` | [infrastructure](infrastructure.md#hanging-jobs) |
| 2026-10-28 07:35 | RMA stage 1 eval: 6.48 / 6.63 RPM | [RMA](rma.md#log) |
| 2026-10-28 09:30 | Reward review; queued `abl_B_rotlin` and `abl_B_rotsoft` | [rewards](rewards.md) |
| 2026-10-28 09:35 | B seed 42 eval: 5.66 / 6.55 RPM; C vs B at matched seed | [asymmetric critic](asymmetric_critic.md#log) |
| 2026-10-28 12:20 | Privileged teacher eval: 7.44 / 7.63 RPM, still improving → `teacher_cont18k` | [teacher-student](teacher_student.md#log) |
| 2026-10-28 14:30 | asym hist3 eval: ≈ B nominal, more robust perturbed | [history](history_and_memory.md#log) |
| 2026-10-28 | Paper → run mapping and code availability checked | [literature](literature.md#paper--run-mapping-and-code-availability) |
| 2026-10-28 | Decision: Phase 1 = architecture + reward; fixed enforced action delay is the first Phase 2 run | [index](README.md#phases), [sim2real](sim2real.md) |
| 2026-10-28 | Next architecture candidates agreed (alpha05, relact, big, student fine-tune) | [action space](action_space.md), [PPO](ppo_and_capacity.md), [teacher-student](teacher_student.md) |
| 2026-10-28 | Student PPO fine-tuning review (existing code, stabilizers) | [teacher-student](teacher_student.md#student-ppo-fine-tuning-background-from-the-old-notes-15) |
| 2026-10-28 | Docs restructured from `HYPERPARAM_NOTES.md` / `ABLATION_RESULTS.md` into `docs/ablation/` | [index](README.md) |
| 2026-10-28 | Entropy 5e-3 eval: 7.33 RPM (B 5.45); replicate + 1e-2 queued | [PPO / capacity](ppo_and_capacity.md) |
