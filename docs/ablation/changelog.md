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
| 2026-10-28 | teacher_cont18k cancelled → fresh teacher with entropy 5e-3 queued; entropy runs moved to the front of the A100 queue | [teacher-student](teacher_student.md) |
| 2026-10-28 | No-DR policy on DR task: 3.32 RPM → DR training is necessary; mirror eval queued (new `-asym-nodr-v0`) | [sim2real](sim2real.md) |
| 2026-10-28 | rotsoft = reward hack (‖ω‖ any axis pays; ~0 RPM); RMA stage 2 = 6.20 RPM (96 % of stage 1) | [rewards](rewards.md), [RMA](rma.md) |
| 2026-10-28 | hist10 eval: no speed gain (5.01 RPM), most robust under perturbation; history = robustness, not speed | [history](history_and_memory.md) |
| 2026-10-28 | student_pt (transformer, gradient_length 15) ran out of memory on the L40S at 8192 envs; rerun with 4096 envs | [teacher-student](teacher_student.md) |
| 2026-10-28 | speed-tracking sweep queued (does RPM follow the commanded speed?) for ent5e-3 and the teacher | [eval protocol](eval_protocol.md) |
| 2026-10-28 | grading = live demo speed + duration without drop → one-direction (+z, faster in 11/12 policies) max-speed run queued | [rewards](rewards.md) |
| 2026-10-28 | drop timing: half of eval drops are reset artifacts in the first 0.35–0.65 s; <3 % RPM cost, sim-only → grasp cache deprioritized | [eval protocol](eval_protocol.md) |
| 2026-10-28 | Random action holding (TeleDexter's action masking, arXiv 2607.11481) implemented + CPU-tested as `-asym-acthold-v0`: p ~ U(0, 0.1) per episode, whole hand, obs = command sent; not run, queued for Phase 2 with the fixed delay | [sim2real](sim2real.md#random-action-holding) |
| 2026-10-29 | Actuator net (arXiv 2603.04351 / Hwangbo 2019) implemented + CPU-tested as `-asym-actnet-v0`: delay + play + τ + bounded learned target correction on the implicit PD; v1→v1b held-out RMSE 2.20° → 1.38°; needs v2 chirps, Phase 2 | [actuator net](actuator_net.md) |
| 2026-10-29 | student_pt (transformer student): 6.89 / 7.52 RPM nominal / perturbed, DFSR 95 % = 93 % of the teacher's speed, best drop-free rate of any deployable policy | [teacher-student](teacher_student.md) |
| 2026-10-29 | History export: `scripts/export/export_history_policy.py` → `policy_history.pt` keeps the obs history inside TorchScript; bit-exact vs Isaac Lab's ObservationManager on CPU for student_pt, RMA s2, hist3/10 (LSTM/GRU ≤ 2e-6); found that play.py's LSTM export never resets its hidden state | [export](export.md) |
| 2026-10-29 | **rotlin wins**: 10.41 / 11.07 RPM nominal / perturbed (B 5.45, ent5e-3 7.33, teacher 7.44), DFSR 84 %, but less robust under perturbation and more off-axis wobble. alpha05 / relact moved onto rotlin + ent 5e-3; rotlin + ent 5e-3 queued | [rewards](rewards.md) |
| 2026-10-29 | C seed 1: 4.60 RPM → two-seed means C 5.34 vs B 5.45: DR params in the critic give nothing; seed spread ±0.7 RPM | [asym critic](asymmetric_critic.md) |
| 2026-10-29 | teacher-student pipeline on the rotlin reward: `-teacher-rotlin-v0` (queued, 10k), `-student-pt-rotlin-v0`, `-student-mlp-h10-rotlin-v0`; `scripts/check_env_cfgs.py` instantiates env cfgs on CPU before jobs start | [teacher-student](teacher_student.md) |
| 2026-10-29 | Export verified in sim (student_pt, 64 envs × 1000 steps, ~830 resets): rebuilt history == env obs on every step; exported == CPU policy exactly; GPU vs CPU arithmetic 3e-2 | [export](export.md) |
| 2026-10-29 | entropy replicate: seed 1 5.74 RPM (seed 0 7.33) → ent 5e-3 mean 6.54 vs B 5.45: likely positive, not established (seed spread 1.6 RPM) | [PPO / capacity](ppo_and_capacity.md) |
| 2026-10-29 | one-direction run with an unclipped speed reward was a reward hack (flings the cube at ~30 rad/s, 99.9 % drops): killed, speed reward now clipped at 3 rad/s, rerun as `abl_B_onedir_clip3_ent5e-3` | [rewards](rewards.md) |
| 2026-10-29 | ent5e-3 evaluated without DR: 8.58 RPM (7.33 with DR): DR at test time costs ~15 %; most of the gap to the no-DR-trained policy (10.55) is the conservative gait learned under DR | [sim2real](sim2real.md#sanity-runs-no-dr) |
| 2026-10-29 | speed sweep (ent5e-3): RPM 6.7–7.3 for commands 0.6–2.0 rad/s, i.e. the policy ignores the commanded speed; the speed term only meant "faster" | [eval protocol](eval_protocol.md) |
| 2026-10-29 | queued on rotlin + ent 5e-3: off-axis penalty −15 (A10) and the 1024-1024-512-256 network (A40) | [rewards](rewards.md), [PPO / capacity](ppo_and_capacity.md) |
| 2026-10-29 | relact fails without grasp-cache resets: the relative target starts at the open reset pose, the cube drops within 6 steps, no learning (killed at it 96) | [action space](action_space.md) |
| 2026-10-29 | teacher_ent5e-3: 7.96 RPM (best 8.16) vs 7.44, and 8 RPM already at 4k iterations (old teacher 3.8) → rotlin teacher rerun with entropy 5e-3 for 6k iterations | [teacher-student](teacher_student.md) |
| 2026-10-29 | entropy 1e-2: 7.60 RPM but DFSR 75 % (5e-3: 6.54 / 88 %, 1e-4: 5.45 / 88 %): speed rises with entropy, drops too at 1e-2; rotlin + ent 1e-2 queued | [PPO / capacity](ppo_and_capacity.md) |
| 2026-10-29 | queued `abl_B_rotlin_cont10k` (resume rotlin to 10k); README key results rewritten + deployability legend | [README](README.md) |
| 2026-10-29 | onedir_clip3 learns but plateaus at ~50 % of episodes ending in a drop (it 400–1000): queued onedir + drop penalty (`-asym-onedir-droppen-v0`) | [rewards](rewards.md) |
| 2026-10-29 | **drop penalty on B: 7.94 RPM (best 8.34 @3000 with 94 % drop-free) vs 5.45**: faster and more robust; rotlin / onedir + drop penalty running / queued | [rewards](rewards.md) |
| 2026-10-29 | Phase 2 checklist + sim2sim in MuJoCo (CPU): harness being built by a subagent; step 2 = fake ROS hand driver for the course's policy node | [sim2real](sim2real.md) |
| 2026-10-29 | rotlin + ent 5e-3: 9.50 / 10.43 RPM (rotlin 10.41 / 11.07): no extra speed, but 8.9 RPM already at 1k and 58 vs 44 % drop-free under perturbation; confirms rotlin's ~10 RPM level | [rewards](rewards.md) |
| 2026-10-29 | rotlin + ent 5e-3 + drop penalty: 8.29 RPM / 89 % drop-free (without penalty 9.50 / 81 %): on rotlin the penalty trades speed for safety; no gain under perturbation | [rewards](rewards.md) |
| 2026-10-29 | sim2sim harness (MuJoCo, CPU): no interface bug; MLPs keep 48–73 % of their Isaac speed and tie (rotlin 5.39 ≈ ent5e-3 5.38 RPM), student_pt only ~40 %; reset distribution note corrected; deploy needs Isaac's articulation joint order and raw last_action | [sim2sim](sim2sim.md) |
| 2026-10-29 | off-axis −15 on rotlin: the policy stops rotating (0.19 RPM, holds the cube still); keep −5; monitor the rotation reward in training | [rewards](rewards.md) |
| 2026-10-29 | onedir (+z, speed reward clipped at 3 rad/s): **15.2 RPM** (1.59 rad/s) but only 44 % drop-free (1.3 drops/min; perturbed 9 %); onedir + drop penalty running | [rewards](rewards.md) |
| 2026-10-29 | RTX 6000 pod online (Claude worker `rtx6000`, Isaac Sim 5.1 / Isaac Lab 2.3.2 installed in a venv, Vulkan via the EGL ICD): rotlin seed 1 and droppen seed 1 running at ~5 s/it with 2 runs on the GPU | [infrastructure](infrastructure.md) |
| 2026-10-29 | student_lstm crashed when loading the MLP teacher (rsl_rl 3.1.2 sizes a non-recurrent teacher for the RNN output); fixed in `student_recurrent.py` (CPU test: teacher actions identical). Student comparison (transformer / LSTM / MLP-h10) moved onto the rotlin teacher, all three start after it | [teacher-student](teacher_student.md) |
| 2026-10-29 | rotlin teacher: **11.85 RPM, 95 % drop-free** (best so far, still improving at 6k); three students queued on it | [teacher-student](teacher_student.md) |
| 2026-10-29 | rotlin + ent 1e-2 killed at it 640 (action std grows to 3, 95 % drops, no learning); its A10 slot gets a copy of rotlin_cont10k | [PPO / capacity](ppo_and_capacity.md) |
| 2026-10-29 | alpha05 on rotlin killed at it 224 (4-step episodes: the start noise knocks the cube off); retry with init_noise_std 0.5 queued; cont10k running on A10 | [action space](action_space.md) |
| 2026-10-29 | pod on CUDA MPS: 3 runs at 0.658 it/s total = **1.84x** the 2-run non-MPS throughput, each run faster; seeds resumed from model_2000; a 4th run (16384 envs) being added | [infrastructure](infrastructure.md) |
| 2026-10-29 | decision rule: MuJoCo sim2sim is informational; design decisions for new ablations are based solely on the Isaac eval | [sim2sim](sim2sim.md) |
| 2026-10-30 | drop penalty replicated: seed 1 7.00 RPM → two-seed mean 7.47 vs B 5.45 (+2.0 RPM); LSTM student running on the pod | [rewards](rewards.md) |
