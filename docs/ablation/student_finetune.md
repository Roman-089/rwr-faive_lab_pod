# Student PPO fine-tuning (teacher → distilled student → RL)

Code: `source/faive_lab/faive_lab/tasks/orca_inhand/student_finetune.py`. Runner cfgs: `PPORunnerCfgOrcaV2StudentFinetune`
and `PPORunnerCfgOrcaV2StudentResidual` in `rsl_rl_teacher_student_cfg.py`. Tasks are registered in `__init__.py`.

| task id | what it trains |
|---|---|
| `orca-v2-inhand-repose-student-mlp-h10-ft-v0` | full fine-tune of the MLP-h10 student (actor ← student, critic ← teacher critic) |
| `orca-v2-inhand-repose-student-mlp-h10-res-v0` | fallback: frozen student plus a zero-initialized residual head |

Both use the env `OrcaV2ReposeTeacherHist10EnvCfg`: the 10-step noisy `policy` history for the actor, and the noise-free
81-dim `teacher` group for the critic and the regularizer.

**Why.** The students are trained by imitation, so the teacher caps them (teacher: 7.4 RPM). They also miss the
teacher's privileged cube pose. RL on the student's own observations can go past the imitation optimum. Done naively,
it collapses at the start. Parkour in the Wild says this outright: performance "steadily degrades once the RL loss is
applied". Each design choice below targets one cause of that collapse.

## 1. Literature

[code] = I read the setting in the cloned repo (`scratchpad/repos/`). [paper] = from the PDF text. I fetched all arXiv
ids below from arxiv.org and checked them against the titles, except Athena-WBC.

| paper | code | how it stabilizes the start of fine-tuning (numbers) |
|---|---|---|
| **TGRL**, Shenfeld et al., ICML 2023, arXiv 2307.03186 | github.com/idanshen/cleanrl, `cleanrl/tgrl_continuous_action.py`. github.com/Improbable-AI/tgrl does not exist; the project page sites.google.com/view/tgrl-paper links the cleanrl fork. | [code] SAC-based (off-policy). Loss = SAC actor loss + λ·(KL(π‖π_T) − Q_KL) + an auxiliary reward-only actor. λ starts at 1.0 and moves ±0.01 every 1000 env steps by the sign of (main − aux mean return over the last 20 episodes), floored at 0. The teacher is frozen. In this file the teacher reads the same obs as the student; the paper's privileged-teacher setting is not in this script. |
| **DPPO**, Ren et al., arXiv 2409.00588 | github.com/irom-princeton/dppo | [code] `n_critic_warmup_itr` 0 (gym), 1 (furniture), 2 (robomimic), with a **randomly initialized** critic: only the critic optimizer steps during warm-up. Actor LR 1e-5 (furniture) / 1e-4 (gym, robomimic), critic LR 1e-3 (10–100× the actor), cosine schedule with 10 warm-up steps, min actor LR 1e-6. Gaussian baseline: PPO clip **0.01**, std learned, init 0.04–0.1, **clamped to [0.01, 0.2]**. Diffusion: clip 0.001–0.01, `min_sampling_denoising_std` 0.04–0.1. `ent_coef` 0 by default, `target_kl` 1 (effectively off), 5–10 epochs, AdamW. A BC loss option exists but is off in every cfg. |
| **Kickstarting**, Schmitt et al., arXiv 1803.03835 | none | [paper] L = L_RL + λ_k·H(π_T‖π_S) on the student's own trajectories. Table 2, DMLab-30 score at 10B frames (teacher 39.7): constant λ=1 → 41.2, λ=2 → 40.8, so a **constant weight plateaus at the teacher**. Linear 1→0 at 1B → 59.4, at 2B → 60.8, at 4B → 53.1. Linear 2→0 at 1B/2B/4B → 60.5/51.5/50.3. PBT → 56.9, and PBT drove λ "to almost negligible weight after 2 billion steps". Best: anneal to 0 within 10–20 % of training. |
| **Fine-tuning RL is secretly a forgetting problem**, Wołczyk et al., ICML 2024 (spotlight), arXiv 2402.02868 | none linked | [paper] Forgetting in states the fine-tuned policy has not reached yet; retention (KS/BC) is the fix. NetHack (APPO): **critic (baseline head) pre-trained for 500M env steps with the rest frozen**; KS weight **0.5 × 0.99998^step**; BC weight 2.0 without decay; **entropy bonus off** whenever retention is used; encoders frozen; EWC 2·10⁶. KS was best on NetHack, BC on RoboticSequence. Retention is applied to the actor only, never the critic. |
| **ResiP** (From Imitation to Refinement), Ankile et al., arXiv 2407.16677 | github.com/ankile/robust-rearrangement | [code] Frozen base policy plus a residual MLP (2×256) on [obs, base action]. Last layer **initialized with std 0.0 (zero)**, residual × `action_scale` **0.1**. log-std −1.0 fixed (base cfg) or −1.5 learned (MLP cfg), i.e. effective noise 0.022–0.037 in normalized action units. Actor LR 3e-4 with 5 warm-up steps (cosine), critic LR 5e-3, critic last layer std 0.25 / bias 0.25. Entropy 0, clip 0.2, `target_kl` 0.1 early stop, 50 epochs × 1 minibatch, reward normalization + clip 5. A value-only phase exists (`n_iterations_train_only_value`), default 0. |
| **Policy Decorator**, Yuan, Mu et al., arXiv 2412.13630 | github.com/tongzhoumu/policy_decorator | [code] SAC residual, bounded: a = a_base + α·tanh-residual, α = `res_scale` 0.05 (RGBD) / 0.1 (diffusion) / 0.2 (BeT). **Progressive exploration**: per env, the residual is enabled with probability min(t/H, 1), H = 100k steps. Policy/Q LR 1e-4, learning starts after 8k steps. |
| **Distillation-PPO** (humanoid perceptive locomotion), Zhang et al., arXiv 2503.08299 | none (the arXiv page lists no repo) | [paper] Stage 2 trains the student with L = α·‖π_T − π_S‖² + β·L_PPO, **α = β = 0.5, constant, no annealing**. Teacher fully observed, student partially. This is joint DAgger+PPO rather than a fine-tune of a converged student. |
| **Parkour in the Wild**, Rudin et al., arXiv 2505.11164 | none | [paper] Distil several experts with DAgger, then fine-tune with RL. "Done naively, the performance of the policy steadily degrades once the RL loss is applied." Three fixes: (1) **action noise during distillation**, so the student is robust to exploration noise; (2) a **reduced initial std**, together with conservative RL hyper-parameters; (3) **critic pre-training with the policy weights frozen**. No numbers given in the text. |
| **Light-Loco-Parkour**, Chen et al., arXiv 2608.02653 | none | [paper] 14k distillation iterations, then a **1k-iteration** reward-only RL fine-tune. Instead of a KL to the teacher, dense-reward terrain groups and an AMP prior act as anchors against drift. |
| **Athena-WBC**, arXiv 2607.04837 (from a web search; arxiv.org returned 404, **not verified**) | ? | (per the search summary) Actor from the DAgger student, "small initial action standard deviation", critic-only warm-up with the actor frozen, following Parkour in the Wild. |

Hora, Visual Dexterity, DeXtreme and DextrAH-RGB stop at distillation and do no RL fine-tuning of the student.

### Collapse causes in rsl_rl 3.1.2 [code]
- `Distillation.update` trains only `act_inference` (the mean). The student's `log_std` stays at `init_noise_std` = 0.1,
  so PPO would start with an untrained std.
- `PPO.update` adaptive schedule: LR × 1.5 whenever KL < desired_kl/2, up to **1e-2**. LR / 1.5 when KL > 2·desired_kl,
  down to 1e-5. One optimizer and one LR cover actor and critic.
- The critic would start random, and the advantage normalization blows its noise up into full-size policy gradients.
- New finding: Adam's first steps are ≈ lr·sign(g) regardless of the loss scale. At unfreeze, a cold Adam state at full
  LR moves the whole network coherently. In the CPU test, the first actor iteration at LR 1e-5 gave KL 0.03–0.44 (target
  0.005).
- Teacher evidence: the teacher run ended at LR **1e-5**, the adaptive schedule's floor (optimizer state of `model_9999.pt`).
  So near convergence, even 1e-5 overshot desired_kl = 0.01.

## 2. Design

| # | element | choice | why |
|---|---|---|---|
| 1 | actor init | `student.*` → `actor.*`, `student_obs_normalizer.*` → `actor_obs_normalizer.*`; optimizer not loaded; iteration 0 | Identical deterministic actions at t=0 (verified bit-exact). |
| 2 | critic init | teacher PPO checkpoint `critic.*`, `critic_obs_normalizer.*`; critic obs = `teacher` group (asymmetric) | Trained privileged value instead of a random one. The teacher critic matches the obs group exactly (81-dim). |
| 3 | warm-up | `critic_warmup_iterations=150`: the loss is the value loss only. Actor and std get no gradient (Adam skips `grad=None`, so the actor's Adam state stays empty). Then the actor LR ramps linearly over `actor_lr_warmup_iterations=25`. | DPPO, Parkour in the Wild, Wołczyk (500M steps). The critic still has to adapt from V^teacher to V^student (different policy and noise). The ramp handles the Adam cold start (ResiP 5, DPPO 10 warm-up steps). |
| 4 | std | the teacher's final per-joint std (0.18–0.57, mean 0.377) × `init_std_scale=1.0`, **frozen** (`learn_std=False`); entropy 0 | Matches the noise level the critic was trained under, and removes std collapse or blow-up as a failure mode. `learn_std=True` clamps to [`std_min`, `std_max`] (DPPO-style). |
| 5 | LR / clip | actor LR starts at 1e-5 (the teacher's final LR); adaptive with `desired_kl=0.005` (half the teacher's), clamped to [1e-6, 1e-4]; critic LR 1e-4 fixed in its own param group; clip 0.1 | rsl_rl's ×1.5-to-1e-2 growth is the main runaway risk, so the clamp caps it. Separate actor/critic LRs follow DPPO and ResiP. |
| 6 | teacher term | β·KL(N(μ_T, σ) ‖ N(μ_S, σ)) = β·Σ_j (μ_S − μ_T)_j² / (2σ_j²), on the student's own rollout states, with μ_T from the frozen teacher actor on the `teacher` group. β = 1 during warm-up, then linear to 0 over 1000 iterations. `teacher_loss_type="mse"` gives the plain β·MSE. | Kickstarting (anneal to 0, otherwise it plateaus at the teacher) and Wołczyk (KS weight 0.5 with decay). |
| 7 | normalizers | the actor normalizer is frozen (kept in eval mode); the critic normalizer keeps updating; the teacher normalizer is frozen | The student's input statistics are part of its function. |
| 8 | logs (`Loss/*` in W&B) | `behavior_mse` (same quantity as Distillation's `behavior`), `teacher` (the KL term), `teacher_coef` (β), `explained_variance` (rollout values vs returns, before the update), `kl`, `actor_lr`, `actor_lr_ramp`, `critic_lr`, `actor_frozen`, `std`, plus PPO's `value_function`, `surrogate`, `entropy` | |

**Deviations from the starting recipe:**
- The teacher term is the Gaussian KL rather than raw MSE, so β is in the same units as kickstarting's λ and PPO's KL.
  The plain MSE is still logged as `behavior_mse`.
- I added an actor-LR ramp, because of the Adam cold-start finding above.
- The actor LR starts at 1e-5, not 1e-4: the teacher's own schedule sat at 1e-5.

**Checkpoint plumbing.** `train.py` loads one checkpoint: `--load_run/--checkpoint` = the **student** Distillation run.
`load_state_dict` recognizes its `student.*` keys and pulls the critic, the teacher actor and the std from
`policy.teacher_checkpoint`. That defaults to the 7.4 RPM teacher, `logs/rsl_rl/orca_v2_repose/2026-10-27_23-55-26_teacher/model_9999.pt`,
relative to the repo root. Override it with Hydra: `agent.policy.teacher_checkpoint=/abs/path/model_X.pt`.

The student checkpoint also contains `teacher.*` (StudentTeacher copies the teacher actor). These are compared against
the teacher checkpoint, and a `[WARNING]` is printed when the student was distilled from a different teacher. A
fine-tune checkpoint has `teacher_actor.*` keys and loads strictly, as a resume (optimizer, iteration and `ft_iter`
restored). This is the path `eval_policy.py` and `play.py` take. The inference policy is `actor(actor_obs_normalizer(policy obs))`
only; the teacher group is never read.

**Residual fallback** (`-res-v0`, `ActorCriticResidualFinetune`). `actor = ResidualActor`: student (frozen) + 0.1 ·
MLP[256,128]([obs_norm, a_student]), with the last layer zero-initialized. Its actor LR is 1e-4 (max 3e-4), and β=0
because the residual is bounded and starts at zero. `actor` is the combined module, so `act_inference`, the JIT export
and the ONNX export all see student + residual. Not ported from Policy Decorator: progressive exploration (H=100k
steps), since the frozen-std noise is applied to the whole action from t=0.

**LSTM / transformer students (designed, not implemented).** `_init_from_distillation` raises `NotImplementedError` on
`memory_s.*`.
- *LSTM*: subclass `ActorCriticRecurrent` with a recurrent actor (`memory_s.*` → `memory_a.*`, `student.*` → `actor.*`).
  The critic must be the teacher's **MLP** on the `teacher` group, so `memory_c` is dropped and `evaluate` is overridden
  to skip it. `PPOFinetune.update` already uses `recurrent_mini_batch_generator`. `teacher_mean` needs no hidden state.
  `obs_groups` stays the same, and the env is `OrcaV2ReposeTeacherEnvCfg` (no history).
- *Transformer*: the actor is `_TransformerStudent` (`student.*` → `actor.*`), with `_tokens` and the per-token
  normalizer reused from `transformer_student.py`, and the action head as the output. **Dropout must be 0, or the module
  must stay in eval mode during rollouts and updates**: train-mode dropout makes the PPO ratio noisy. The recon head is
  unused (or kept as an auxiliary loss). The exporter needs `actor[0].in_features`, so give the actor a `__getitem__`
  like `ResidualActor` has.

## 3. Defaults (`PPORunnerCfgOrcaV2StudentFinetune`)

| | value |
|---|---|
| obs_groups | policy=`policy` (740 = 10×74), critic=`teacher` (81), teacher=`teacher` |
| networks | actor [512,512,256,128] (= student), critic [512,512,256,128] (= teacher), elu |
| max_iterations / save_interval | 3000 / 50 (the cluster job script overrides save_interval to 500) |
| PPO | clip 0.1, entropy 0, 5 epochs × 4 minibatches, γ 0.99, λ 0.95, value coef 1, clipped value loss, grad norm 1 |
| actor LR | 1e-5 initial, adaptive (desired_kl 0.005) within [1e-6, 1e-4], ramp 25 iterations after warm-up |
| critic LR | 1e-4 fixed |
| warm-up | 150 critic-only iterations (≈ 30M env steps at 8192 envs × 24 steps) |
| std | the teacher's per-joint std × 1.0, frozen |
| β | 1.0 (KL units), linear to 0 over iterations 150–1150 |
| residual (`-res-v0`) | MLP [256,128], scale 0.1, zero-init; actor LR 1e-4 (max 3e-4); β 0 |

## 4. Launch

Prerequisite: a finished student run `logs/rsl_rl/orca_v2_repose/<date>_student_mlp_h10/model_1999.pt` (task
`-student-mlp-h10-v0`). `resume=True` is already set in the cfg, so `--resume` is not needed. Run from the repo root:

```bash
python scripts/rsl_rl/train.py --task orca-v2-inhand-repose-student-mlp-h10-ft-v0 --headless --num_envs 8192 \
    --load_run <date>_student_mlp_h10 --checkpoint model_1999.pt --run_name student_mlp_h10_ft \
    --logger wandb --log_project_name orca_v2_repose
# optional: agent.policy.teacher_checkpoint=/zhome/b3/e/222334/rwr_faive_lab/logs/rsl_rl/orca_v2_repose/2026-10-27_23-55-26_teacher/model_9999.pt
```

On the cluster (the job script adds `agent.save_interval=500`; the run takes ≈ 4.3 s/iteration on an L40S, so ≈ 3.6 h for 3000):

```bash
hpc/submit_ablation.sh orca-v2-inhand-repose-student-mlp-h10-ft-v0 student_mlp_h10_ft 3000 gpul40s 05:00 \
    "--load_run <date>_student_mlp_h10 --checkpoint model_1999.pt"
```

For the residual fallback, use the same arguments with `-student-mlp-h10-res-v0` and `student_mlp_h10_res`. To evaluate:

```bash
python scripts/eval/eval_policy.py --task orca-v2-inhand-repose-student-mlp-h10-ft-v0 --headless \
    --checkpoints logs/rsl_rl/orca_v2_repose/<ft run>/model_0.pt .../model_<k>.pt
```

`model_0.pt` is saved during the warm-up, so its actor equals the student: it is the **baseline under the identical eval
setting**. Passing the student's own `model_1999.pt` to the `-ft-v0` task also works, since it loads through the same
init path.

## 5. What to monitor, and when to roll back

During the warm-up (iterations 0–150):
- `Loss/explained_variance` should rise toward ≥ 0.8. If it is still < 0.5 at iteration 150, extend the warm-up
  (`agent.algorithm.critic_warmup_iterations=300`) or set `agent.algorithm.use_clipped_value_loss=False`; clip 0.1 also
  limits value moves.
- `Loss/value_function` should fall.
- `Train/mean_reward` should be **flat**, because the actor is frozen. A drop here means a pipeline bug (normalizer,
  wrong checkpoint).

After unfreezing:
- `Loss/kl` near 0.005.
- `Loss/actor_lr`: if it sits at the 1e-6 floor permanently, the actor is effectively still frozen. Raise `desired_kl`
  to 0.01, or lower β.
- `Loss/behavior_mse`: it may grow slowly after β → 0. A jump of > 3× within 50 iterations means collapse.
- `Train/mean_episode_length`: drops mean the cube is being dropped.

**Rollback**: evaluate every saved checkpoint with `eval_policy.py` and keep the best by RPM. Abort and roll back to the
last good checkpoint if RPM falls **> 20 % below `model_0.pt`** (the student baseline), or if DFSR drops > 10 points.
Then retry once with `init_std_scale=0.5` and `critic_warmup_iterations=300`. If that also fails, switch to the
residual task.

## 6. Open risks

- **The std mismatch is the biggest unknown.** The student was distilled under σ = 0.1 action noise (rsl_rl
  `Distillation.act` samples with `init_noise_std` 0.1); the fine-tune explores at the teacher's 0.18–0.57. Parkour in
  the Wild recommends a reduced initial std. If reward drops already at iteration 0 (that is, during the warm-up, before
  any actor update), the noise itself is the problem: use `agent.policy.init_std_scale=0.5`.
- The critic's normalizer and value are those of the noise-free teacher policy. The student's returns are lower, so
  early advantages carry a negative bias until the critic adapts; the warm-up exists for that. Neither its length nor
  its value-loss clipping has been checked on the real reward scale.
- β in KL units with σ ≈ 0.18 on some joints weights those joints ~10× more than the wide ones. Use
  `teacher_loss_type="mse"` if that over-anchors.
- Not tested on GPU or Isaac Sim: env obs layout, real reward scale, throughput and memory. The rollout storage now
  holds 740 + 81 floats per env and step, ≈ 0.65 GB at 8192 × 24.
- `train.py` with `resume=True` and no `--load_run` picks the newest run in `orca_v2_repose`. If that is not a
  Distillation run, the load fails with a clear `ValueError` rather than silently loading the wrong thing.
- No multi-GPU test. The KL is all-reduced, so every rank sets the same LR.
- LSTM and transformer students are not implemented (§2).

## 7. Verification (CPU, isaaclab stubbed)

Test: `/tmp/claude-322334/-zhome-b3-e-222334/e98b2254-0dd6-46ce-be5a-af9ce9954bee/scratchpad/ftest/test_student_finetune.py`.
It uses the **real** Isaac Lab `configclass` and rsl_rl cfgs (pure-python modules loaded without pxr), the real package
`__init__` (task registration), the **real teacher checkpoint**, a student produced by the real `DistillationRunner`
with `DistillationRunnerCfgOrcaV2` on a fake VecEnv, and the real `OnPolicyRunner` load → learn → save → load path.
- (a) After `runner.load(<student ckpt>)`: |actor − student| = 0, |V − V_teacher| = 0, |μ_teacher − teacher
  act_inference| = 0, |std − teacher std| = 0 (all bit-exact).
- (b) Warm-up iterations 0–2: actor params, actor normalizer and log_std unchanged; critic changed.
- (c) β per iteration [1, 1, 1, 1, 0.75, 0.5, 0.25, 0, 0] (warm-up 3, anneal 4); the ramp is 0 → 0.5 → 1; std constant
  at 0.3772; entropy = the analytic Gaussian entropy 6.3821 every iteration; entropy coef 0.
- (d) save → new runner → load: all tensors equal, iteration, `ft_iter` and optimizer LRs restored; the inference
  policy is deterministic, equals `actor(normalizer(policy obs))`, and ignores the teacher group. JIT export matches
  bit-exactly and the ONNX export runs, for both the MLP and the residual actor.
- Guards: a student from another teacher prints a warning; the teacher PPO checkpoint passed as `--load_run` raises
  `ValueError`; a missing teacher checkpoint raises `FileNotFoundError`.
