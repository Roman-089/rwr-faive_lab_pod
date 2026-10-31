# Exporting history policies for the robot

[Index](README.md) · related: [history](history_and_memory.md), [teacher-student](teacher_student.md), [RMA](rma.md),
[deploy-repo facts](sim2real.md#deploy-repo-facts-task-pdf)

## Why

The policy node (`rwr_faive_system/experiments/rl_inference/policy_inference_cube_spin_v2.py`) loads a TorchScript
file and calls it once per 50 ms control step with the **current** 74-dim observation (joint_pos 17 | joint_vel 17 |
goal_command 6 | last_action 17 | last_joint_pos_cmd 17). Our best deployable policies (`student_pt`, `arch_rma_s2`,
the history MLPs) were trained on an observation **history** that Isaac Lab's ObservationManager builds, so the
exported module has to keep that history itself. A wrong history layout or start-up fill would fail silently on the
robot (the policy still outputs plausible actions), so the export is checked for exact equivalence with training.

play.py's `exported/policy.pt` (Isaac Lab exporter) does not do this. It wraps `actor` / `student` + normalizer for one
flat input. For `student_pt` and `arch_rma_s2` the file scripts, but every call fails (shape errors for both a 74- and
a 740 / 2220-dim input). For `arch_asym_hist10` / `hist3` it needs the 740 / 222-dim flat history as input, which the
node does not build. For recurrent policies it has no `reset()` ([below](#recurrent-policies-lstm-gru)). It stays the
right export for plain single-step MLPs (e.g. `abl_B_asym_ent5e-3`).

## Usage

```bash
# CPU, no Isaac Sim, ~10 s; any run with params/agent.yaml + params/env.yaml
python scripts/export/export_history_policy.py --run logs/rsl_rl/orca_v2_repose/<run> [--checkpoint model_X.pt] \
    [--task <task id>] [--overwrite] [--no_check]
```

- Writes `<run>/exported/policy_history.pt` + `policy_history.json`. The JSON is also embedded in the .pt
  (`torch.jit.load(p, _extra_files={"policy_history.json": ""})`). It never writes `policy.pt`.
- Default checkpoint: the latest `model_*.pt`. Default task id: `<run>/eval/metrics.json`.
- After writing, the script reloads the file and compares it with the original policy's `act_inference` on a random
  observation sequence with resets. The training-time input is built with Isaac Lab's `CircularBuffer` the way the
  ObservationManager does. The script exits 1 on a mismatch.
- The policy is rebuilt without Isaac Sim. The rsl_rl / repo policy class comes from `agent.yaml`
  (`policy.class_name` + kwargs). The flat policy-input size comes from `env.yaml` (term list, `history_length`). The
  sizes of the unused critic / teacher sets come from the checkpoint. Then `load_state_dict(strict=True)`, as the
  runner does. Repo classes (`transformer_student.py`, `rma.py`, `student_finetune.py`) are loaded from their files
  without running the package `__init__` (gym registration, Isaac Sim imports). Their cfg classes' `configclass` /
  `isaaclab_rl` imports are stubbed while the file executes; inside Isaac Sim the normal import is used.

**Deployable artifact (2026-10-29):** `student_pt` `model_1999.pt` →
`logs/rsl_rl/orca_v2_repose/2026-10-28_22-01-33_student_pt/exported/policy_history.pt` (10.4 MB, sha256 in the JSON;
`logs/` is gitignored). Task `orca-v2-inhand-repose-student-pt-v0`, history 10 × 74 → 740, 17 actions.

## History semantics in Isaac Lab 2.3.2 (verified in the source)

Paths relative to `~/IsaacLab/source/isaaclab/isaaclab/` (`isaaclab_rl/...`: `~/IsaacLab/source/isaaclab_rl/`); line
numbers at v2.3.2 (`37ddf62`).

| What | Where | Finding |
|---|---|---|
| Per-term processing order | `managers/observation_manager.py:393-407` | func → modifiers → **noise** → clip → scale, then the history (409-412). The history stores the noisy, scaled per-step values. |
| Group overrides terms | `observation_manager.py:549-552` | A group `history_length` (not None) sets every term's `history_length` and `flatten_history_dim`. Our runs set it on the group (`OrcaV2Repose*Hist3/10EnvCfg`, `OrcaV2ReposeRMAStudentEnvCfg`: 3 / 10 / 30). |
| Noise switch | `observation_manager.py:547-548` | `enable_corruption=False` removes the term noise (the policy group trains with it on). |
| One buffer per term | `observation_manager.py:637-641` | `CircularBuffer(max_len=history_length, batch_size=num_envs)` per term. |
| Buffer order | `utils/buffers/circular_buffer.py:80-90` | `.buffer` is (N, H, D) with the **oldest** entry first and the newest last. |
| Flatten + concat | `observation_manager.py:423-424, 431-433` | Per term `buffer.reshape(N, -1)`, then `torch.cat` over the terms in cfg order. The flat input is **term-major**: `[joint_pos t-H+1..t, joint_vel t-H+1..t, goal_command …, last_action …, last_joint_pos_cmd …]`, each term oldest → newest. |
| First push after a reset | `circular_buffer.py:136-139` | If `num_pushes == 0`, the obs is written into **all H slots**: the history starts as H **copies of the first observation**, not zeros (also documented at `managers/manager_term_cfg.py:184-190`). |
| Reset | `circular_buffer.py:96-110`; `observation_manager.py:301-309` | `num_pushes = 0`; the slots are zeroed, but nobody reads them before the next append (next row). |
| When the policy sees it | `envs/manager_based_rl_env.py:216-238` | `step()`: terminated / timed-out envs → `_reset_idx` → `observation_manager.reset(env_ids)` (:369), then `compute(update_history=True)` (:238). The obs returned for a reset env is already the new episode's first obs, with the history filled with copies of it. The initial `reset()` works the same way (`envs/manager_based_env.py:372-386`). |
| No extra push | `isaaclab_rl/rsl_rl/vecenv_wrapper.py:143-149` | `get_observations()` calls `compute()` without `update_history`: it returns the stored history and does not shift it. |
| First obs of an episode | `managers/action_manager.py:364-365`; `envs/mdp/actions/joint_actions_to_limits.py:269-277` | After a reset, `last_action = 0` and `last_joint_pos_cmd` = the joint positions at reset (EMA action reset). This obs is the one copied H times. |

How the policies consume the flat history (repo code):

- History MLPs (`ActorCritic`, `StudentTeacher`, fine-tune variants): `mlp(normalizer(flat))`, with the normalizer
  over the whole flat vector (740 / 222).
- `StudentTeacherTransformer` (`transformer_student.py`): `student_obs_normalizer` on the flat 740 vector, then
  `_tokens` cuts the term-major vector into 10 per-step tokens of 74, then the transformer (`act_inference` = action
  head).
- `StudentTeacherRMA` (`rma.py`): `_tokens` first, then the frozen stage-1 74-dim `actor_obs_normalizer` per token,
  then `actor([tokens[:, -1], tanh(adaptation(tokens))])`.

## The exported module

`HistoryPolicy` (`scripts/export/export_history_policy.py`) holds a (B, H, 74) buffer plus `num_pushes` (B).

- `forward(obs)` takes obs `[B, 74]` (or `[74]`) and returns actions `[B, 17]` (or `[17]`). It does
  `CircularBuffer.append` for all terms at once: shift, append, and fill all H slots on the first push after a reset.
  Then it flattens term-major exactly like `compute_group`. Terms may have different history lengths, and terms with
  0 add only step t. Then it runs a scripted copy of the policy's `act_inference` path: MLP, transformer or RMA body,
  with the **trained normalizer at the same place as in training**. It adds no noise, and its parameters are frozen
  (same numerics with or without autograd).
- `reset()` / `reset(env_ids: LongTensor)` starts a new episode: the next `forward` fills the history.
  `flat_history()` returns the flat 740-dim input of the last `forward` without changing state (for debugging and
  logging).
- Misuse raises instead of silently misbehaving: wrong obs size, non-float32 obs, wrong device, or a batch-size change
  without `reset()` (the first call after load / `reset()` may use any batch size). It is safe under
  `torch.inference_mode()`: `reset()` works out of place, so calling `forward` in inference mode and `reset()` outside
  it also works (the test caught this in a first version).
- Attributes `history_length` and `obs_dim`. Loading needs nothing but torch: the test loads and runs it in a
  `python -I` interpreter and checks that no repo / rsl_rl / isaaclab / tensordict module gets imported.

## Deploy contract

1. **Input** = the 74-dim per-step obs exactly as the node already builds it for `policy.pt`: same term and joint
   order, `joint_pos` limit-normalized, `joint_vel × 0.2`, **no noise**, float32, shape `[1, 74]` (or `[74]`).
2. **Call `forward` exactly once per 50 ms control step.** Every call shifts the history. Do not call it for warm-up,
   logging or a second time in one tick (use `flat_history()` to inspect). Do not skip a tick: in sim every 50 ms
   step pushes one entry, so the history spacing must stay 50 ms.
3. **Call `reset()` when a new episode starts** (cube placed, policy (re)started), before that episode's first
   `forward`. That `forward` fills the history with copies of its obs, as in sim. A fresh `torch.jit.load` is
   equivalent to `reset()`. Ideally the first obs looks like the sim's post-reset obs (`last_action = 0`,
   `last_joint_pos_cmd` = current joint positions); this was already true for `policy.pt`.
4. If `reset()` is forgotten, the first H − 1 steps of the new episode see the previous episode's tail (in the test up
   to 2.6 action units off for `student_pt`). After H steps (0.5 s; RMA 1.5 s) it is identical again.
5. Node changes needed: load `policy_history.pt` (or copy it as `policy.pt` in our `rwr_faive_system` fork), call
   `policy.reset()` at episode start, and keep one call per tick. The JSON records the history length, obs layout
   (with training noise and scales for reference), action term (EMA α 0.2, rescale to limits), checkpoint and task.

### The real policy node (rwr_system, inspected 2026-10-31)

`rwr-fast/rwr_system` (commit f0f8cf6), `experiments/rl_inference/policy_inference_cube_spin_v2.py`:

- loads the file with `torch.jit.load(policy_ckpt_path)` (any path, so `policy_history.pt` works as is), calls it as
  `policy(obs[1, 74])` under `torch.no_grad()`, 17 outputs asserted;
- builds the 74-dim obs exactly as the task: limit-normalized joint_pos (USD soft limits, USD joint order),
  joint_vel × 0.2, goal_command `[0, 0, -0.05, 0, 0, goal_ang_vel]` (= the sim's command, checked in the eval
  rollouts: cmd_pos (0, 0, −0.05)), raw last action, last EMA joint command; actions clipped to [−1, 1], rescaled to
  the limits, EMA α 0.2, clipped to the limits; last_action = 0 and last command = measured pose on the first step;
- one call per 20 Hz timer tick (a tick without hand state or with a non-finite state is skipped);
- **one warm-up call `policy(torch.zeros(1, 74))` at load** and no `reset()`; no episode hook (one node start = one
  trial).

Consequences: memoryless MLP policies (all +z PPO policies) run unchanged. On the real hand (2026-10-31) the student's first steps stalled the loop 216–593 ms after a single warm-up call (TorchScript's profiling executor optimizing on the first real calls); **three warm-up calls, then `reset()`**, brought the longest loop step down to 74 ms. A history policy (student-pt, MLP-h10, LSTM)
needs `self.policy.reset()` right after the warm-up call: otherwise the zero observation stays in the history for the
first H − 1 steps (CPU test on `student_pt_rotlin`, random obs: up to 29 action units off in the first 10 steps,
identical afterwards; with `reset()` bit-identical). Restart the node per trial (or add a reset on a new trial).
`goal_ang_vel` defaults to 1.0 (the default task samples |w| in [0.9, 1.5] with either sign); the +z policies were
trained on exactly +1.5, so run them with `-p goal_ang_vel:=1.5` and `goal_flip_period_s` 0.

## Tests (CPU)

`/tmp/claude-322334/-zhome-b3-e-222334/e98b2254-0dd6-46ce-be5a-af9ce9954bee/scratchpad/exporttest/`
(`run_tests.sh` → `test_export_history.py`, deploy loop `run_module.py`, output `test_output.txt`; scratch dir, not in
the repo; the export script's self-check is the in-repo version).

- **Reference:** Isaac Lab's **real** `ObservationManager` (`observation_manager.py` with the real `CircularBuffer`) on
  a fake env. It is imported on the login node with Isaac Sim's bundled `pxr`, a stubbed `omni.timeline` and without
  the `isaaclab.managers` package `__init__`. The policy group is built from the run's `env.yaml`: group history, term
  order, `joint_vel` scale 0.2, noise off. A random but plausible 74-dim sequence runs over 4 envs, with resets at
  different steps per env (after one step, back-to-back, before the history is full, two envs at once; env 0 never
  resets, so the buffer wraps). Step order is as in `ManagerBasedRLEnv.step` (`reset(env_ids)` then
  `compute(update_history=True)`), and the original policy's `act_inference` runs on the flat obs.
- **Module side:** the single-step obs comes from the same manager (a second group without history). The `.pt` file
  is run in a clean `python -I` interpreter, batched with `reset(env_ids)` and at batch 1 per env with `reset()` (the
  deploy case), under grad, `no_grad` and `inference_mode`. Batch-1 outputs are compared with batch-1 reference
  outputs.

| Policy | Class | History | Max \|Δaction\| | Flat input bit-identical | Broken variants caught |
|---|---|---|---|---|---|
| `student_pt` (CLI artifact) | StudentTeacherTransformer | 10 | 0 | yes | all 4 |
| `arch_rma_s2` | StudentTeacherRMA | 30 | 0 | yes | all 4 |
| `arch_asym_hist10` / `hist3` | ActorCritic | 10 / 3 | 0 | yes | all 4 |
| `abl_B_asym_ent5e-3` | ActorCritic | none | 0 | yes | – |
| `student_mlp_h10` (random init) | StudentTeacher | 10 | 0 | yes | all 4 |
| `student_mlp_h10_ft` / `_res` (random init) | ActorCritic(Residual)Finetune | 10 | 0 | yes | all 4 |
| `arch_asym_lstm` | ActorCriticRecurrent (LSTM) | – | 1.8e-6 (batch 1, grad); else 0 | – | forgotten reset |
| `student_lstm` (random init) / 2-layer GRU (random init) | StudentTeacherRecurrent / ActorCriticRecurrent | – | ≤ 4e-8 | – | forgotten reset |

Broken variants on `student_pt` (max |Δaction|; all flat inputs differ): step-major instead of term-major 22.3,
newest-first 2.2, zero-filled history at reset 2.0, deploy forgets `reset()` 2.6. The other policies show 0.07–6.1.
Random-init models get randomized normalizer statistics, so normalizer placement matters in the test.

Numerics: with the same thread count the export is bit-identical to `act_inference`. float32 kernels depend on the
batch size and the OpenMP thread count (`OMP_NUM_THREADS=4` vs 1: up to 7.9e-6 for the transformer) and on the LSTM
grad / no-grad path (1.8e-6). These are rounding differences, not layout errors (the flat inputs stay bit-identical).
GPU training vs CPU deploy differ by at least this much anyway.

### Recurrent policies (LSTM, GRU)

Isaac Lab's exporter (play.py's `policy.pt`) gets the LSTM forward right but scripts **no `reset()`**.
`_TorchPolicyExporter.__init__` assigns `self.reset = self.reset_memory` as an instance attribute
(`isaaclab_rl/rsl_rl/exporter.py:73, 76`). This shadows the class's `@torch.jit.export reset` (:103-104) and is not
exported itself, so `torch.jit.script` (:116) compiles only `forward`. Calling `m.reset()` raises `AttributeError`.
On the robot the hidden state would carry over between episodes: in the test the first episode matches exactly, but
later episodes are off by up to 4.9 (`arch_asym_lstm`); the state is also batch 1 only. **Export `student_lstm` /
`arch_asym_lstm` with `export_history_policy.py`.** That gives `LSTMPolicy` / `GRUPolicy` with the same API, the state
kept per env, and `reset()` zeroing it like rsl_rl's `Memory.reset(dones)`.

## Verify in sim on a GPU

`scripts/export/verify_export_in_sim.py` runs the run's task in Isaac Sim (its own noise and DR). Episodes are
shortened to `--episode_s 4` and their start is staggered, so time-outs and drops reset envs at many different steps.
Every step it feeds the exported module (on CPU) **only** the newest per-step obs, sliced from the env's own flat
policy obs, and `reset(env_ids)` for the envs the env has just reset. It checks:

1. `flat_history()` equals the env's flat policy obs bit for bit on every step (layout, fill and reset timing against
   the real env, noise included);
2. actions agree with a CPU copy of the original policy (tolerance 1e-5) and, for information, with the runner's policy
   on the GPU (as in play / eval).

```bash
# interactive GPU node (e.g. a10 / l40s queue), repo root
source ~/rwr_faive_lab/activate_rwr.sh
python scripts/export/verify_export_in_sim.py --run logs/rsl_rl/orca_v2_repose/2026-10-28_22-01-33_student_pt \
    --headless --num_envs 64 --steps 1000 --episode_s 4 [--drive exported]
```

It prints `OK` / `FAILED` (exit code 0 / 1) and writes `<run>/exported/verify_in_sim.json`. The task and checkpoint
come from the export's JSON. `--drive exported` steps the env with the exported module's actions instead of the
original policy's.

### Result for `student_pt` (job 29845853, A10, 2026-10-29 05:50)

`hpc/verify_export_job.sh` ran `verify_export_in_sim.py` twice (64 envs × 1000 steps, 4 s episodes):

| `--drive` | env resets | flat history == env `policy` obs | max \|Δa\| exported vs CPU policy | vs runner policy on GPU |
|---|---|---|---|---|
| `policy` | 830 | yes, 0 mismatching steps | 0 (also 0 in the first 10 steps after a reset) | 3.2e-2 |
| `exported` | 841 | yes, 0 mismatching steps | 0 | 3.0e-2 |

The history the module rebuilds from single-step observations is identical to what the live env feeds the policy,
through ~830 resets, so layout, fill-at-reset and term order are right. The exported module reproduces the policy
bit for bit on CPU. The remaining 3e-2 (max over ~1M action values) is the same transformer computed on GPU vs CPU;
the inputs are identical, so this is arithmetic, not the export. It is below the student's own action noise during
distillation (std 0.1), and it is what any CPU deployment of this network will see.

## Files

| File | Contents |
|---|---|
| `scripts/export/export_history_policy.py` | CLI + library: run dir → policy (`load_run`), `HistoryPolicy` / `LSTMPolicy` / `GRUPolicy`, export, self-check, reference rollout with Isaac Lab's CircularBuffer |
| `scripts/export/verify_export_in_sim.py` | in-sim check of an export (GPU, Isaac Sim) |
