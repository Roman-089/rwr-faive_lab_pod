"""Export a history-based rsl_rl policy as a self-contained TorchScript module for the real robot.

The policy node on the robot calls the exported module once per control step (20 Hz) with the CURRENT 74-dim policy
observation only. Policies trained on an observation history (Isaac Lab's ``ObservationManager`` keeps one
``CircularBuffer`` per term and flattens it term-major) therefore need that history inside the exported module. This
script rebuilds the policy from a run directory without Isaac Sim (params/agent.yaml, params/env.yaml, model_X.pt),
wraps it in ``HistoryPolicy`` and writes

    <run>/exported/policy_history.pt    TorchScript: forward(obs[B, 74]) -> actions[B, 17], reset(env_ids=None)
    <run>/exported/policy_history.json  history length, obs layout, checkpoint, task id (also inside the .pt)

play.py's single-step export (exported/policy.pt) is never touched. History semantics and the deploy contract:
docs/ablation/export.md.

Supported (``policy.class_name`` in agent.yaml): ActorCritic and StudentTeacher MLPs on a history (asym-hist3 /
hist10, student-mlp-h10, the fine-tuned students), StudentTeacherTransformer (student-pt), StudentTeacherRMA
(rma-student), ActorCriticROA (servofast-roa: history-encoder latent). A policy without history exports too (the
module is then stateless). Recurrent policies (ActorCriticRecurrent, StudentTeacherRecurrent: asym-lstm, student-lstm)
export as LSTM- / GRUPolicy with the same API,
because play.py's export (Isaac Lab exporter) has no working reset() for them.

After writing, the file is re-loaded and checked against the original policy's ``act_inference`` on a random
observation sequence with resets, the history being built with Isaac Lab's CircularBuffer as the ObservationManager
does it (``--no_check`` skips this).

Usage (CPU, no Isaac Sim):
    python scripts/export/export_history_policy.py --run logs/rsl_rl/orca_v2_repose/<run> [--checkpoint model_X.pt]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import types
from typing import List, Optional

import torch
import torch.nn as nn
import yaml
from tensordict import TensorDict

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TASK_DIR = os.path.join(REPO_DIR, "source", "faive_lab", "faive_lab", "tasks", "orca_inhand")

# Per-step size of each policy term (orca_v2_repose_env_cfg.KinematicObsGroupCfg; transformer_student.POLICY_TERM_DIMS)
TERM_DIMS = {"joint_pos": 17, "joint_vel": 17, "goal_command": 6, "last_action": 17, "last_joint_pos_cmd": 17}
# ObservationGroupCfg fields that are not terms (ObservationManager._prepare_terms skips the same names)
GROUP_FIELDS = ("enable_corruption", "concatenate_terms", "history_length", "flatten_history_dim", "concatenate_dim")
# policy classes defined in this repo (module file in TASK_DIR)
TASK_POLICY_MODULES = {
    "StudentTeacherTransformer": "transformer_student",
    "StudentTeacherRMA": "rma",
    "ActorCriticROA": "roa",
    "ActorCriticFinetune": "student_finetune",
    "ActorCriticResidualFinetune": "student_finetune",
}
# state-dict keys whose input size gives the size of a non-policy observation set (only needed to build the module)
SET_DIM_KEYS = {
    "critic": ("critic_obs_normalizer._mean", "critic.0.weight"),
    "teacher": (
        "teacher_obs_normalizer._mean", "teacher.0.weight", "teacher_actor.0.weight",
        "priv_obs_normalizer._mean", "env_mlp.0.weight",  # StudentTeacherRMA: teacher set = RMA "priv" group
    ),
    "priv": ("priv_encoder.0.weight",),  # ActorCriticROA: privileged encoder input (not used by the export)
}
OUTPUT_NAME = "policy_history"


# ---------------------------------------------------------------------------------------------------------------------
# Run directory -> observation layout and the original rsl_rl policy
# ---------------------------------------------------------------------------------------------------------------------


class _ParamsLoader(yaml.SafeLoader):
    """SafeLoader that reads the python tags Isaac Lab writes into params/*.yaml (tuples, slices) as plain data."""


def _construct_python_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def load_params(run_dir: str) -> tuple[dict, dict]:
    """(agent cfg, env cfg) as saved by the training script in <run>/params."""
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        agent_cfg = yaml.load(f, Loader=_ParamsLoader)
    with open(os.path.join(run_dir, "params", "env.yaml")) as f:
        env_cfg = yaml.load(f, Loader=_ParamsLoader)
    return agent_cfg, env_cfg


def policy_obs_layout(env_cfg: dict, group: str = "policy") -> dict:
    """Per-term layout of an observation group, resolved like ObservationManager._prepare_terms.

    A group-level ``history_length`` (not None) overrides every term's ``history_length`` / ``flatten_history_dim``
    (observation_manager.py: "check group history params and override terms"); otherwise each term keeps its own.
    """
    g = env_cfg["observations"][group]
    if not g.get("concatenate_terms", True):
        raise NotImplementedError(f"group '{group}' is not concatenated")
    if g.get("concatenate_dim", -1) not in (-1, 0):
        raise NotImplementedError(f"group '{group}' concatenates along dim {g['concatenate_dim']}")
    group_hist = g.get("history_length")
    terms, start = [], 0
    for name, t in g.items():
        if name in GROUP_FIELDS or t is None:
            continue
        if name not in TERM_DIMS:
            raise NotImplementedError(f"unknown policy term '{name}' (known: {list(TERM_DIMS)}); add it to TERM_DIMS")
        if t.get("modifiers"):
            raise NotImplementedError(f"term '{name}' has modifiers; the deploy obs would have to include them")
        hist = int(group_hist if group_hist is not None else (t.get("history_length") or 0))
        flatten = g.get("flatten_history_dim", True) if group_hist is not None else t.get("flatten_history_dim", True)
        if hist > 0 and not flatten:
            raise NotImplementedError(f"term '{name}' keeps an unflattened history")
        terms.append({
            "name": name,
            "start": start,
            "dim": TERM_DIMS[name],
            "history_length": hist,
            "func": t.get("func"),
            "params": t.get("params") or {},
            "clip": t.get("clip"),
            "scale": t.get("scale"),
            "train_noise": t.get("noise") if g.get("enable_corruption", False) else None,
        })
        start += TERM_DIMS[name]
    return {
        "group": group,
        "terms": terms,
        "obs_dim": start,
        "history_length": max(t["history_length"] for t in terms),
        "flat_dim": sum(t["dim"] * max(t["history_length"], 1) for t in terms),
    }


@contextlib.contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]):
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _import_task_module(name: str) -> types.ModuleType:
    """Import a policy module of this repo (transformer_student.py, rma.py, ...).

    Inside Isaac Sim (faive_lab.tasks already imported) this is the normal import. Standalone, the file is loaded under
    a private package name so that the package __init__ (gym registration, Isaac Sim imports) does not run; the cfg
    classes at the bottom of these files only need ``configclass`` and the isaaclab_rl cfg base classes, which are
    stubbed while the file executes (they are not used for export). The policy classes are the repo's own code.
    """
    if "faive_lab.tasks.orca_inhand" in sys.modules:
        return importlib.import_module(f"faive_lab.tasks.orca_inhand.{name}")
    pkg_name = "_faive_orca_inhand_export"
    full = f"{pkg_name}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [TASK_DIR]
        sys.modules[pkg_name] = pkg

    utils_stub = types.ModuleType("isaaclab.utils")
    utils_stub.configclass = lambda cls: cls
    rl_stub = types.ModuleType("isaaclab_rl.rsl_rl")
    rl_stub.__getattr__ = lambda attr: type(attr, (), {})
    rl_pkg_stub = types.ModuleType("isaaclab_rl")
    rl_pkg_stub.__path__ = []
    rl_pkg_stub.rsl_rl = rl_stub
    stubs = {"isaaclab.utils": utils_stub, "isaaclab_rl": rl_pkg_stub, "isaaclab_rl.rsl_rl": rl_stub}
    if importlib.util.find_spec("isaaclab") is None:
        isaaclab_stub = types.ModuleType("isaaclab")
        isaaclab_stub.__path__ = []
        stubs["isaaclab"] = isaaclab_stub
    with _temporary_modules(stubs):
        spec = importlib.util.spec_from_file_location(full, os.path.join(TASK_DIR, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(full, None)
            raise
    return module


def policy_class(class_name: str) -> type:
    if class_name in ("ActorCriticRMA",):
        raise NotImplementedError("RMA stage 1 needs the privileged 'priv' obs: not deployable, export the stage-2 run")
    if class_name in TASK_POLICY_MODULES:
        return getattr(_import_task_module(TASK_POLICY_MODULES[class_name]), class_name)
    import rsl_rl.modules

    if not hasattr(rsl_rl.modules, class_name):
        raise NotImplementedError(f"unknown policy class '{class_name}'")
    return getattr(rsl_rl.modules, class_name)


def _set_input_dim(state_dict: dict, obs_set: str) -> int:
    for key in SET_DIM_KEYS.get(obs_set, ()):
        if key in state_dict:
            return int(state_dict[key].shape[-1])
    raise ValueError(f"cannot infer the size of the '{obs_set}' observation set from the checkpoint")


def build_policy(agent_cfg: dict, layout: dict, state_dict: dict, verbose: bool = False) -> nn.Module:
    """Construct the rsl_rl policy module of a run and load its weights (strict), like the runner does.

    The policy set gets the real flat history size from env.yaml; the other sets (critic / teacher), which the export
    does not use, get their sizes from the checkpoint.
    """
    pcfg = dict(agent_cfg["policy"])
    cls = policy_class(pcfg.pop("class_name"))
    obs_groups = agent_cfg.get("obs_groups") or {}
    if obs_groups.get("policy", ["policy"]) != [layout["group"]]:
        raise NotImplementedError(f"policy obs set {obs_groups.get('policy')} is not the one group '{layout['group']}'")
    default_sets = ["teacher"] if agent_cfg.get("class_name") == "DistillationRunner" else ["critic"]
    sets = {"policy": [layout["group"]]}
    dims = {layout["group"]: layout["flat_dim"]}
    for obs_set in sorted(set(default_sets) | set(obs_groups) - {"policy"}):
        sets[obs_set] = [f"_{obs_set}"]
        dims[f"_{obs_set}"] = _set_input_dim(state_dict, obs_set)
    obs = TensorDict({k: torch.zeros(1, d) for k, d in dims.items()}, batch_size=[1])
    std = state_dict.get("log_std", state_dict.get("std"))
    num_actions = int(std.shape[0]) if std is not None else TERM_DIMS["last_action"]
    with contextlib.redirect_stdout(sys.stdout if verbose else io.StringIO()):
        policy = cls(obs, sets, num_actions, **pcfg)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()  # StudentTeacher.train() returns None: do not chain
    policy.export_obs_dims = dims  # group sizes, for dummy inputs of the non-policy sets in reference_rollout
    return policy


def latest_checkpoint(run_dir: str) -> str:
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return max(ckpts, key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))


def task_id_of_run(run_dir: str) -> str | None:
    """Task id from the run's eval/metrics.json (written by scripts/eval/eval_policy.py), if any."""
    path = os.path.join(run_dir, "eval", "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            for entry in json.load(f).values():
                if isinstance(entry, dict) and entry.get("task"):
                    return entry["task"]
    return None


def load_run(run_dir: str, checkpoint: str | None = None, verbose: bool = False) -> tuple[nn.Module, dict, dict]:
    """(policy, layout, info) of a run directory."""
    run_dir = os.path.abspath(run_dir)
    agent_cfg, env_cfg = load_params(run_dir)
    ckpt_path = latest_checkpoint(run_dir) if checkpoint is None else (
        checkpoint if os.path.isabs(checkpoint) else os.path.join(run_dir, checkpoint))
    layout = policy_obs_layout(env_cfg, (agent_cfg.get("obs_groups") or {}).get("policy", ["policy"])[0])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    policy = build_policy(agent_cfg, layout, state_dict, verbose=verbose)
    info = {
        "run": run_dir,
        "checkpoint": ckpt_path,
        "policy_class": agent_cfg["policy"]["class_name"],
        "control_dt": float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"]),
        "action": {k: v for k, v in (env_cfg.get("actions") or {}).get("joint_pos", {}).items()
                   if k in ("class_type", "alpha", "rescale_to_limits", "joint_names")},
    }
    return policy, layout, info


# ---------------------------------------------------------------------------------------------------------------------
# Exported module
# ---------------------------------------------------------------------------------------------------------------------


def term_major_to_tokens(flat: torch.Tensor, history_length: int, term_dims: List[int]) -> torch.Tensor:
    """(B, H * sum(d)) term-major flat history -> (B, H, sum(d)) per-step tokens (transformer_student._tokens)."""
    parts: List[torch.Tensor] = []
    i = 0
    for d in term_dims:
        parts.append(flat[:, i:i + history_length * d].reshape(-1, history_length, d))
        i += history_length * d
    return torch.cat(parts, dim=-1)


class MlpBody(nn.Module):
    """ActorCritic / StudentTeacher act_inference: mlp(normalizer(flat history))."""

    def __init__(self, normalizer: nn.Module, mlp: nn.Module, state_dependent_std: bool = False):
        super().__init__()
        self.normalizer = normalizer
        self.mlp = mlp
        self.state_dependent_std = state_dependent_std

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.normalizer(flat))
        if self.state_dependent_std:
            out = out[..., 0, :]
        return out


class TransformerBody(nn.Module):
    """StudentTeacherTransformer.act_inference: student(_tokens(student_obs_normalizer(flat history)))[0]."""

    def __init__(self, normalizer: nn.Module, student: nn.Module, history_length: int, term_dims: List[int]):
        super().__init__()
        self.normalizer = normalizer
        self.student = student
        self.history_length = history_length
        self.term_dims = term_dims

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        tokens = term_major_to_tokens(self.normalizer(flat), self.history_length, self.term_dims)
        return self.student(tokens)[0]


class RMABody(nn.Module):
    """StudentTeacherRMA.act_inference: tokens = norm(_tokens(flat)); actor([tokens[:, -1], tanh(student(tokens))])."""

    def __init__(self, normalizer: nn.Module, adaptation: nn.Module, actor: nn.Module, history_length: int,
                 term_dims: List[int]):
        super().__init__()
        self.normalizer = normalizer
        self.adaptation = adaptation
        self.actor = actor
        self.history_length = history_length
        self.term_dims = term_dims

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        tokens = self.normalizer(term_major_to_tokens(flat, self.history_length, self.term_dims))
        z = torch.tanh(self.adaptation(tokens))
        return self.actor(torch.cat([tokens[:, -1], z], dim=-1))


class ROABody(nn.Module):
    """ActorCriticROA.act_inference: tokens = norm(_tokens(flat)); actor([tokens[:, -1], history_encoder(tokens)])."""

    def __init__(self, normalizer: nn.Module, history_encoder: nn.Module, actor: nn.Module, history_length: int,
                 term_dims: List[int]):
        super().__init__()
        self.normalizer = normalizer
        self.history_encoder = history_encoder
        self.actor = actor
        self.history_length = history_length
        self.term_dims = term_dims

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        tokens = self.normalizer(term_major_to_tokens(flat, self.history_length, self.term_dims))
        return self.actor(torch.cat([tokens[:, -1], self.history_encoder(tokens)], dim=-1))


def make_body(policy: nn.Module, layout: dict) -> nn.Module:
    """Scriptable copy of the policy's inference path on the flat (term-major) history."""
    if getattr(policy, "is_recurrent", False):
        raise NotImplementedError("recurrent policy: use make_recurrent()")
    term_dims = [t["dim"] for t in layout["terms"]]
    hists = {t["history_length"] for t in layout["terms"]}
    name = type(policy).__name__
    if name in ("StudentTeacherTransformer", "StudentTeacherRMA", "ActorCriticROA"):
        module = sys.modules[type(policy).__module__]
        if len(hists) != 1 or hists != {policy.history_length}:
            raise ValueError(f"{name}: env history {hists} != policy history_length {policy.history_length}")
        if tuple(term_dims) != tuple(module.POLICY_TERM_DIMS):
            raise ValueError(f"{name}: env term dims {term_dims} != POLICY_TERM_DIMS {module.POLICY_TERM_DIMS}")
        if name == "StudentTeacherTransformer":
            body = TransformerBody(copy.deepcopy(policy.student_obs_normalizer), copy.deepcopy(policy.student),
                                   policy.history_length, term_dims)
        elif name == "ActorCriticROA":
            body = ROABody(copy.deepcopy(policy.actor_obs_normalizer), copy.deepcopy(policy.history_encoder),
                           copy.deepcopy(policy.actor), policy.history_length, term_dims)
        else:
            body = RMABody(copy.deepcopy(policy.actor_obs_normalizer), copy.deepcopy(policy.student),
                           copy.deepcopy(policy.actor), policy.history_length, term_dims)
    elif hasattr(policy, "student") and hasattr(policy, "student_obs_normalizer"):  # StudentTeacher
        body = MlpBody(copy.deepcopy(policy.student_obs_normalizer), copy.deepcopy(policy.student))
    elif hasattr(policy, "actor") and hasattr(policy, "actor_obs_normalizer"):  # ActorCritic (+ fine-tune variants)
        if type(policy).act_inference is not _actor_critic_act_inference():
            raise NotImplementedError(f"{name} overrides act_inference; add a body for it")
        body = MlpBody(copy.deepcopy(policy.actor_obs_normalizer), copy.deepcopy(policy.actor),
                       bool(getattr(policy, "state_dependent_std", False)))
    else:
        raise NotImplementedError(f"unsupported policy class {name}")
    body.eval()
    body.requires_grad_(False)  # same numerics with and without autograd on the deploy side
    return body


def _actor_critic_act_inference():
    from rsl_rl.modules import ActorCritic

    return ActorCritic.act_inference


class HistoryPolicy(nn.Module):
    """Single-step observation in, actions out; keeps the observation history the policy was trained on.

    History semantics = Isaac Lab ObservationManager + CircularBuffer (see docs/ablation/export.md):
      - per term a buffer of its last ``history_length`` per-step values, oldest first; the flat policy input is
        [term_1 steps t-H+1..t, term_2 steps t-H+1..t, ...] (term-major), terms with history_length 0 add only step t;
      - the first observation after a reset fills the whole history with copies of itself.
    ``reset()`` / ``reset(env_ids)`` marks environments as new episodes; their next ``forward`` fills the history.
    """

    def __init__(self, body: nn.Module, layout: dict):
        super().__init__()
        self.body = body
        self.obs_dim = int(layout["obs_dim"])
        self.history_length = max(int(layout["history_length"]), 1)
        self.term_starts = [int(t["start"]) for t in layout["terms"]]
        self.term_dims = [int(t["dim"]) for t in layout["terms"]]
        self.term_history = [int(t["history_length"]) for t in layout["terms"]]
        # (B, H, obs_dim), oldest step first; num_pushes == 0 marks a new episode (as in CircularBuffer)
        self.register_buffer("history", torch.zeros(1, self.history_length, self.obs_dim))
        self.register_buffer("num_pushes", torch.zeros(1, dtype=torch.long))

    @torch.jit.export
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Start a new episode (all envs, or the given env indices): the next forward() fills the history."""
        # out of place: the buffers may be inference tensors (forward under torch.inference_mode), which cannot be
        # modified in place outside inference mode
        if env_ids is None:
            self.num_pushes = torch.zeros_like(self.num_pushes)
            self.history = torch.zeros_like(self.history)
        else:
            num_pushes = self.num_pushes.clone()
            history = self.history.clone()
            num_pushes[env_ids] = 0
            history[env_ids] = 0.0
            self.num_pushes = num_pushes
            self.history = history

    def _push(self, obs: torch.Tensor) -> None:
        """CircularBuffer.append for all terms at once: shift, append, fill on the first push after a reset."""
        new = obs.unsqueeze(1)
        hist = torch.cat([self.history[:, 1:], new], dim=1)
        first = (self.num_pushes == 0).view(-1, 1, 1)
        self.history = torch.where(first, new.expand(-1, self.history_length, -1), hist)
        self.num_pushes = torch.clamp(self.num_pushes + 1, max=self.history_length)

    def _flatten(self, hist: torch.Tensor) -> torch.Tensor:
        """ObservationManager.compute_group: per term buffer.reshape(B, -1), then torch.cat over terms."""
        parts: List[torch.Tensor] = []
        for i in range(len(self.term_dims)):
            a = self.term_starts[i]
            d = self.term_dims[i]
            h = self.term_history[i]
            if h == 0:
                parts.append(hist[:, -1, a:a + d])
            else:
                parts.append(hist[:, self.history_length - h:, a:a + d].reshape(hist.shape[0], h * d))
        return torch.cat(parts, dim=-1)

    @torch.jit.export
    def flat_history(self) -> torch.Tensor:
        """The flat policy input of the last forward() (= the env's policy obs during training)."""
        return self._flatten(self.history)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, obs_dim] -> actions [B, num_actions]; a single obs [obs_dim] -> actions [num_actions] (batch 1)."""
        if obs.dim() == 1:
            return self._step(obs.unsqueeze(0)).squeeze(0)
        return self._step(obs)

    def _step(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[1] != self.obs_dim:
            raise RuntimeError("expected obs of shape [B, " + str(self.obs_dim) + "], got " + str(obs.shape))
        if obs.dtype != self.history.dtype:
            raise RuntimeError("expected obs dtype " + str(self.history.dtype) + ", got " + str(obs.dtype))
        if obs.device != self.history.device:
            raise RuntimeError("obs on " + str(obs.device) + ", policy on " + str(self.history.device))
        if obs.shape[0] != self.history.shape[0]:
            if bool((self.num_pushes > 0).any()):
                raise RuntimeError("batch size changed from " + str(self.history.shape[0]) + " to "
                                   + str(obs.shape[0]) + " without reset()")
            self.history = torch.zeros(obs.shape[0], self.history_length, self.obs_dim, dtype=obs.dtype,
                                       device=obs.device)
            self.num_pushes = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        self._push(obs)
        return self.body(self._flatten(self.history))


class _RecurrentPolicy(nn.Module):
    """rsl_rl recurrent act_inference, mlp(rnn(normalizer(obs))), with the RNN state kept per env.

    Same API as HistoryPolicy. reset() / reset(env_ids) zeroes the state like rsl_rl's Memory.reset(dones); the
    initial state is zero (rsl_rl: hidden_state None). Isaac Lab's stock exporter (play.py) scripts no reset() for
    recurrent policies (docs/ablation/export.md).
    """

    def __init__(self, normalizer: nn.Module, rnn: nn.Module, mlp: nn.Module, obs_dim: int, state_dependent_std: bool):
        super().__init__()
        self.normalizer = normalizer
        self.rnn = rnn
        self.mlp = mlp
        self.obs_dim = int(obs_dim)
        self.history_length = 0
        self.state_dependent_std = state_dependent_std
        self.register_buffer("hidden_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))  # LSTM only
        self.register_buffer("num_pushes", torch.zeros(1, dtype=torch.long))

    @torch.jit.export
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Start a new episode (all envs, or the given env indices): zero their RNN state."""
        if env_ids is None:
            self.hidden_state = torch.zeros_like(self.hidden_state)
            self.cell_state = torch.zeros_like(self.cell_state)
            self.num_pushes = torch.zeros_like(self.num_pushes)
        else:
            hidden_state = self.hidden_state.clone()
            cell_state = self.cell_state.clone()
            num_pushes = self.num_pushes.clone()
            hidden_state[:, env_ids] = 0.0
            cell_state[:, env_ids] = 0.0
            num_pushes[env_ids] = 0
            self.hidden_state = hidden_state
            self.cell_state = cell_state
            self.num_pushes = num_pushes

    def _rnn(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, obs_dim] -> actions [B, num_actions]; a single obs [obs_dim] -> actions [num_actions] (batch 1)."""
        if obs.dim() == 1:
            return self._step(obs.unsqueeze(0)).squeeze(0)
        return self._step(obs)

    def _step(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[1] != self.obs_dim:
            raise RuntimeError("expected obs of shape [B, " + str(self.obs_dim) + "], got " + str(obs.shape))
        if obs.dtype != self.hidden_state.dtype:
            raise RuntimeError("expected obs dtype " + str(self.hidden_state.dtype) + ", got " + str(obs.dtype))
        if obs.device != self.hidden_state.device:
            raise RuntimeError("obs on " + str(obs.device) + ", policy on " + str(self.hidden_state.device))
        if obs.shape[0] != self.hidden_state.shape[1]:
            if bool((self.num_pushes > 0).any()):
                raise RuntimeError("batch size changed from " + str(self.hidden_state.shape[1]) + " to "
                                   + str(obs.shape[0]) + " without reset()")
            shape = [self.hidden_state.shape[0], obs.shape[0], self.hidden_state.shape[2]]
            self.hidden_state = torch.zeros(shape, dtype=obs.dtype, device=obs.device)
            self.cell_state = torch.zeros(shape, dtype=obs.dtype, device=obs.device)
            self.num_pushes = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        out = self._rnn(self.normalizer(obs).unsqueeze(0)).squeeze(0)
        self.num_pushes = torch.clamp(self.num_pushes + 1, max=1)
        out = self.mlp(out)
        if self.state_dependent_std:
            out = out[..., 0, :]
        return out


class LSTMPolicy(_RecurrentPolicy):
    def _rnn(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.rnn(x, (self.hidden_state, self.cell_state))
        self.hidden_state = h
        self.cell_state = c
        return out


class GRUPolicy(_RecurrentPolicy):
    def _rnn(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.rnn(x, self.hidden_state)
        self.hidden_state = h
        return out


def make_recurrent(policy: nn.Module, layout: dict) -> nn.Module:
    """ActorCriticRecurrent (memory_a, actor) / StudentTeacherRecurrent (memory_s, student) -> LSTM- / GRUPolicy."""
    if layout["history_length"]:
        raise NotImplementedError("recurrent policy on an observation history")
    if hasattr(policy, "memory_a"):
        memory, mlp, normalizer = policy.memory_a, policy.actor, policy.actor_obs_normalizer
    else:
        memory, mlp, normalizer = policy.memory_s, policy.student, policy.student_obs_normalizer
    cls = LSTMPolicy if isinstance(memory.rnn, nn.LSTM) else GRUPolicy
    module = cls(copy.deepcopy(normalizer), copy.deepcopy(memory.rnn), copy.deepcopy(mlp), layout["obs_dim"],
                 bool(getattr(policy, "state_dependent_std", False)))
    module.eval()
    module.requires_grad_(False)
    return module


def make_module(policy: nn.Module, layout: dict) -> nn.Module:
    """The (unscripted) deploy module of a policy."""
    if getattr(policy, "is_recurrent", False):
        return make_recurrent(policy, layout)
    return HistoryPolicy(make_body(policy, layout), layout)


def export_metadata(layout: dict, info: dict, task: str | None) -> dict:
    return {
        "format": "rwr_faive_lab history policy v1",
        "file": None,  # filled in by export()
        "task": task,
        "run": os.path.basename(info["run"].rstrip("/")),
        "checkpoint": info["checkpoint"],
        "policy_class": info["policy_class"],
        "history_length": layout["history_length"],
        "obs_dim": layout["obs_dim"],
        "flat_obs_dim": layout["flat_dim"],
        "action_dim": None,  # filled in by export()
        "control_dt": info["control_dt"],
        "obs_layout": [
            {k: t[k] for k in ("name", "start", "dim", "history_length", "func", "params", "clip", "scale",
                               "train_noise")}
            for t in layout["terms"]
        ],
        "action_term": info.get("action"),
        "api": {
            "forward": "actions[B, action_dim] = policy(obs[B, obs_dim]) (or [obs_dim] -> [action_dim]); float32; "
                       "updates the history: call exactly once per control step",
            "reset": "policy.reset() (all) or policy.reset(env_ids: LongTensor) when a new episode starts; the next "
                     "forward fills the history with copies of its obs (Isaac Lab CircularBuffer semantics)",
            "flat_history": "policy.flat_history(): the flat term-major policy input of the last forward (debugging)",
            "obs": "the per-step policy observation as the env computes it before the history (term func -> clip -> "
                   "scale, e.g. joint_vel * 0.2), WITHOUT the training noise",
        },
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }


def export(policy: nn.Module, layout: dict, path: str, metadata: dict | None = None) -> torch.jit.ScriptModule:
    """Script ``make_module(policy)`` and save it (metadata JSON also embedded as an extra file)."""
    module = make_module(policy, layout).cpu().eval()
    if metadata is not None and isinstance(module, _RecurrentPolicy):
        metadata["recurrent"] = {"rnn_type": type(module.rnn).__name__.lower(), "hidden_dim": module.rnn.hidden_size,
                                 "num_layers": module.rnn.num_layers}
        metadata["api"]["reset"] = ("policy.reset() (all) or policy.reset(env_ids: LongTensor) when a new episode "
                                    "starts: zeroes the RNN state (rsl_rl Memory.reset)")
        del metadata["api"]["flat_history"]
    scripted = torch.jit.script(module)
    scripted.reset()
    extra = {}
    if metadata is not None:
        metadata["file"] = os.path.basename(path)
        metadata["action_dim"] = int(scripted(torch.zeros(1, layout["obs_dim"])).shape[-1])
        scripted.reset()
        extra[f"{OUTPUT_NAME}.json"] = json.dumps(metadata, indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.jit.save(scripted, path, _extra_files=extra)
    return scripted


# ---------------------------------------------------------------------------------------------------------------------
# Reference (training-time) inputs and the self-check
# ---------------------------------------------------------------------------------------------------------------------


def load_circular_buffer() -> type | None:
    """Isaac Lab's CircularBuffer (isaaclab/utils/buffers/circular_buffer.py only needs torch), or None."""
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = os.path.join(spec.submodule_search_locations[0], "utils", "buffers", "circular_buffer.py")
    if not os.path.exists(path):
        return None
    mspec = importlib.util.spec_from_file_location("_isaaclab_circular_buffer", path)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    return module.CircularBuffer


def random_obs_sequence(layout: dict, num_steps: int, num_envs: int, resets: dict[int, list[int]], seed: int = 0,
                        dt: float = 0.05) -> torch.Tensor:
    """Plausible (T, N, obs_dim) policy observations; every env starts a new episode (new parameters) at its resets.

    joint_pos / last_action in [-1, 1] (normalized), joint_vel = 0.2 * rad/s, goal_command = cube position target +
    angular velocity command (+-1.5 rad/s about z), last_joint_pos_cmd in rad; smooth sinusoids plus small noise.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.rand(*shape, generator=gen)

    obs = torch.zeros(num_steps, num_envs, layout["obs_dim"])
    for e in range(num_envs):
        starts = sorted({0} | {k for k, ids in resets.items() if e in ids})
        for s, s_next in zip(starts, starts[1:] + [num_steps]):
            t = torch.arange(s_next - s, dtype=torch.float32).unsqueeze(1) * dt
            for term in layout["terms"]:
                a, d = term["start"], term["dim"]
                freq, phase = 0.2 + 1.5 * rnd(d), 2 * math.pi * rnd(d)
                wave = torch.sin(2 * math.pi * freq * t + phase)
                noise = 0.02 * torch.randn(len(t), d, generator=gen)
                if term["name"] == "joint_pos":
                    val = (0.6 * (2 * rnd(d) - 1) + 0.3 * wave + noise).clamp(-1, 1)
                elif term["name"] == "joint_vel":
                    val = 0.2 * (2 * math.pi * freq * 0.3 * torch.cos(2 * math.pi * freq * t + phase)) + noise
                elif term["name"] == "goal_command":
                    sign = 1.0 if rnd(1).item() < 0.5 else -1.0
                    cmd = torch.tensor([0.0, 0.0, -0.05, 0.0, 0.0, 1.5 * sign])
                    val = cmd + torch.cat([0.005 * torch.randn(len(t), 3, generator=gen), torch.zeros(len(t), 3)], 1)
                elif term["name"] == "last_action":
                    val = torch.tanh(0.5 * (2 * rnd(d) - 1) + 0.8 * wave + 5 * noise)
                elif term["name"] == "last_joint_pos_cmd":
                    val = 0.8 * (2 * rnd(d) - 1) + 0.4 * wave
                else:
                    val = wave + noise
                obs[s:s_next, e, a:a + d] = val
    return obs


def current_obs(flat: torch.Tensor, layout: dict) -> torch.Tensor:
    """Newest step of each term of a flat term-major policy obs -> the (B, obs_dim) single-step obs."""
    parts, i = [], 0
    for t in layout["terms"]:
        h, d = max(t["history_length"], 1), t["dim"]
        parts.append(flat[:, i + (h - 1) * d:i + h * d])
        i += h * d
    return torch.cat(parts, dim=-1)


def policy_inputs(policy: nn.Module, layout: dict, flat: torch.Tensor) -> TensorDict:
    """act_inference input: the flat policy obs + zeros for the other sets (they do not enter the actions; RMA computes
    its unused teacher latent anyway)."""
    obs = {g: torch.zeros(flat.shape[0], d, device=flat.device)
           for g, d in getattr(policy, "export_obs_dims", {}).items() if g != layout["group"]}
    obs[layout["group"]] = flat
    return TensorDict(obs, batch_size=[flat.shape[0]], device=flat.device)


def reference_rollout(policy: nn.Module, layout: dict, obs_seq: torch.Tensor, resets: dict[int, list[int]],
                      buffer_cls: type) -> tuple[torch.Tensor, torch.Tensor]:
    """(actions, flat policy obs) of the original policy, its input built like ObservationManager.compute_group.

    Mirrors ManagerBasedRLEnv.step: reset the buffers of done envs, then compute(update_history=True) on the new obs.
    Recurrent policies get rsl_rl's reset(dones) before the new episode's first obs (runner / play.py order).
    """
    num_steps, num_envs, _ = obs_seq.shape
    bufs = {t["name"]: buffer_cls(t["history_length"], num_envs, "cpu") for t in layout["terms"] if t["history_length"]}
    recurrent = getattr(policy, "is_recurrent", False)
    if recurrent:
        policy.reset()
    actions, flats = [], []
    for k in range(num_steps):
        if resets.get(k):
            for b in bufs.values():
                b.reset(batch_ids=resets[k])
            if recurrent:
                dones = torch.zeros(num_envs, dtype=torch.long)
                dones[resets[k]] = 1
                policy.reset(dones)
        parts = []
        for t in layout["terms"]:
            x = obs_seq[k, :, t["start"]:t["start"] + t["dim"]].clone()
            if t["history_length"]:
                bufs[t["name"]].append(x)
                parts.append(bufs[t["name"]].buffer.reshape(num_envs, -1))
            else:
                parts.append(x)
        flat = torch.cat(parts, dim=-1)
        with torch.no_grad():
            actions.append(policy.act_inference(policy_inputs(policy, layout, flat)))
        flats.append(flat)
    return torch.stack(actions), torch.stack(flats)


def module_rollout(module, obs_seq: torch.Tensor, resets: dict[int, list[int]], batched: bool = True,
                   call_reset: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(actions, flat history) of an exported module fed one step at a time (batched with reset(env_ids), or one env
    at a time with reset())."""
    num_steps, num_envs, _ = obs_seq.shape
    has_flat = hasattr(module, "flat_history")  # recurrent modules: the input is the single-step obs
    if batched:
        module.reset()
        actions, flats = [], []
        for k in range(num_steps):
            if call_reset and resets.get(k):
                module.reset(torch.tensor(resets[k], dtype=torch.long))
            actions.append(module(obs_seq[k]))
            flats.append(module.flat_history() if has_flat else obs_seq[k])
        return torch.stack(actions), torch.stack(flats)
    per_env_a, per_env_f = [], []
    for e in range(num_envs):
        module.reset()
        actions, flats = [], []
        for k in range(num_steps):
            if call_reset and k > 0 and e in resets.get(k, []):
                module.reset()
            actions.append(module(obs_seq[k, e:e + 1]))
            flats.append(module.flat_history() if has_flat else obs_seq[k, e:e + 1])
        per_env_a.append(torch.cat(actions))
        per_env_f.append(torch.cat(flats))
    return torch.stack(per_env_a, dim=1), torch.stack(per_env_f, dim=1)


def default_resets(num_steps: int, num_envs: int, history_length: int) -> dict[int, list[int]]:
    """Env 0 never resets (long enough for the buffer to wrap); the others reset at different steps, including
    back-to-back resets (one-step episodes) and resets before the history is full."""
    h = max(history_length, 1)
    plan = {1: [max(h // 2, 1), num_steps - h // 2 - 1], 2: [2, 3, h + 4], 3: [h + 1, 2 * h + 3]}
    resets: dict[int, list[int]] = {}
    for e, steps in plan.items():
        if e < num_envs:
            for k in steps:
                if 0 < k < num_steps:
                    resets.setdefault(k, []).append(e)
    return {k: sorted(v) for k, v in sorted(resets.items())}


def single_env_resets(resets: dict[int, list[int]], env: int) -> dict[int, list[int]]:
    """The reset schedule of one env, as env index 0 of a batch of one."""
    return {k: [0] for k, ids in resets.items() if env in ids}


def reference_rollout_per_env(policy: nn.Module, layout: dict, obs_seq: torch.Tensor, resets: dict[int, list[int]],
                              buffer_cls: type) -> tuple[torch.Tensor, torch.Tensor]:
    """reference_rollout with one env at a time (batch 1, the deploy case); results stacked to (T, N, ...).

    float32 kernels depend on the batch size (~1e-6 relative), so batch-1 module outputs are compared with batch-1
    reference outputs.
    """
    out = [reference_rollout(policy, layout, obs_seq[:, e:e + 1], single_env_resets(resets, e), buffer_cls)
           for e in range(obs_seq.shape[1])]
    return torch.cat([a for a, _ in out], dim=1), torch.cat([f for _, f in out], dim=1)


def self_check(policy: nn.Module, layout: dict, path: str, num_envs: int = 4, seed: int = 0) -> dict:
    """Reload the saved file and compare it with the original policy on a random rollout with resets."""
    buffer_cls = load_circular_buffer()
    if buffer_cls is None:
        print("[export] isaaclab not importable: self-check skipped")
        return {}
    module = torch.jit.load(path)
    num_steps = 3 * max(layout["history_length"], 1) + 8
    resets = default_resets(num_steps, num_envs, layout["history_length"])
    obs_seq = random_obs_sequence(layout, num_steps, num_envs, resets, seed)
    refs = {"batched": reference_rollout(policy, layout, obs_seq, resets, buffer_cls),
            "batch1": reference_rollout_per_env(policy, layout, obs_seq, resets, buffer_cls)}
    res = {}
    for mode, (ref_a, ref_flat) in refs.items():
        a, flat = module_rollout(module, obs_seq, resets, batched=mode == "batched")
        res[mode] = {"max_abs_action_diff": float((a - ref_a).abs().max()),
                     "flat_obs_equal": bool(torch.equal(flat, ref_flat))}
    module.reset()
    return res


def main():
    parser = argparse.ArgumentParser(description="Export a history-based policy as TorchScript for the robot.")
    parser.add_argument("--run", required=True, help="run directory (logs/rsl_rl/<experiment>/<run>)")
    parser.add_argument("--checkpoint", default=None, help="model_X.pt in the run dir (default: latest)")
    parser.add_argument("--task", default=None, help="task id for the metadata (default: from <run>/eval/metrics.json)")
    parser.add_argument("--overwrite", action="store_true", help=f"overwrite an existing exported/{OUTPUT_NAME}.pt")
    parser.add_argument("--no_check", action="store_true", help="skip the equivalence self-check")
    parser.add_argument("--tol", type=float, default=1e-5, help="self-check tolerance on the actions")
    args = parser.parse_args()

    policy, layout, info = load_run(args.run, args.checkpoint)
    task = args.task or task_id_of_run(info["run"])
    out_dir = os.path.join(info["run"], "exported")
    pt_path = os.path.join(out_dir, f"{OUTPUT_NAME}.pt")
    json_path = os.path.join(out_dir, f"{OUTPUT_NAME}.json")
    if os.path.exists(pt_path) and not args.overwrite:
        sys.exit(f"[export] {pt_path} exists (use --overwrite)")
    print(f"[export] {info['policy_class']} from {info['checkpoint']}")
    terms = ", ".join(f"{t['name']} {t['dim']}" for t in layout["terms"])
    print(f"[export] task {task}, history {layout['history_length']} x {layout['obs_dim']} -> flat {layout['flat_dim']}"
          f" ({terms})")
    metadata = export_metadata(layout, info, task)
    export(policy, layout, pt_path, metadata)
    with open(pt_path, "rb") as f:
        metadata["sha256"] = hashlib.sha256(f.read()).hexdigest()
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=1)
    print(f"[export] wrote {pt_path} and {json_path}")
    if not args.no_check:
        res = self_check(policy, layout, pt_path)
        for mode, r in res.items():
            ok = r["max_abs_action_diff"] <= args.tol and r["flat_obs_equal"]
            print(f"[export] self-check {mode}: max |action diff| {r['max_abs_action_diff']:.2e}, flat obs identical: "
                  f"{r['flat_obs_equal']} -> {'OK' if ok else 'FAILED'}")
            if not ok:
                sys.exit(1)


if __name__ == "__main__":
    main()
