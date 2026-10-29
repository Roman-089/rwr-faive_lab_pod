# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recurrent (LSTM / GRU) student distilled from our feed-forward MLP teacher (docs/ablation/teacher_student.md).

rsl_rl 3.1.2's ``StudentTeacherRecurrent`` builds the teacher MLP with ``rnn_hidden_dim`` inputs even when
``teacher_recurrent=False``, so loading an MLP teacher fails (`size mismatch for 0.weight: [512, 81]` vs `[512, 256]`;
job 29839015, 2026-10-29). Its ``evaluate()`` already feeds the normalized teacher observation straight into the teacher
MLP for a non-recurrent teacher, so only the constructor is wrong: rebuild the teacher MLP on the teacher obs size.

rsl_rl builds the policy with ``eval(class_name)`` inside ``rsl_rl.runners.distillation_runner``, so ``register()``
(called from this package's ``__init__``) puts the class into that module's namespace.
"""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.modules import StudentTeacherRecurrent
from rsl_rl.networks import MLP


class StudentTeacherRecurrentMLPTeacher(StudentTeacherRecurrent):
    """``StudentTeacherRecurrent`` whose non-recurrent teacher MLP takes the teacher observation as input."""

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], num_actions: int, **kwargs) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if not self.teacher_recurrent:
            num_teacher_obs = sum(obs[g].shape[-1] for g in obs_groups["teacher"])
            self.teacher = MLP(
                num_teacher_obs,
                num_actions,
                kwargs.get("teacher_hidden_dims", [256, 256, 256]),
                kwargs.get("activation", "elu"),
            )
            self.teacher.eval()
            print(f"Teacher MLP (rebuilt on {num_teacher_obs} teacher obs): {self.teacher}")


def register():
    import rsl_rl.runners.distillation_runner as runner_module

    runner_module.StudentTeacherRecurrentMLPTeacher = StudentTeacherRecurrentMLPTeacher
