# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom noise configurations for observations."""

import torch
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


def apply_gaussian_noise_and_normalize_quat(data: torch.Tensor, cfg: "GaussianNoiseAndNormalizeQuatCfg") -> torch.Tensor:
    """Apply Gaussian noise to quaternion components, then renormalize to unit norm.

    This preserves the quaternion constraint while adding noise.

    Args:
        data: Quaternion tensor of shape (..., 4) in (w, x, y, z) format.
        cfg: Noise configuration with std parameter.

    Returns:
        Noisy quaternion with unit norm preserved.
    """
    # Add Gaussian noise to all components
    noise = torch.randn_like(data) * cfg.std
    noisy_quat = data + noise

    # Renormalize to unit norm
    noisy_quat = noisy_quat / torch.norm(noisy_quat, dim=-1, keepdim=True)

    return noisy_quat


@configclass
class GaussianNoiseAndNormalizeQuatCfg(NoiseCfg):
    """Gaussian noise applied to quaternions with post-normalization.

    Adds Gaussian noise to quaternion components then normalizes to preserve
    the unit norm constraint. This is simpler than angle-axis noise and keeps
    the same observation dimension.
    """

    func = apply_gaussian_noise_and_normalize_quat
    """The noise function."""

    std: float = MISSING
    """Standard deviation of Gaussian noise applied to quaternion components."""

    operation: str = "add"
    """Must be 'add' for this noise type."""
