"""Minimal batch hook illustrating the published training sequence."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .anatomy import myo_lv_boundary_exchange
from .synthesis import synthesize_from_labels
from .vessels import mask_distal_aopa


@torch.no_grad()
def prepare_training_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    spacing_zyx: Sequence[float],
    acquisition_profiles: Sequence[Sequence[float]],
    generator: torch.Generator,
    synthetic_probability: float = 0.10,
    exchange_myo_lv: bool = True,
    mask_aopa: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the real/synthetic branch then optional label operations.

    Inputs are already augmented nnU-Net tensors of shape (B,1,Z,Y,X).
    Returns images, targets (possibly containing -1), and synthetic selection.
    For deep supervision, build each target scale from the unmasked labels, then
    apply the endpoint mask separately at that scale with its adjusted spacing.
    """
    if images.ndim != 5 or images.shape[1] != 1 or targets.shape != images.shape:
        raise ValueError("images and targets must both have shape (B,1,Z,Y,X)")
    if not 0 <= synthetic_probability <= 1:
        raise ValueError("synthetic_probability must be in [0,1]")
    out_images, out_targets = images.clone(), targets.clone()
    selected = torch.rand(images.shape[0], generator=generator, device=images.device) < synthetic_probability
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        label = out_targets[index, 0]
        if exchange_myo_lv:
            label = myo_lv_boundary_exchange(label, generator)
        out_targets[index, 0] = label
        out_images[index, 0] = synthesize_from_labels(label, acquisition_profiles, generator).to(images.dtype)
    if mask_aopa:
        out_targets = mask_distal_aopa(out_targets, spacing_zyx=spacing_zyx)
    return out_images, out_targets, selected
