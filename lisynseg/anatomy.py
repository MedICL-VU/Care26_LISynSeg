"""Myo–LV interface exchange (paper Section 2.2)."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(mask[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0


def _smooth_noise(shape: tuple[int, int, int], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    coarse = tuple(max(2, (size + 7) // 8) for size in shape)
    noise = torch.randn((1, 1, *coarse), generator=generator, device=device)
    return F.interpolate(noise, size=shape, mode="trilinear", align_corners=False)[0, 0]


@torch.no_grad()
def myo_lv_boundary_exchange(
    labels: torch.Tensor,
    generator: torch.Generator,
    *,
    direction: str = "random",
    myo_label: int = 1,
    lv_label: int = 3,
) -> torch.Tensor:
    """Move a smooth subset of the one-voxel endocardial band.

    The Myo/LV union and every other class are preserved. When either class is
    absent or the volume gate allows no transfer, the input is returned unchanged.
    ``labels`` is one (Z, Y, X) task-label patch after nnU-Net augmentation.
    """
    if labels.ndim != 3:
        raise ValueError("labels must have shape (Z, Y, X)")
    if direction not in {"random", "thicken", "thin"}:
        raise ValueError("direction must be random, thicken, or thin")
    myo, lv = labels == myo_label, labels == lv_label
    if not bool(myo.any()) or not bool(lv.any()):
        return labels.clone()
    if direction == "random":
        direction = "thicken" if torch.rand((), generator=generator, device=labels.device) < 0.5 else "thin"

    if direction == "thicken":
        band = _dilate(myo) & lv  # B_{LV -> Myo}
        donor_budget = math.floor(0.20 * int(lv.sum()))
        recipient_budget = math.floor(0.35 * int(myo.sum()))
        new_label = myo_label
    else:
        band = _dilate(lv) & myo  # B_{Myo -> LV}
        donor_budget = math.floor(0.28 * int(myo.sum()))
        recipient_budget = math.floor(0.20 * int(lv.sum()))
        new_label = lv_label

    maximum = min(donor_budget, recipient_budget)
    if maximum < 1:
        return labels.clone()
    fraction = 0.35 + 0.50 * float(torch.rand((), generator=generator, device=labels.device))
    candidate_indices = torch.nonzero(band.reshape(-1), as_tuple=False).squeeze(1)
    if candidate_indices.numel() == 0:
        return labels.clone()
    n_exchange = min(candidate_indices.numel(), max(1, math.floor(maximum * fraction)))
    scores = _smooth_noise(tuple(labels.shape), generator, labels.device).reshape(-1)
    chosen = candidate_indices[torch.topk(scores[candidate_indices], n_exchange).indices]
    result = labels.clone()
    result.reshape(-1)[chosen] = new_label
    assert torch.equal((result == myo_label) | (result == lv_label), myo | lv)
    return result
