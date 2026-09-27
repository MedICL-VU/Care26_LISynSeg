"""Rooted AO/PA distal endpoint loss mask (paper Section 2.3)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    return F.max_pool3d(
        mask[None, None].float(), 2 * radius + 1, stride=1, padding=radius
    )[0, 0] > 0


@torch.no_grad()
def mask_distal_aopa(
    target: torch.Tensor,
    *,
    spacing_zyx: Sequence[float],
    ignore_label: int = -1,
    endpoint_quantile: float = 0.90,
    cap_quantile: float = 0.96,
    continuation_radius: int = 4,
    max_tail_fraction: float = 0.40,
) -> torch.Tensor:
    """Mark uncertain distal vessel and nearby continuation voxels as ignored.

    Accepts a single (Z,Y,X) map or an nnU-Net style (B,1,Z,Y,X) target.
    AO is rooted at LV; PA at RV. Spacing follows tensor axis order (Z,Y,X).
    Apply to *both* real and synthetic targets just before loss computation.
    The loss must support ``ignore_label`` for Dice and cross-entropy.
    """
    if target.ndim == 3:
        batch = target[None, None]
    elif target.ndim == 5 and target.shape[1] == 1:
        batch = target
    else:
        raise ValueError("target must have shape (Z,Y,X) or (B,1,Z,Y,X)")
    if len(spacing_zyx) != 3 or any(float(s) <= 0 for s in spacing_zyx):
        raise ValueError("spacing_zyx must contain three positive values")
    if not 0 < endpoint_quantile <= cap_quantile < 1:
        raise ValueError("expected 0 < endpoint_quantile <= cap_quantile < 1")
    if continuation_radius < 0 or not 0 < max_tail_fraction < 1:
        raise ValueError("invalid continuation radius or tail fraction")

    out = batch.clone()
    spacing = torch.as_tensor(spacing_zyx, dtype=torch.float32, device=out.device)
    for item in out[:, 0]:
        for vessel_label, chamber_label in ((6, 3), (7, 5)):
            vessel = item == vessel_label
            chamber = item == chamber_label
            n_vessel = int(vessel.sum())
            if n_vessel < 4 or not bool(chamber.any()):
                continue
            roots = torch.nonzero(vessel & _dilate(chamber, 1), as_tuple=False)
            if roots.numel() == 0:
                continue  # no ventricular contact within the patch
            root_mm = (roots.float() * spacing).mean(dim=0)
            coords = torch.nonzero(vessel, as_tuple=False)
            distances = torch.linalg.vector_norm(coords.float() * spacing - root_mm, dim=1)
            tail_cutoff = torch.quantile(distances, endpoint_quantile)
            tail = distances >= tail_cutoff
            if int(tail.sum()) / n_vessel > max_tail_fraction:
                continue  # ties would erase too much supervision
            item[tuple(coords[tail].T)] = ignore_label

            cap_cutoff = torch.quantile(distances, cap_quantile)
            cap = torch.zeros_like(vessel)
            cap[tuple(coords[distances >= cap_cutoff].T)] = True
            nearby_background = _dilate(cap, continuation_radius) & (item == 0)
            bg_coords = torch.nonzero(nearby_background, as_tuple=False)
            if bg_coords.numel() == 0:
                continue
            bg_distances = torch.linalg.vector_norm(bg_coords.float() * spacing - root_mm, dim=1)
            outward = bg_distances >= cap_cutoff - 0.5 * float(spacing.min())
            item[tuple(bg_coords[outward].T)] = ignore_label
    return out[0, 0] if target.ndim == 3 else out
