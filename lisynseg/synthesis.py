"""Fold-calibrated, task-label-only image synthesis (paper Section 2.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch.nn import functional as F


def training_acquisition_profiles(
    native_spacing_by_case: Mapping[str, Sequence[float]],
    train_cases: Sequence[str],
    validation_cases: Sequence[str],
    target_spacing_zyx: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Compute native/target spacing ratios from training subjects only.

    Supply spacing in tensor axis order (Z,Y,X). A repeated profile retains
    its empirical frequency; profiles are capped at 3 when sampled for rendering.
    """
    if set(train_cases) & set(validation_cases):
        raise ValueError("train and validation cases overlap")
    if not train_cases or len(target_spacing_zyx) != 3:
        raise ValueError("provide training cases and three target spacings")
    target = tuple(float(x) for x in target_spacing_zyx)
    if any(not math.isfinite(x) or x <= 0 for x in target):
        raise ValueError("target spacing must be positive and finite")
    profiles = []
    for case in train_cases:
        native = tuple(float(x) for x in native_spacing_by_case[case])
        if len(native) != 3 or any(not math.isfinite(x) or x <= 0 for x in native):
            raise ValueError(f"invalid native spacing for {case}")
        profiles.append(tuple(max(1.0, a / b) for a, b in zip(native, target)))
    return tuple(profiles)


def _uniform(generator: torch.Generator, device: torch.device, upper: float) -> float:
    return upper * float(torch.rand((), generator=generator, device=device))


def _smooth_random_field(
    shape: tuple[int, int, int], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    low = (4, 4, 4)
    field = torch.randn((1, 1, *low), generator=generator, device=device)
    field = F.interpolate(field, size=shape, mode="trilinear", align_corners=False)[0, 0]
    return (field - field.mean()) / field.std(unbiased=False).clamp_min(1e-6)


def _acquisition_effect(image: torch.Tensor, factors: Sequence[float]) -> torch.Tensor:
    """Blur, downsample, and restore to the nnU-Net patch grid."""
    factors = tuple(max(1.0, min(3.0, float(f))) for f in factors)
    if max(factors) <= 1.01:
        return image
    size = tuple(int(x) for x in image.shape)
    low_size = tuple(max(2, round(n / f)) for n, f in zip(size, factors))
    kernel = tuple(2 * round(f) + 1 for f in factors)
    blurred = F.avg_pool3d(
        image[None, None], kernel_size=kernel, stride=1,
        padding=tuple(k // 2 for k in kernel), count_include_pad=False,
    )
    low = F.interpolate(blurred, size=low_size, mode="trilinear", align_corners=False)
    return F.interpolate(low, size=size, mode="trilinear", align_corners=False)[0, 0]


@torch.no_grad()
def synthesize_from_labels(
    labels: torch.Tensor,
    acquisition_profiles: Sequence[Sequence[float]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Render one standardized image from a 3D task-label map (values 0..7).

    Every present label has its own sampled Gaussian intensity. Two iterations
    of mask averaging make soft partial-volume weights. No spatial transform is
    applied here: call this after nnU-Net's standard augmentation. Neither CT/MR
    identity nor source intensities are used by the renderer.
    """
    if labels.ndim != 3 or bool(((labels < 0) | (labels > 7)).any()):
        raise ValueError("labels must be a (Z,Y,X) map with values 0..7")
    if not acquisition_profiles:
        raise ValueError("training-fold acquisition profiles are required")
    for profile in acquisition_profiles:
        if len(profile) != 3 or any(not math.isfinite(float(f)) or float(f) <= 0 for f in profile):
            raise ValueError("each acquisition profile needs three positive finite factors")
    device = labels.device
    shape = tuple(int(x) for x in labels.shape)
    image = torch.zeros(shape, device=device)
    total_weight = torch.zeros_like(image)
    for value in torch.unique(labels).tolist():
        weight = (labels == int(value)).float()
        for _ in range(2):
            weight = F.avg_pool3d(
                weight[None, None], 3, stride=1, padding=1, count_include_pad=False
            )[0, 0]
        mean = _uniform(generator, device, 255.0)
        std = _uniform(generator, device, 12.0)
        region = mean + std * torch.randn(shape, generator=generator, device=device)
        image += weight * region
        total_weight += weight
    image /= total_weight.clamp_min(1e-6)

    bias_std = _uniform(generator, device, 0.35)
    image *= torch.exp(_smooth_random_field(shape, generator, device) * bias_std)
    gamma = float(torch.exp(torch.randn((), generator=generator, device=device) * 0.15))
    lo, hi = image.amin(), image.amax()
    span = (hi - lo).clamp_min(1e-6)
    image = ((image - lo) / span).clamp(0, 1).pow(gamma) * span + lo
    noise_std = _uniform(generator, device, 3.0)
    image += noise_std * torch.randn(shape, generator=generator, device=device)

    index = int(torch.randint(len(acquisition_profiles), (), generator=generator, device=device))
    image = _acquisition_effect(image, acquisition_profiles[index])
    image -= image.mean()
    return image / image.std(unbiased=False).clamp_min(1e-6)
