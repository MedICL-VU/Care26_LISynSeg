# LISynSeg

**MICCAI 2026 CARE Workshop Best Paper Award**

**LISynSeg: Data-Centric Label-to-Image Synthesis for Cross-Modality Whole-Heart Segmentation**

Jiacheng Wang, Ivana Isgum, Ipek Oguz

[Paper (arXiv:2608.31073)](https://arxiv.org/pdf/2608.31073) · [CARE 2026 challenge](https://www.zmic.org.cn/care_2026/) · [CARE Whole-Heart track](https://www.zmic.org.cn/care_2026/track_wholeheart/)

LISynSeg changes the **training examples and supervision** of a standard 3D full-resolution nnU-Net. It mixes real CT/MRI patches with images synthesized from cardiac labels, perturbs the Myo–LV interface on synthetic examples, and excludes uncertain distal AO/PA endpoint voxels from the training loss. Inference uses ordinary nnU-Net.

This is a **compact method release**: the central operations and equations are provided in readable form. It is not a full reproduction of the original nnU-Net experiments, data preparation, challenge submission, or trained weights. The code expects preprocessed, already augmented nnU-Net patches and internal labels `0..7`.

## What is here

| Component | Code | Role |
| --- | --- | --- |
| Label-to-image synthesis | [`lisynseg/synthesis.py`](lisynseg/synthesis.py) | Sample Gaussian class appearance, partial volume, bias, gamma, noise, and training-fold acquisition resolution. |
| Myo–LV boundary exchange | [`lisynseg/anatomy.py`](lisynseg/anatomy.py) | Change only the endocardial boundary while preserving the Myo/LV union. |
| AO/PA endpoint masking | [`lisynseg/vessels.py`](lisynseg/vessels.py) | Keep the proximal trunks supervised and ignore uncertain distal foreground and nearby background. |
| Training order | [`lisynseg/training.py`](lisynseg/training.py) | Demonstrate the 10% synthetic branch and masking on both real and synthetic targets. |

## Method in equations

The input pair `(xᵃ, yᵃ)` has already received standard nnU-Net augmentation. For each example, draw `z ~ Bernoulli(ρ)`, where `ρ = 0.10`:

```text
yᵍ = G(yᵃ; η)                         Myo–LV boundary exchange
(x′, y′) = (S(yᵍ; θ), yᵍ)   if z = 1  synthetic branch
          = (xᵃ, yᵃ)       if z = 0  real branch
```

`S` uses the task labels directly. Each present label `l` has independently sampled `μₗ ~ U(0,255)` and `σₗ ~ U(0,12)`. Soft class weights simulate partial volume. Further draws use a bias-field scale from `U(0,0.35)`, `log γ ~ N(0,0.15²)`, Gaussian-noise scale from `U(0,3)`, and a native-to-target spacing profile sampled only from the **training cases of the current fold**. Resolution factors are capped at 3. The final image has zero mean and unit variance. No additional spatial warp is applied in `S` because the patch was already spatially augmented.

For Myo set `M`, LV set `V`, and one-voxel 3D dilation `D₁`, the candidate interface bands are:

```text
B(LV→Myo) = D₁(M) ∩ V
B(Myo→LV) = D₁(V) ∩ M
```

Thickening or thinning is chosen equally. A smooth random field selects a subset of the band. The transfer is limited so the new/original volumes stay within `[0.72,1.35]` for Myo and `[0.80,1.20]` for LV; a fraction from `U(0.35,0.85)` of the admissible transfer is used. `M ∪ V` and all other labels remain unchanged. This operation runs only on selected synthetic examples.

For each vessel `c`, the proximal root `R_c` consists of AO voxels touching a one-voxel dilation of LV, or PA voxels touching a one-voxel dilation of RV. With target spacing `s` and voxel coordinate `v`:

```text
p(v) = s ⊙ v
r_c = mean { p(v) : v ∈ R_c }
d_c(v) = ||p(v) − r_c||₂
```

The most distal 10% of annotated vessel voxels are ignored. Background within four voxels of the distal cap (the 96th distance percentile) is also ignored when at or beyond the cap distance, with a half-minimum-spacing tolerance. Masking is skipped if there is no ventricular root in the patch or distance ties would mask more than 40% of a vessel. The objective uses Dice plus cross-entropy over the remaining voxels:

```text
L = L_Dice+CE(fφ(x′), y′; Ω \ (U_AO ∪ U_PA))
```

AO/PA masking runs on **both** real and synthetic examples during training. It does not alter validation targets or inference. Set the nnU-Net loss `ignore_label` to `-1` for both Dice and cross-entropy.

## Quick start

Install a compatible [PyTorch](https://pytorch.org/get-started/locally/) build, then:

```bash
python -m examples.demo
python -m unittest discover -s tests -v
```

Minimal use after the ordinary nnU-Net batch augmentations:

```python
import torch
from lisynseg import prepare_training_batch, training_acquisition_profiles

profiles = training_acquisition_profiles(
    native_spacing_by_case, train_cases, validation_cases, target_spacing_zyx
)
generator = torch.Generator(device=images.device).manual_seed(2026)
images, targets, was_synthetic = prepare_training_batch(
    images, targets, spacing_zyx=target_spacing_zyx,
    acquisition_profiles=profiles, generator=generator,
)
# Pass images and targets to a Dice+CE loss configured with ignore_label=-1.
```

The public CARE labels are `205=Myo, 420=LA, 500=LV, 550=RA, 600=RV, 820=AO, 850=PA`. Remap them to nnU-Net task labels `1..7` in that order before using this code. The release contains no patient images, annotations, model checkpoints, fold assignments, or challenge credentials.

## Citation and acknowledgment

If this work helps your research, please cite the paper:

```bibtex
@misc{wang2026lisynseg,
  title={LISynSeg: Data-Centric Label-to-Image Synthesis for Cross-Modality Whole-Heart Segmentation},
  author={Wang, Jiacheng and Isgum, Ivana and Oguz, Ipek},
  year={2026},
  eprint={2608.31073},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.31073}
}
```

We thank the organizers of the [CARE 2026 challenge](https://www.zmic.org.cn/care_2026/) and its [Whole-Heart track](https://www.zmic.org.cn/care_2026/track_wholeheart/) for providing the benchmark, labels, evaluation protocol, and workshop. Research using the challenge data should also cite the dataset papers listed on the official Whole-Heart track page (Zhuang and Shen, *Medical Image Analysis*, 2016; Zhuang, *IEEE TPAMI*, 2019; Gao et al., *Medical Image Analysis*, 2023) and follow the challenge data terms.
