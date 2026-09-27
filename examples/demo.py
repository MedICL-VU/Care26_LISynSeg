"""Tiny synthetic-only smoke example; no CARE data are needed."""

import torch

from lisynseg import prepare_training_batch


def main() -> None:
    images = torch.randn(2, 1, 24, 24, 32)
    targets = torch.zeros_like(images, dtype=torch.int16)
    targets[:, :, 4:20, 4:20, 4:20] = 1  # Myo
    targets[:, :, 7:17, 7:17, 7:17] = 3  # LV
    targets[:, :, 12, 12, 17:29] = 6    # AO touching LV
    generator = torch.Generator().manual_seed(26)
    out_images, out_targets, selected = prepare_training_batch(
        images, targets,
        spacing_zyx=(1.5, 1.0, 1.0),
        acquisition_profiles=((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
        generator=generator,
        synthetic_probability=1.0,  # force synthesis so the example is easy to inspect
    )
    print(f"images: {tuple(out_images.shape)}; selected: {selected.tolist()}")
    print(f"changed image voxels: {int((out_images != images).sum())}")
    print(f"ignored endpoint voxels: {int((out_targets == -1).sum())}")


if __name__ == "__main__":
    main()
