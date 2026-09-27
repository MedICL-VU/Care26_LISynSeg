import unittest

import torch

from lisynseg import (
    mask_distal_aopa,
    myo_lv_boundary_exchange,
    prepare_training_batch,
    training_acquisition_profiles,
)


class CoreTests(unittest.TestCase):
    def test_exchange_preserves_union_and_other_labels(self):
        target = torch.zeros((24, 24, 24), dtype=torch.int16)
        target[4:20, 4:20, 4:20] = 1
        target[7:17, 7:17, 7:17] = 3
        target[1:3, 1:3, 1:3] = 5
        result = myo_lv_boundary_exchange(
            target, torch.Generator().manual_seed(29), direction="thicken"
        )
        union = (target == 1) | (target == 3)
        self.assertTrue(torch.equal(union, (result == 1) | (result == 3)))
        self.assertTrue(torch.equal(target[~union], result[~union]))
        self.assertGreater(int((result == 1).sum()), int((target == 1).sum()))

    def test_endpoint_mask_retains_root_and_ignores_continuation(self):
        target = torch.zeros((9, 9, 18), dtype=torch.int16)
        target[4, 4, 1:4] = 3
        target[4, 4, 4:14] = 6
        result = mask_distal_aopa(
            target, spacing_zyx=(1, 1, 1),
            endpoint_quantile=0.80, cap_quantile=0.90,
            continuation_radius=2,
        )
        self.assertTrue(torch.equal(target[4, 4, 4:9], result[4, 4, 4:9]))
        self.assertTrue(torch.all(result[4, 4, 12:14] == -1))
        self.assertEqual(int(result[4, 4, 14]), -1)
        self.assertEqual(int(result[4, 4, 0]), 0)

    def test_profiles_exclude_validation_subjects(self):
        spacings = {"train": (2, 1, 1), "val": (3, 1, 1)}
        self.assertEqual(
            training_acquisition_profiles(spacings, ["train"], ["val"], (1, 1, 1)),
            ((2.0, 1.0, 1.0),),
        )
        with self.assertRaises(ValueError):
            training_acquisition_profiles(spacings, ["train"], ["train"], (1, 1, 1))

    def test_batch_changes_synthetic_target_and_masks_real_and_synthetic(self):
        images = torch.zeros((2, 1, 20, 20, 24))
        targets = torch.zeros_like(images, dtype=torch.int16)
        targets[:, :, 3:17, 3:17, 3:17] = 1
        targets[:, :, 6:14, 6:14, 6:14] = 3
        targets[:, :, 10, 10, 14:22] = 6
        output, masked, selected = prepare_training_batch(
            images, targets, spacing_zyx=(1, 1, 1),
            acquisition_profiles=((1, 1, 1),),
            generator=torch.Generator().manual_seed(7),
            synthetic_probability=1.0,
        )
        self.assertTrue(bool(selected.all()))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(bool((output != images).any()))
        self.assertTrue(bool((masked == -1).any()))
        self.assertTrue(torch.equal(
            (masked == 1) | (masked == 3),
            (targets == 1) | (targets == 3),
        ))

    def test_real_branch_keeps_image_but_masks_endpoint(self):
        images = torch.randn((1, 1, 9, 9, 18))
        targets = torch.zeros_like(images, dtype=torch.int16)
        targets[0, 0, 4, 4, 1:4] = 3
        targets[0, 0, 4, 4, 4:14] = 6
        output, masked, selected = prepare_training_batch(
            images, targets, spacing_zyx=(1, 1, 1),
            acquisition_profiles=((1, 1, 1),),
            generator=torch.Generator().manual_seed(1),
            synthetic_probability=0.0,
        )
        self.assertFalse(bool(selected.any()))
        self.assertTrue(torch.equal(output, images))
        self.assertTrue(bool((masked == -1).any()))


if __name__ == "__main__":
    unittest.main()
