from __future__ import annotations

import inspect

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)


_HAS_UNPACK_DATASET = "unpack_dataset" in inspect.signature(nnUNetTrainerDA5.__init__).parameters


def _default_device(device):
    if device is None:
        import torch

        device = torch.device("cuda")
    return device


def _init_da5_trainer(trainer, plans, configuration, fold, dataset_json, unpack_dataset, device):
    kwargs = {
        "plans": plans,
        "configuration": configuration,
        "fold": fold,
        "dataset_json": dataset_json,
        "device": _default_device(device),
    }
    if "unpack_dataset" in inspect.signature(nnUNetTrainerDA5.__init__).parameters:
        kwargs["unpack_dataset"] = unpack_dataset
    nnUNetTrainerDA5.__init__(trainer, **kwargs)


class _CAREForceNumpyDatasetMixin:
    def initialize(self) -> None:
        from nnunetv2.training.dataloading import nnunet_dataset
        from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as trainer_module

        dataset_class = getattr(nnunet_dataset, "nnUNetDatasetNumpy", None)
        if dataset_class is None:
            return super().initialize()

        old_infer_dataset_class = trainer_module.infer_dataset_class
        trainer_module.infer_dataset_class = lambda folder: dataset_class
        try:
            return super().initialize()
        finally:
            trainer_module.infer_dataset_class = old_infer_dataset_class


if _HAS_UNPACK_DATASET:

    class nnUNetTrainer_CARE_DA5_10epochs(_CAREForceNumpyDatasetMixin, nnUNetTrainerDA5):
        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device=None,
        ):
            _init_da5_trainer(self, plans, configuration, fold, dataset_json, unpack_dataset, device)
            self.num_epochs = 10


    class nnUNetTrainer_CARE_DA5_500epochs(_CAREForceNumpyDatasetMixin, nnUNetTrainerDA5):
        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device=None,
        ):
            _init_da5_trainer(self, plans, configuration, fold, dataset_json, unpack_dataset, device)
            self.num_epochs = 500

else:

    class nnUNetTrainer_CARE_DA5_10epochs(_CAREForceNumpyDatasetMixin, nnUNetTrainerDA5):
        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device=None,
        ):
            _init_da5_trainer(self, plans, configuration, fold, dataset_json, True, device)
            self.num_epochs = 10


    class nnUNetTrainer_CARE_DA5_500epochs(_CAREForceNumpyDatasetMixin, nnUNetTrainerDA5):
        def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device=None,
        ):
            _init_da5_trainer(self, plans, configuration, fold, dataset_json, True, device)
            self.num_epochs = 500


__all__ = [
    "nnUNetTrainer_CARE_DA5_10epochs",
    "nnUNetTrainer_CARE_DA5_500epochs",
]
