from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: int
    dataset_name: str
    trainer: str
    plans: str
    folds: tuple[int, ...]
    model_bundle_name: str
    input_contract: str
    modalities: tuple[str, ...]
    checkpoint_name: str = "checkpoint_best.pth"

    @property
    def model_relative_path(self) -> str:
        return f"{self.dataset_name}/{self.trainer}__{self.plans}__3d_fullres"


FIVE_FOLDS = (0, 1, 2, 3, 4)

# Keep this order aligned with the frozen OOF probability accumulation order.
SOURCES = (
    SourceSpec(
        "613_basic", 613, "Dataset613_CAREWHS_CTMR_5fold",
        "nnUNetTrainer_500epochs", "nnUNetPlans", FIVE_FOLDS,
        "dataset613_basic", "raw_both", ("ct", "mr"),
    ),
    SourceSpec(
        "615_basic", 615, "Dataset615_CAREWHS_CTMR_MRN4_5fold",
        "nnUNetTrainer_500epochs", "nnUNetPlans", FIVE_FOLDS,
        "dataset615_basic", "n4_both", ("ct", "mr"),
    ),
    SourceSpec(
        "611_basic", 611, "Dataset611_CAREWHS_CT_5fold",
        "nnUNetTrainer_500epochs", "nnUNetPlans", FIVE_FOLDS,
        "dataset611_basic", "raw_ct", ("ct",),
    ),
    SourceSpec(
        "613_da5", 613, "Dataset613_CAREWHS_CTMR_5fold",
        "nnUNetTrainer_CARE_DA5_500epochs", "nnUNetPlans", FIVE_FOLDS,
        "dataset613_da5", "raw_both", ("ct", "mr"),
    ),
    SourceSpec(
        "615_da5", 615, "Dataset615_CAREWHS_CTMR_MRN4_5fold",
        "nnUNetTrainer_CARE_DA5_500epochs", "nnUNetPlans", FIVE_FOLDS,
        "dataset615_da5", "n4_both", ("ct", "mr"),
    ),
    SourceSpec(
        "613_resenc", 613, "Dataset613_CAREWHS_CTMR_5fold",
        "nnUNetTrainer", "nnUNetResEncUNetLPlans", FIVE_FOLDS,
        "dataset613_resenc", "raw_both", ("ct", "mr"),
    ),
    SourceSpec(
        "611_resenc", 611, "Dataset611_CAREWHS_CT_5fold",
        "nnUNetTrainer", "nnUNetResEncUNetLPlans", (1,),
        "dataset611_resenc", "raw_ct", ("ct",),
    ),
    SourceSpec(
        "679_resenc", 679, "Dataset679_CAREWHS_MROnly_From613_5fold",
        "nnUNetTrainer", "nnUNetResEncUNetLPlans", FIVE_FOLDS,
        "dataset679_resenc", "raw_mr", ("mr",),
    ),
    SourceSpec(
        "679_resenc_final", 679, "Dataset679_CAREWHS_MROnly_From613_5fold",
        "nnUNetTrainer", "nnUNetResEncUNetLPlans", FIVE_FOLDS,
        "dataset679_resenc", "raw_mr", ("mr",), "checkpoint_final.pth",
    ),
)


VARIANTS = {
    "submission1": {
        "description": "CT high020; MR high025; keep all MR PA components",
        "ct_weights": {
            "613_basic": 2, "615_basic": 2, "611_basic": 2,
            "613_da5": 2, "615_da5": 2, "613_resenc": 2,
            "611_resenc": 3,
        },
        "mr_weights": {
            "613_basic": 3, "615_basic": 3, "613_da5": 3,
            "615_da5": 3, "613_resenc": 3, "679_resenc": 5,
            "679_resenc_final": 0,
        },
        "mr_skip_lcc_labels": (850,),
    },
    "submission2": {
        "description": "CT high025 DSC tilt; MR high020; keep all MR PA components",
        "ct_weights": {
            "613_basic": 1, "615_basic": 1, "611_basic": 1,
            "613_da5": 1, "615_da5": 1, "613_resenc": 1,
            "611_resenc": 2,
        },
        "mr_weights": {
            "613_basic": 4, "615_basic": 4, "613_da5": 4,
            "615_da5": 4, "613_resenc": 4, "679_resenc": 5,
            "679_resenc_final": 0,
        },
        "mr_skip_lcc_labels": (850,),
    },
    "submission3": {
        "description": "CT dual-ResEnc group balance; MR equal-six; keep all MR PA components",
        "ct_weights": {
            "613_basic": 2, "615_basic": 2, "611_basic": 2,
            "613_da5": 3, "615_da5": 3, "613_resenc": 3,
            "611_resenc": 3,
        },
        "mr_weights": {
            "613_basic": 1, "615_basic": 1, "613_da5": 1,
            "615_da5": 1, "613_resenc": 1, "679_resenc": 1,
            "679_resenc_final": 0,
        },
        "mr_skip_lcc_labels": (850,),
    },
    "submission1_snapshot": {
        "description": (
            "CT high020; MR Dataset679 best/final 50/50 at total share 0.25; "
            "keep all MR PA components"
        ),
        "ct_weights": {
            "613_basic": 2, "615_basic": 2, "611_basic": 2,
            "613_da5": 2, "615_da5": 2, "613_resenc": 2,
            "611_resenc": 3,
        },
        "mr_weights": {
            "613_basic": 6, "615_basic": 6, "613_da5": 6,
            "615_da5": 6, "613_resenc": 6,
            "679_resenc": 5, "679_resenc_final": 5,
        },
        "mr_skip_lcc_labels": (850,),
    },
}


def normalized_weights(variant: str, modality: str) -> dict[str, float]:
    raw = VARIANTS[variant][f"{modality}_weights"]
    total = float(sum(raw.values()))
    return {name: float(value) / total for name, value in raw.items()}
