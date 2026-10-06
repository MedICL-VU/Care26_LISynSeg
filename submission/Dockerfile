FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="CARE-Whole-Heart-VMHeart-Test"
LABEL org.opencontainers.image.description="Dataset679 best/final snapshot candidate for CARE 2026 hidden-test inference"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    nnUNet_raw=/models/nnUNet_raw \
    nnUNet_preprocessed=/models/nnUNet_preprocessed \
    nnUNet_results=/models/nnUNet_results \
    nnUNet_compile=False \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    OMP_NUM_THREADS=8 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 \
    PYTHONPATH=/workspace/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY app /workspace/app
RUN python - <<'PY'
from pathlib import Path
import shutil
import nnunetv2

destination = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "data_augmentation" / "nnUNetTrainer_CARE_DA5.py"
shutil.copy2("/workspace/app/nnUNetTrainer_CARE_DA5.py", destination)
print("installed", destination)
PY

COPY model_bundle /models/bundle
COPY model_bundle_manifest.json /models/model_bundle_manifest.json
RUN mkdir -p /models/nnUNet_raw /models/nnUNet_preprocessed /models/nnUNet_results \
    && python /workspace/app/link_models.py \
    && python -m py_compile /workspace/app/*.py

ENTRYPOINT ["python", "/workspace/app/entrypoint.py"]
CMD ["--variant", "submission1_snapshot"]
