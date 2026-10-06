# CARE submission

Frozen nnU-Net v2 inference from the 2026-08-03 `submission1_snapshot` package: 8 model families, 41 checkpoints, about 9.6 GB of weights. nnU-Net is installed as a dependency; the inference code is in `app/`.

## Weights and run

Download the [original Docker archive with weights](https://drive.google.com/file/d/1kkk_x59ZrX4NR8WcuntI0RALQrTLUomR/view) (12.8 GB), then run from this folder:

```bash
bash prepare_weights.sh /path/to/CARE-WHS-VMHeart-submission1-snapshot-20260803.tar.gz
docker build --platform linux/amd64 -t care-lisynseg:submission .
docker run --rm --gpus all --ipc=host \
  -v /path/to/test:/input:ro -v /path/to/predictions:/output \
  care-lisynseg:submission
```

If you already have the extracted weights, use `bash prepare_weights.sh /path/to/model_bundle` instead. Weights stay in `model_bundle/` locally and are included in the built image. GitHub stores the code, download link, and checkpoint hashes.

For the original image without rebuilding, use `docker load --input <archive>` and run `care-whs-vmheart-test:submission1-snapshot-20260803` with the same mounts.

## Input and output

```text
/input/ct_test/CaseCTTest001_image.nii.gz
/input/mr_test/CaseMRTest001_image.nii.gz
/output/ct_test/CaseCTTest001_pred.nii.gz
/output/mr_test/CaseMRTest001_pred.nii.gz
```

The [CARE test layout](https://www.zmic.org.cn/care_2026/track_wholeheart/) expects 54 CT and 36 MR cases. Outputs retain input geometry and use labels `0,205,420,500,550,600,820,850`. CT uses per-label 26-connected largest components; MR retains every PA component. N4 is applied only to the MR inputs used by Dataset615. The snapshot splits Dataset679's MR contribution equally between best and final checkpoints.

Inference requires an NVIDIA GPU and runs offline. Scratch space defaults to a 12 GiB probability batch budget, with roughly 24 GiB for probabilities and accumulators plus preprocessing. Use `-e CARE_BATCH_PROBABILITY_GB=4` for smaller batches. `-e CARE_EXPECTED_CT=1 -e CARE_EXPECTED_MR=1` supports a two-case smoke run; append `--preflight-only` to verify software and all weight hashes without inference.

The original `submission1`, `submission2`, and `submission3` recipes remain available through `--variant`. Synthesis and anatomy perturbations are training operations; this folder preserves the frozen challenge ensemble.
