#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi

from config import SOURCES, VARIANTS, normalized_weights
from n4 import correct_all


NNUNET_TO_SUBMISSION = {0: 0, 1: 205, 2: 420, 3: 500, 4: 550, 5: 600, 6: 820, 7: 850}
SUBMISSION_LABELS = {0, 205, 420, 500, 550, 600, 820, 850}
FOREGROUND_LABELS = (205, 420, 500, 550, 600, 820, 850)
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
PROBABILITY_KEYS = ("probabilities", "softmax", "data")
PROBABILITY_CHANNELS = 8
FLOAT32_BYTES = 4


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_report() -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
        "nnunetv2": importlib.metadata.version("nnunetv2"),
        "numpy": np.__version__,
        "simpleitk": sitk.Version_VersionString(),
        "scipy": importlib.metadata.version("scipy"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
    return report


def verify_models() -> dict[str, object]:
    result_root = Path(os.environ["nnUNet_results"])
    manifest_path = Path("/models/model_bundle_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Model bundle manifest is not complete: {manifest_path}")
    manifest_models = {model["name"]: model for model in manifest.get("models", [])}
    expected_names = {source.model_bundle_name for source in SOURCES}
    if set(manifest_models) != expected_names:
        raise RuntimeError(
            "Model bundle names differ from the frozen source contract: "
            f"found={sorted(manifest_models)} expected={sorted(expected_names)}"
        )
    expected_checkpoint_contracts: dict[str, set[tuple[int, str]]] = {}
    for source in SOURCES:
        contracts = expected_checkpoint_contracts.setdefault(source.model_bundle_name, set())
        contracts.update((fold, source.checkpoint_name) for fold in source.folds)
    expected_checkpoint_count = sum(
        len(contracts) for contracts in expected_checkpoint_contracts.values()
    )
    if manifest.get("model_count") != len(expected_names):
        raise RuntimeError(f"Unexpected model count in {manifest_path}")
    if manifest.get("checkpoint_count") != expected_checkpoint_count:
        raise RuntimeError(f"Unexpected checkpoint count in {manifest_path}")

    rows = []
    for bundle_name in sorted(expected_names):
        sources = [source for source in SOURCES if source.model_bundle_name == bundle_name]
        relative_paths = {source.model_relative_path for source in sources}
        fold_contracts = {source.folds for source in sources}
        if len(relative_paths) != 1 or len(fold_contracts) != 1:
            raise RuntimeError(f"Inconsistent source aliases for model bundle {bundle_name}")
        model_dir = result_root / relative_paths.pop()
        if not model_dir.is_dir():
            raise NotADirectoryError(model_dir)
        model_manifest = manifest_models[bundle_name]
        folds = fold_contracts.pop()
        if tuple(model_manifest.get("folds", ())) != folds:
            raise RuntimeError(
                f"Fold contract changed for {bundle_name}: "
                f"{model_manifest.get('folds')} vs {list(folds)}"
            )
        for filename in ("dataset.json", "plans.json"):
            metadata = model_dir / filename
            if not metadata.is_file():
                raise FileNotFoundError(metadata)
            expected_hash = model_manifest["metadata_sha256"][filename]
            actual_hash = sha256_file(metadata)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Metadata hash mismatch for {bundle_name}/{filename}: "
                    f"{actual_hash} vs {expected_hash}"
                )
        checkpoint_manifest = {
            (
                int(row["fold"]),
                str(row.get("checkpoint_name") or Path(row["destination"]).name),
            ): row
            for row in model_manifest.get("checkpoints", [])
        }
        expected_rows = expected_checkpoint_contracts[bundle_name]
        if set(checkpoint_manifest) != expected_rows:
            raise RuntimeError(
                f"Checkpoint rows changed for {bundle_name}: "
                f"found={sorted(checkpoint_manifest)} expected={sorted(expected_rows)}"
            )
        checkpoint_hashes = {}
        for fold, checkpoint_name in sorted(expected_rows):
            checkpoint = model_dir / f"fold_{fold}" / checkpoint_name
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(checkpoint)
            actual_hash = sha256_file(checkpoint)
            expected_hash = checkpoint_manifest[(fold, checkpoint_name)]["destination_sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Checkpoint hash mismatch for {bundle_name}/fold_{fold}/{checkpoint_name}: "
                    f"{actual_hash} vs {expected_hash}"
                )
            checkpoint_hashes[f"{fold}:{checkpoint_name}"] = actual_hash
        rows.append(
            {
                "model_bundle": bundle_name,
                "source_aliases": [source.name for source in sources],
                "model_dir": str(model_dir),
                "folds": list(folds),
                "checkpoint_sha256": checkpoint_hashes,
            }
        )
    return {
        "models": rows,
        "model_count": len(expected_names),
        "checkpoint_count": expected_checkpoint_count,
        "manifest_sha256": sha256_file(manifest_path),
    }


def discover_inputs(root: Path, folder: str, prefix: str, expected: int) -> list[tuple[str, Path]]:
    directory = root / folder
    paths = sorted(path for path in directory.glob(f"{prefix}*_image.nii.gz") if not path.name.startswith("._"))
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} inputs in {directory}, found {len(paths)}")
    rows = []
    for path in paths:
        case_id = path.name[: -len("_image.nii.gz")]
        if not case_id.startswith(prefix):
            raise RuntimeError(f"Bad input case name: {path.name}")
        rows.append((case_id, path.resolve()))
    if len({case_id for case_id, _ in rows}) != len(rows):
        raise RuntimeError(f"Duplicate case IDs in {directory}")
    return rows


def symlink_input(case_id: str, source: Path, destination_dir: Path) -> None:
    destination = destination_dir / f"{case_id}_0000.nii.gz"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source, destination)


def stage_inputs(
    input_root: Path,
    work: Path,
    expected_ct: int,
    expected_mr: int,
    n4_workers: int,
) -> tuple[dict[str, Path], dict[str, list[tuple[str, Path]]]]:
    cases = {
        "ct": discover_inputs(input_root, "ct_test", "CaseCTTest", expected_ct),
        "mr": discover_inputs(input_root, "mr_test", "CaseMRTest", expected_mr),
    }
    staged = {key: work / "inputs" / key for key in ("raw_ct", "raw_mr", "n4_mr")}
    for directory in staged.values():
        directory.mkdir(parents=True)
    for case_id, path in cases["ct"]:
        symlink_input(case_id, path, staged["raw_ct"])
    n4_jobs = []
    for case_id, path in cases["mr"]:
        symlink_input(case_id, path, staged["raw_mr"])
        n4_jobs.append((path, staged["n4_mr"] / f"{case_id}_0000.nii.gz"))
    log(f"Running image-only N4 correction for {len(n4_jobs)} MR cases with {n4_workers} workers")
    corrected = correct_all(n4_jobs, n4_workers)
    if len(corrected) != expected_mr:
        raise RuntimeError(f"N4 produced {len(corrected)}/{expected_mr} cases")
    return staged, cases


def estimate_probability_bytes(path: Path) -> int:
    image = sitk.ReadImage(str(path))
    voxels = int(np.prod(np.asarray(image.GetSize(), dtype=np.int64), dtype=np.int64))
    return voxels * PROBABILITY_CHANNELS * FLOAT32_BYTES


def partition_estimates(
    estimates: list[tuple[str, Path, int]], max_bytes: int
) -> list[list[tuple[str, Path, int]]]:
    if max_bytes <= 0:
        raise ValueError("Batch probability budget must be positive")
    batches: list[list[tuple[str, Path, int]]] = []
    current: list[tuple[str, Path, int]] = []
    current_bytes = 0
    for row in estimates:
        row_bytes = row[2]
        if row_bytes <= 0:
            raise ValueError(f"Invalid probability estimate for {row[0]}: {row_bytes}")
        if current and current_bytes + row_bytes > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += row_bytes
    if current:
        batches.append(current)
    return batches


def plan_batches(
    cases: list[tuple[str, Path]], max_bytes: int
) -> list[list[tuple[str, Path, int]]]:
    estimates = [
        (case_id, path, estimate_probability_bytes(path)) for case_id, path in cases
    ]
    return partition_estimates(estimates, max_bytes)


def staged_key_for(source, modality: str) -> str:
    if modality == "ct":
        return "raw_ct"
    if modality != "mr":
        raise RuntimeError(f"Unknown modality: {modality}")
    return "n4_mr" if source.input_contract == "n4_both" else "raw_mr"


def materialize_batch_input(
    source_dir: Path,
    batch: list[tuple[str, Path, int]],
    destination: Path,
) -> Path:
    destination.mkdir(parents=True)
    for case_id, _, _ in batch:
        source = source_dir / f"{case_id}_0000.nii.gz"
        if not source.is_file():
            raise FileNotFoundError(source)
        os.symlink(source.resolve(), destination / source.name)
    return destination


def run_prediction(source, input_dir: Path, output_dir: Path, device: str) -> None:
    command = [
        "nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(source.dataset_id), "-c", "3d_fullres", "-f",
        *[str(fold) for fold in source.folds], "-tr", source.trainer,
        "-p", source.plans, "-chk", source.checkpoint_name,
        "-npp", os.environ.get("CARE_NPP", "1"),
        "-nps", os.environ.get("CARE_NPS", "1"),
        "-device", device, "--disable_progress_bar", "--save_probabilities",
    ]
    log(f"Predicting {source.name}: {' '.join(command)}")
    subprocess.run(command, check=True)


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        for key in PROBABILITY_KEYS:
            if key in payload:
                array = payload[key]
                if array.ndim != 4:
                    raise RuntimeError(f"{path} has probability shape {array.shape}, expected CZYX")
                return array.astype(np.float32, copy=False)
        raise RuntimeError(f"{path} lacks a known probability key; found {sorted(payload.files)}")


def modality_for_case(case_id: str) -> str:
    if case_id.startswith("CaseCTTest"):
        return "ct"
    if case_id.startswith("CaseMRTest"):
        return "mr"
    raise RuntimeError(f"Unknown test case ID: {case_id}")


def accumulate_source(
    source,
    source_dir: Path,
    weights: dict[str, dict[str, float]],
    expected: dict[str, int],
    accumulator_root: Path,
    reference_root: Path,
) -> list[dict[str, object]]:
    rows = []
    counts = {"ct": 0, "mr": 0}
    for probability_path in sorted(source_dir.glob("Case*Test*.npz")):
        case_id = probability_path.stem
        modality = modality_for_case(case_id)
        if modality not in source.modalities:
            raise RuntimeError(f"{source.name} unexpectedly predicted {case_id}")
        coefficient = np.float32(weights[modality][source.name])
        probabilities = load_probabilities(probability_path)
        accumulator_path = accumulator_root / modality / f"{case_id}.npy"
        accumulator_path.parent.mkdir(parents=True, exist_ok=True)
        if accumulator_path.exists():
            accumulator = np.lib.format.open_memmap(accumulator_path, mode="r+")
            if accumulator.shape != probabilities.shape:
                raise RuntimeError(f"Probability shape changed for {case_id}: {probabilities.shape} vs {accumulator.shape}")
        else:
            accumulator = np.lib.format.open_memmap(accumulator_path, mode="w+", dtype=np.float32, shape=probabilities.shape)
            accumulator[:] = 0
        accumulator += probabilities * coefficient
        accumulator.flush()
        del accumulator

        reference = reference_root / modality / f"{case_id}.nii.gz"
        reference.parent.mkdir(parents=True, exist_ok=True)
        source_reference = source_dir / f"{case_id}.nii.gz"
        if not source_reference.is_file():
            raise FileNotFoundError(source_reference)
        if not reference.exists():
            shutil.copy2(source_reference, reference)
        counts[modality] += 1
        rows.append({"source": source.name, "case_id": case_id, "modality": modality, "normalized_weight": float(coefficient), "probability_shape": list(probabilities.shape)})
        del probabilities

    unexpected = {modality: count for modality, count in counts.items() if modality not in expected and count}
    if unexpected:
        raise RuntimeError(f"{source.name} produced unexpected modalities: {unexpected}")
    for modality, expected_count in expected.items():
        if counts[modality] != expected_count:
            raise RuntimeError(f"{source.name} produced {counts[modality]}/{expected_count} {modality} cases")
    shutil.rmtree(source_dir)
    return rows


def write_raw_ensemble(case_id: str, accumulator: Path, reference: Path, destination: Path) -> dict[str, object]:
    probabilities = np.lib.format.open_memmap(accumulator, mode="r")
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"Non-finite ensemble probabilities for {case_id}")
    mass_error = float(np.max(np.abs(probabilities.sum(axis=0, dtype=np.float32) - 1.0)))
    if mass_error > 5e-3:
        raise RuntimeError(f"Probability mass error for {case_id}: {mass_error}")
    labels = np.argmax(probabilities, axis=0).astype(np.uint8)
    reference_image = sitk.ReadImage(str(reference))
    output_image = sitk.GetImageFromArray(labels)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(destination), useCompression=True)
    return {"case_id": case_id, "mass_error": mass_error, "probability_shape": list(probabilities.shape)}


def remap_labels(array: np.ndarray) -> np.ndarray:
    labels = {int(value) for value in np.unique(array)}
    if not labels.issubset(NNUNET_TO_SUBMISSION):
        raise RuntimeError(f"Unexpected nnU-Net labels: {sorted(labels)}")
    result = np.zeros(array.shape, dtype=np.uint16)
    for source, destination in NNUNET_TO_SUBMISSION.items():
        if source:
            result[array == source] = destination
    return result


def postprocess_case(raw_path: Path, destination: Path, skip_lcc: set[int]) -> list[dict[str, object]]:
    image = sitk.ReadImage(str(raw_path))
    remapped = remap_labels(sitk.GetArrayFromImage(image))
    cleaned = np.zeros(remapped.shape, dtype=np.uint16)
    rows = []
    for label in FOREGROUND_LABELS:
        mask = remapped == label
        labeled, components = ndi.label(mask, structure=CONNECTIVITY_26)
        sizes = np.bincount(labeled.ravel())[1:] if components else np.asarray([], dtype=np.int64)
        before = int(mask.sum())
        if label in skip_lcc or components <= 1:
            keep = mask
            rule = "skip_lcc_keep_all" if label in skip_lcc else "unchanged"
        else:
            keep = labeled == int(np.argmax(sizes)) + 1
            rule = "largest_component_only"
        cleaned[keep] = label
        after = int(keep.sum())
        rows.append({"case_id": raw_path.stem.replace(".nii", ""), "label": label, "components_before": int(components), "voxels_before": before, "voxels_after": after, "removed_voxels": before - after, "rule": rule})
    output = sitk.GetImageFromArray(cleaned)
    output.CopyInformation(image)
    output = sitk.Cast(output, sitk.sitkUInt16)
    sitk.WriteImage(output, str(destination), useCompression=True)
    return rows


def validate_output(input_path: Path, output_path: Path) -> dict[str, object]:
    source = sitk.ReadImage(str(input_path))
    prediction = sitk.ReadImage(str(output_path))
    if source.GetSize() != prediction.GetSize():
        raise RuntimeError(f"Shape mismatch: {input_path} vs {output_path}")
    for name, first, second in (
        ("spacing", source.GetSpacing(), prediction.GetSpacing()),
        ("origin", source.GetOrigin(), prediction.GetOrigin()),
        ("direction", source.GetDirection(), prediction.GetDirection()),
    ):
        if not np.allclose(first, second, rtol=0, atol=1e-6):
            raise RuntimeError(f"{name} mismatch: {input_path} vs {output_path}")
    array = sitk.GetArrayFromImage(prediction)
    labels = {int(value) for value in np.unique(array)}
    if not labels.issubset(SUBMISSION_LABELS):
        raise RuntimeError(f"Unexpected output labels in {output_path}: {sorted(labels)}")
    if prediction.GetPixelID() != sitk.sitkUInt16:
        raise RuntimeError(f"Output is not UInt16: {output_path}")
    return {"input": str(input_path), "output": str(output_path), "size": list(source.GetSize()), "labels": sorted(labels), "pixel_type": prediction.GetPixelIDTypeAsString()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARE 2026 Whole Heart hidden-test inference")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="submission1_snapshot")
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/output"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    started = time.time()
    software = software_report()
    models = verify_models()
    log(json.dumps({"software": software, "variant": args.variant, "models": models["model_count"]}, sort_keys=True))
    if software["nnunetv2"] != "2.5.1":
        raise RuntimeError(f"Expected nnU-Net 2.5.1, found {software['nnunetv2']}")
    if args.preflight_only:
        log("Preflight complete")
        return

    device = os.environ.get("CARE_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This performance submission requires an NVIDIA GPU; run Docker with --gpus all")
    expected = {
        "ct": int(os.environ.get("CARE_EXPECTED_CT", "54")),
        "mr": int(os.environ.get("CARE_EXPECTED_MR", "36")),
    }
    n4_workers = max(1, int(os.environ.get("CARE_N4_WORKERS", "4")))
    batch_probability_gb = float(os.environ.get("CARE_BATCH_PROBABILITY_GB", "12"))
    if not np.isfinite(batch_probability_gb) or batch_probability_gb <= 0:
        raise RuntimeError(f"Invalid CARE_BATCH_PROBABILITY_GB: {batch_probability_gb}")
    batch_probability_bytes = int(batch_probability_gb * (1024**3))
    args.output_root.mkdir(parents=True, exist_ok=True)
    for folder in ("ct_test", "mr_test"):
        destination = args.output_root / folder
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()

    work = Path(os.environ.get("CARE_WORK_DIR", "/tmp/care_whs_test_work"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    staged, cases = stage_inputs(args.input_root, work, expected["ct"], expected["mr"], n4_workers)

    weights = {
        "ct": normalized_weights(args.variant, "ct"),
        "mr": normalized_weights(args.variant, "mr"),
    }
    accumulation_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    postprocess_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    mr_skip = set(VARIANTS[args.variant]["mr_skip_lcc_labels"])
    for modality in ("ct", "mr"):
        prefix = "CaseCTTest" if modality == "ct" else "CaseMRTest"
        output_folder = "ct_test" if modality == "ct" else "mr_test"
        skip_lcc = set() if modality == "ct" else mr_skip
        batches = plan_batches(cases[modality], batch_probability_bytes)
        log(
            f"Planned {len(batches)} {modality.upper()} batches with "
            f"{batch_probability_gb:g} GiB probability budget"
        )
        modality_sources = [
            source
            for source in SOURCES
            if modality in source.modalities and weights[modality][source.name] > 0
        ]
        for batch_index, batch in enumerate(batches, start=1):
            batch_root = work / "batches" / modality / f"batch_{batch_index:03d}"
            batch_case_ids = [case_id for case_id, _, _ in batch]
            batch_estimated_bytes = sum(row[2] for row in batch)
            batch_rows.append(
                {
                    "modality": modality,
                    "batch": batch_index,
                    "case_ids": batch_case_ids,
                    "estimated_probability_bytes": batch_estimated_bytes,
                }
            )
            log(
                f"Starting {modality.upper()} batch {batch_index}/{len(batches)}: "
                f"cases={len(batch)} estimated_prob={batch_estimated_bytes / (1024**3):.2f} GiB"
            )
            input_dirs: dict[str, Path] = {}
            for source in modality_sources:
                staged_key = staged_key_for(source, modality)
                if staged_key not in input_dirs:
                    input_dirs[staged_key] = materialize_batch_input(
                        staged[staged_key], batch, batch_root / "inputs" / staged_key
                    )
                source_output = batch_root / "source" / source.name
                source_output.mkdir(parents=True)
                run_prediction(source, input_dirs[staged_key], source_output, device)
                accumulation_rows.extend(
                    accumulate_source(
                        source,
                        source_output,
                        weights,
                        {modality: len(batch)},
                        batch_root / "accumulators",
                        batch_root / "references",
                    )
                )

            raw_root = batch_root / "raw_ensemble"
            raw_root.mkdir()
            for case_id, input_path, _ in batch:
                accumulator = batch_root / "accumulators" / modality / f"{case_id}.npy"
                reference = batch_root / "references" / modality / f"{case_id}.nii.gz"
                raw_path = raw_root / f"{case_id}.nii.gz"
                ensemble_rows.append(write_raw_ensemble(case_id, accumulator, reference, raw_path))
                output_path = args.output_root / output_folder / f"{case_id}_pred.nii.gz"
                postprocess_rows.extend(postprocess_case(raw_path, output_path, skip_lcc))
                validation_rows.append(validate_output(input_path, output_path))
            if not args.keep_work:
                shutil.rmtree(batch_root)
        output_count = len(list((args.output_root / output_folder).glob(f"{prefix}*_pred.nii.gz")))
        if output_count != expected[modality]:
            raise RuntimeError(f"Output count mismatch for {modality}: {output_count}/{expected[modality]}")

    write_csv(args.output_root / "probability_accumulation.csv", accumulation_rows)
    write_csv(args.output_root / "ensemble_qc.csv", ensemble_rows)
    write_csv(args.output_root / "postprocess_qc.csv", postprocess_rows)
    manifest = {
        "status": "complete",
        "variant": args.variant,
        "description": VARIANTS[args.variant]["description"],
        "ct_weights": weights["ct"],
        "mr_weights": weights["mr"],
        "ct_postprocess": "per-label 26-connected largest component",
        "mr_postprocess": f"per-label 26-connected largest component; keep all labels {sorted(mr_skip)}",
        "software": software,
        "model_manifest_sha256": sha256_file(Path("/models/model_bundle_manifest.json")),
        "counts": expected,
        "batch_probability_budget_gib": batch_probability_gb,
        "batches": batch_rows,
        "validated_outputs": validation_rows,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"Completed {args.variant}: CT={expected['ct']} MR={expected['mr']} elapsed={manifest['elapsed_seconds']:.1f}s")
    if not args.keep_work:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
