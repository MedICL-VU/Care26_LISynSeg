from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _correct_one(job: tuple[str, str]) -> str:
    import SimpleITK as sitk

    source_name, destination_name = job
    source = Path(source_name)
    destination = Path(destination_name)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    image = sitk.ReadImage(str(source), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    if int(sitk.GetArrayViewFromImage(mask).sum()) == 0:
        raise RuntimeError(f"N4 Otsu mask is empty: {source}")
    shrink = [4] * image.GetDimension()
    small_image = sitk.Shrink(image, shrink)
    small_mask = sitk.Shrink(mask, shrink)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(corrected, str(destination), useCompression=True)
    return destination.name


def correct_all(jobs: list[tuple[Path, Path]], workers: int) -> list[str]:
    serialized = [(str(source), str(destination)) for source, destination in jobs]
    if workers <= 1:
        return [_correct_one(job) for job in serialized]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_correct_one, serialized))
