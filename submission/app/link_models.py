from __future__ import annotations

import os
from pathlib import Path

from config import SOURCES


def main() -> None:
    bundle_root = Path("/models/bundle")
    result_root = Path("/models/nnUNet_results")
    result_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for source in SOURCES:
        relative = source.model_relative_path
        if relative in seen:
            continue
        seen.add(relative)
        target = bundle_root / source.model_bundle_name
        if not target.is_dir():
            raise FileNotFoundError(target)
        link = result_root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        os.symlink(target, link, target_is_directory=True)
        print(f"linked {link} -> {target}")


if __name__ == "__main__":
    main()
