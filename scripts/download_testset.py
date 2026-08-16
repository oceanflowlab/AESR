#!/usr/bin/env python3
"""Download and verify the official IPVG 2026 Track 1 test set."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://github.com/HiDream-ai/ipvg-challenge-2026.github.io/"
    "releases/download/testset/IPVG2026-Test-Track1.zip"
)
ARCHIVE_SHA256 = "b6221e019682d508d01552a41bc75cac4c653986a36662ceaaa69896d3c38c08"
DATASET_NAME = "IPVG2026-Test-Track1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "AESR-dataset-downloader"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def validate_members(archive: zipfile.ZipFile, data_root: Path) -> None:
    root = data_root.resolve()
    for member in archive.infolist():
        target = (data_root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe archive path: {member.filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Extraction root (default: data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    dataset_dir = data_root / DATASET_NAME
    archive_path = data_root / f"{DATASET_NAME}.zip"

    if (dataset_dir / "eval.json").is_file() and (dataset_dir / "images").is_dir():
        print(f"Dataset already available at {dataset_dir}")
        return 0

    data_root.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file() or sha256(archive_path) != ARCHIVE_SHA256:
        print(f"Downloading {DATASET_URL}")
        download(DATASET_URL, archive_path)

    actual_sha256 = sha256(archive_path)
    if actual_sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {archive_path}: expected {ARCHIVE_SHA256}, got {actual_sha256}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        validate_members(archive, data_root)
        archive.extractall(data_root)

    if not (dataset_dir / "eval.json").is_file() or not (dataset_dir / "images").is_dir():
        raise RuntimeError(f"Unexpected dataset layout in {dataset_dir}")

    archive_path.unlink(missing_ok=True)
    print(f"Dataset extracted to {dataset_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
