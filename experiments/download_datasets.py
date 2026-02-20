#!/usr/bin/env python3
"""
Dataset Downloader for HistoLab Experiments

Downloads and prepares the three benchmark datasets:
- NCT-CRC-HE-100K (CRC): 9-class colorectal tissue classification
- PatchCamelyon (PCam): Binary tumor detection in lymph node sections
- BACH (ICIAR 2018): 4-class breast cancer histology

Usage:
    # Download all datasets
    python experiments/download_datasets.py --all --data-dir data/datasets

    # Download specific datasets
    python experiments/download_datasets.py --datasets crc pcam --data-dir data/datasets

    # Verify existing datasets only (no download)
    python experiments/download_datasets.py --verify-only --data-dir data/datasets
"""

import sys
import argparse
import subprocess
import zipfile
import tarfile
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------

DATASET_SPECS = {
    "crc": {
        "display_name": "NCT-CRC-HE-100K",
        "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        "num_classes": 9,
        "expected_total": 100000,  # approximate
        "extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg"],
        "download_methods": [
            {
                "name": "zenodo",
                "url": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip",
                "type": "wget_zip",
            },
            {
                "name": "kaggle",
                "dataset": "gpreda/nct-crc-he-100k-subset",
                "type": "kaggle",
            },
        ],
    },
    "pcam": {
        "display_name": "PatchCamelyon",
        "classes": ["Normal", "Tumor"],
        "num_classes": 2,
        "expected_total": 200000,  # approximate (subset)
        "extensions": [".png", ".jpg", ".jpeg", ".tif"],
        "download_methods": [
            {
                "name": "kaggle",
                "dataset": "andrewmvd/metastatic-tissue-classification-patchcamelyon",
                "type": "kaggle",
            },
        ],
    },
    "bach": {
        "display_name": "BACH (ICIAR 2018)",
        "classes": ["Normal", "Benign", "InSitu", "Invasive"],
        "num_classes": 4,
        "expected_total": 400,
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "download_methods": [
            {
                "name": "kaggle",
                "dataset": "truthisneverlinear/bach-breast-cancer-histology-images",
                "type": "kaggle",
            },
        ],
    },
}


def check_kaggle_available() -> bool:
    """Check if kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_with_wget(url: str, output_dir: Path, dataset_name: str) -> bool:
    """Download a dataset via wget and extract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    download_path = output_dir / filename

    logger.info(f"Downloading {dataset_name} from {url}...")

    try:
        subprocess.run(
            ["wget", "-c", "--progress=bar:force", "-O", str(download_path), url],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"wget failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("wget not found. Install with: sudo apt install wget")
        return False

    # Extract
    if filename.endswith(".zip"):
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(output_dir)
    elif filename.endswith((".tar.gz", ".tgz")):
        logger.info(f"Extracting {filename}...")
        with tarfile.open(download_path, "r:gz") as tf:
            tf.extractall(output_dir)

    # Clean up archive
    download_path.unlink(missing_ok=True)

    # CRC-specific: the zip extracts to NCT-CRC-HE-100K/, move contents up
    extracted_dir = output_dir / "NCT-CRC-HE-100K"
    target_dir = output_dir / dataset_name
    if extracted_dir.exists() and extracted_dir != target_dir:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        extracted_dir.rename(target_dir)

    return True


def download_with_kaggle(dataset_id: str, output_dir: Path, dataset_name: str) -> bool:
    """Download a dataset via Kaggle CLI."""
    if not check_kaggle_available():
        logger.error(
            "Kaggle CLI not available. Install with:\n"
            "  pip install kaggle\n"
            "Then configure API key: https://www.kaggle.com/docs/api"
        )
        return False

    target_dir = output_dir / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} from Kaggle: {dataset_id}...")

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "-p", str(target_dir),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle download failed: {e}")
        return False

    return True


def verify_dataset(data_dir: Path, dataset_name: str) -> dict:
    """
    Verify a downloaded dataset has the expected directory structure.

    Returns dict with verification results.
    """
    spec = DATASET_SPECS[dataset_name]
    dataset_path = data_dir / dataset_name
    result = {
        "name": dataset_name,
        "display_name": spec["display_name"],
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "valid": False,
        "classes_found": [],
        "classes_missing": [],
        "samples_per_class": {},
        "total_samples": 0,
    }

    if not dataset_path.exists():
        return result

    # Check for class subdirectories
    for class_name in spec["classes"]:
        class_dir = dataset_path / class_name
        if class_dir.is_dir():
            # Count image files
            count = 0
            for ext in spec["extensions"]:
                count += len(list(class_dir.glob(f"*{ext}")))
            result["classes_found"].append(class_name)
            result["samples_per_class"][class_name] = count
            result["total_samples"] += count
        else:
            result["classes_missing"].append(class_name)

    # Valid if all classes present and have at least some samples
    result["valid"] = (
        len(result["classes_missing"]) == 0
        and result["total_samples"] > 0
    )

    return result


def download_dataset(dataset_name: str, data_dir: Path, force: bool = False) -> bool:
    """Download and verify a single dataset."""
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        logger.error(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_SPECS.keys())}")
        return False

    # Check if already downloaded
    if not force:
        verification = verify_dataset(data_dir, dataset_name)
        if verification["valid"]:
            logger.info(
                f"{spec['display_name']} already downloaded and verified: "
                f"{verification['total_samples']} samples in {len(verification['classes_found'])} classes"
            )
            return True

    # Try download methods in order
    for method in spec["download_methods"]:
        logger.info(f"Trying download method: {method['name']}...")

        success = False
        if method["type"] == "wget_zip":
            success = download_with_wget(method["url"], data_dir, dataset_name)
        elif method["type"] == "kaggle":
            success = download_with_kaggle(method["dataset"], data_dir, dataset_name)

        if success:
            # Verify after download
            verification = verify_dataset(data_dir, dataset_name)
            if verification["valid"]:
                logger.info(
                    f"Successfully downloaded {spec['display_name']}: "
                    f"{verification['total_samples']} samples"
                )
                return True
            else:
                logger.warning(
                    f"Download completed but verification failed for {dataset_name}. "
                    f"Missing classes: {verification['classes_missing']}"
                )
                # Try to fix directory structure
                _try_fix_structure(data_dir, dataset_name, spec)
                verification = verify_dataset(data_dir, dataset_name)
                if verification["valid"]:
                    logger.info(f"Fixed directory structure for {dataset_name}")
                    return True

    logger.error(f"All download methods failed for {dataset_name}")
    _print_manual_instructions(dataset_name, data_dir)
    return False


def _try_fix_structure(data_dir: Path, dataset_name: str, spec: dict):
    """Try to fix common directory structure issues after download."""
    dataset_path = data_dir / dataset_name

    # Common issue: data is in a subdirectory
    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            # Check if this subdir contains the expected classes
            has_classes = any(
                (subdir / cn).is_dir() for cn in spec["classes"]
            )
            if has_classes:
                logger.info(f"Found classes in subdirectory: {subdir.name}, moving up...")
                for item in subdir.iterdir():
                    target = dataset_path / item.name
                    if not target.exists():
                        item.rename(target)
                # Remove now-empty subdir
                if not list(subdir.iterdir()):
                    subdir.rmdir()
                break


def _print_manual_instructions(dataset_name: str, data_dir: Path):
    """Print manual download instructions."""
    spec = DATASET_SPECS[dataset_name]
    target = data_dir / dataset_name

    print(f"\n{'='*60}")
    print(f"MANUAL DOWNLOAD REQUIRED: {spec['display_name']}")
    print(f"{'='*60}")

    for method in spec["download_methods"]:
        if method["type"] == "wget_zip":
            print(f"\nOption ({method['name']}):")
            print(f"  wget {method['url']}")
            print(f"  unzip {method['url'].split('/')[-1]} -d {target}")
        elif method["type"] == "kaggle":
            print(f"\nOption ({method['name']}):")
            print(f"  kaggle datasets download -d {method['dataset']} -p {target} --unzip")

    print(f"\nExpected structure:")
    print(f"  {target}/")
    for cn in spec["classes"]:
        print(f"    {cn}/")
        print(f"      *.png / *.tif / ...")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify histopathology datasets for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=list(DATASET_SPECS.keys()),
        help="Specific datasets to download",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets (crc, pcam, bach)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets",
        help="Base directory for datasets (default: data/datasets)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if datasets exist",
    )

    args = parser.parse_args()

    if not args.datasets and not args.all:
        parser.error("Specify --datasets or --all")

    datasets = list(DATASET_SPECS.keys()) if args.all else args.datasets
    data_dir = Path(args.data_dir)

    print(f"\n{'='*60}")
    print("HistoLab Dataset Manager")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Mode: {'verify only' if args.verify_only else 'download + verify'}")
    print(f"{'='*60}\n")

    results = {}

    for ds_name in datasets:
        if args.verify_only:
            result = verify_dataset(data_dir, ds_name)
            results[ds_name] = result
        else:
            success = download_dataset(ds_name, data_dir, force=args.force)
            result = verify_dataset(data_dir, ds_name)
            result["download_success"] = success
            results[ds_name] = result

    # Print summary table
    print(f"\n{'='*60}")
    print("DATASET VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Status':<10} {'Classes':<10} {'Samples':<12} {'Path'}")
    print(f"{'-'*60}")

    all_ok = True
    for ds_name, result in results.items():
        status = "OK" if result["valid"] else "MISSING"
        if not result["valid"]:
            all_ok = False
        classes_str = f"{len(result['classes_found'])}/{DATASET_SPECS[ds_name]['num_classes']}"
        print(f"{ds_name:<12} {status:<10} {classes_str:<10} {result['total_samples']:<12} {result['path']}")

        if result["valid"] and result["samples_per_class"]:
            for cn, count in sorted(result["samples_per_class"].items()):
                print(f"  {cn}: {count}")

    print(f"{'-'*60}")

    if all_ok:
        print("All datasets ready!")
        return 0
    else:
        missing = [ds for ds, r in results.items() if not r["valid"]]
        print(f"Missing/invalid datasets: {', '.join(missing)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
