"""
Download and extract the UCF-101 dataset.

UCF-101 is distributed as a .rar archive (~6.5 GB).
After extraction, videos are organized as:
    data/UCF-101/{CategoryName}/{video_file}.avi

Usage:
    python scripts/download_ucf101.py [--data-dir data/]
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"


class _DownloadProgress:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
        downloaded = block_num * block_size
        self.pbar.update(block_size)
        if downloaded >= total_size and self.pbar:
            self.pbar.close()


def download(url: str, dest: Path) -> None:
    """Download with progress bar."""
    if dest.exists():
        print(f"Archive already exists at {dest}, skipping download.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCF-101 from {url}...")
    print("(This is ~6.5 GB and may take a while)")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_DownloadProgress())
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Try manually: wget {url} -O {dest}")
        sys.exit(1)


def extract_rar(archive: Path, dest: Path) -> None:
    """Extract .rar archive."""
    dest.mkdir(parents=True, exist_ok=True)
    # Try system unrar first
    try:
        subprocess.run(
            ["unrar", "x", "-o+", str(archive), str(dest)],
            check=True,
            capture_output=True,
        )
        print(f"Extracted to {dest}")
        return
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as e:
        print(f"ERROR: unrar failed: {e.stderr.decode() if e.stderr else e}")
        sys.exit(1)

    # Try unar (macOS)
    try:
        subprocess.run(
            ["unar", "-o", str(dest), str(archive)],
            check=True,
            capture_output=True,
        )
        print(f"Extracted to {dest}")
        return
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as e:
        print(f"ERROR: unar failed: {e.stderr.decode() if e.stderr else e}")
        sys.exit(1)

    print("ERROR: No RAR extractor found.")
    print("Install one of:")
    print("  brew install unrar")
    print("  brew install unar")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download UCF-101 dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    archive_path = args.data_dir / "UCF101.rar"
    download(UCF101_URL, archive_path)
    extract_rar(archive_path, args.data_dir)

    ucf_dir = args.data_dir / "UCF-101"
    if ucf_dir.exists():
        n_categories = sum(1 for p in ucf_dir.iterdir() if p.is_dir())
        print(f"UCF-101 ready: {n_categories} categories in {ucf_dir}")
    else:
        print(f"WARNING: Expected {ucf_dir} but not found. Check extraction.")


if __name__ == "__main__":
    main()
