"""Utility script to download Chronos-Bolt checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
from urllib import request

DEFAULT_URL = (
    "https://huggingface.co/amazon/chronos-bolt-base/resolve/main/pytorch_model.bin?download=1"
)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with a progress indicator."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as response, open(output_path, "wb") as dst:
        total_str = response.headers.get("Content-Length")
        total = int(total_str) if total_str is not None else None
        downloaded = 0
        chunk_size = 1 << 20
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                sys.stdout.write(f"\rDownloaded {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\rDownloaded {downloaded/1e6:.1f}MB")
            sys.stdout.flush()
    sys.stdout.write("\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Where to store checkpoints")
    parser.add_argument(
        "--filename",
        type=str,
        default="chronos_bolt_base.pt",
        help="Filename for the downloaded checkpoint",
    )
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=DEFAULT_URL,
        help="URL hosting the Chronos-Bolt weights",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_path = args.output_dir / args.filename
    print(f"Downloading Chronos-Bolt checkpoint to {output_path}")
    download_file(args.checkpoint_url, output_path)
    print("Download complete. Update configs/model/chronos_bolt.yaml with this path.")


if __name__ == "__main__":
    main()
