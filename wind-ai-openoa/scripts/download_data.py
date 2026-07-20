#!/usr/bin/env python3
"""Download and extract the official OpenOA La Haute Borne example data."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw" / "la_haute_borne.zip"
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "la_haute_borne"
DATA_URL = (
    "https://github.com/NatLabRockies/OpenOA/raw/refs/heads/main/"
    "examples/data/la_haute_borne.zip"
)
EXPECTED_SIZE = 36_762_939
REQUIRED_FILES = {
    "la-haute-borne-data-2014-2015.csv",
    "plant_data.csv",
    "merra2_la_haute_borne.csv",
    "era5_wind_la_haute_borne.csv",
    "la-haute-borne_asset_table.csv",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "wind-ai-openoa/0.1 (+OpenOA example downloader)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False, suffix=".part"
            ) as temp_file:
                temp_path = Path(temp_file.name)
                shutil.copyfileobj(response, temp_file)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"数据下载失败：{exc}") from exc

    temp_path.replace(destination)


def _validate_zip(zip_path: Path) -> None:
    if zip_path.stat().st_size < 30_000_000:
        raise RuntimeError(
            f"下载文件过小（{zip_path.stat().st_size} bytes），可能不是完整数据包。"
        )
    if zip_path.stat().st_size != EXPECTED_SIZE:
        print(
            "提示：文件大小与项目记录值不同，可能是官方数据包发生了更新；"
            "将继续执行 ZIP 完整性校验。",
            file=sys.stderr,
        )
    try:
        with ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"ZIP CRC 校验失败：{bad_member}")
    except BadZipFile as exc:
        raise RuntimeError("下载内容不是有效的 ZIP 文件。") from exc


def _extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(destination)

    missing = sorted(name for name in REQUIRED_FILES if not (destination / name).is_file())
    if missing:
        raise RuntimeError(f"解压后缺少预期文件：{missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="重新下载并覆盖已有数据")
    parser.add_argument("--keep-zip", action="store_true", help="解压完成后保留 ZIP 文件")
    args = parser.parse_args()

    if DEFAULT_EXTRACT_DIR.is_dir() and not args.force:
        missing = [name for name in REQUIRED_FILES if not (DEFAULT_EXTRACT_DIR / name).is_file()]
        if not missing:
            print(f"数据已存在：{DEFAULT_EXTRACT_DIR}")
            return 0

    if args.force:
        shutil.rmtree(DEFAULT_EXTRACT_DIR, ignore_errors=True)
        DEFAULT_ZIP.unlink(missing_ok=True)

    if not DEFAULT_ZIP.is_file():
        print(f"正在下载：{DATA_URL}")
        _download(DATA_URL, DEFAULT_ZIP)

    print("正在校验 ZIP 文件……")
    _validate_zip(DEFAULT_ZIP)
    print(f"正在解压到：{DEFAULT_EXTRACT_DIR}")
    _extract(DEFAULT_ZIP, DEFAULT_EXTRACT_DIR)

    if not args.keep_zip:
        DEFAULT_ZIP.unlink(missing_ok=True)

    print("数据准备完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
