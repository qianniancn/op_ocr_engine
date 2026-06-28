#!/usr/bin/env python3
"""
Lightweight build script for op_ocr_engine.

Examples:
    python build.py
    python build.py -g vs2026 -a x64 -t Release
    python build.py -g vs2022 -a x64 --clean
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


GENERATORS = {
    "vs2022": {"cmake": "Visual Studio 17 2022"},
    "vs2026": {"cmake": "Visual Studio 18 2026"},
}

BUILD_TYPES = ("Debug", "Release", "RelWithDebInfo")
ARCHITECTURES = ("x86", "x64")
ARCH_TO_VS = {"x86": "Win32", "x64": "x64"}
NCNN_PACKAGE_ARCHITECTURES = ("x86", "x64", "arm64")
DEFAULT_NCNN_VERSION = "20260526"
DEFAULT_NCNN_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20260526/"
    "ncnn-20260526-windows-vs2022.zip"
)


def find_vswhere() -> Path | None:
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return vswhere if vswhere.is_file() else None


def find_latest_visual_studio_installation() -> Path | None:
    vswhere = find_vswhere()
    if vswhere is None:
        return None

    try:
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    installation_path = result.stdout.strip()
    if not installation_path:
        return None

    path = Path(installation_path)
    return path.resolve() if path.exists() else None


def detect_supported_visual_studio_generator() -> str | None:
    installation = find_latest_visual_studio_installation()
    if installation is None:
        return None

    version_hint = installation.parent.name.lower()
    version_map = {
        "2022": "vs2022",
        "18": "vs2026",
        "2026": "vs2026",
    }
    return version_map.get(version_hint)


def default_generator_key() -> str:
    return detect_supported_visual_studio_generator() or "vs2022"


def find_visual_studio_cmake() -> Path | None:
    direct = shutil.which("cmake")
    if direct:
        return Path(direct).resolve()

    installation = find_latest_visual_studio_installation()
    if installation is None:
        return None

    cmake = (
        installation
        / "Common7"
        / "IDE"
        / "CommonExtensions"
        / "Microsoft"
        / "CMake"
        / "CMake"
        / "bin"
        / "cmake.exe"
    )
    return cmake.resolve() if cmake.is_file() else None


def ensure_cmake_on_path() -> None:
    if shutil.which("cmake") is not None:
        return

    cmake = find_visual_studio_cmake()
    if cmake is None:
        print("[ERROR] Could not find cmake.exe. Please install Visual Studio CMake tools.")
        sys.exit(1)

    cmake_dir = str(cmake.parent)
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = cmake_dir if not current_path else cmake_dir + os.pathsep + current_path
    print(f"[INFO] Using Visual Studio bundled CMake: {cmake}")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"[RUN] {display}\n")
    result = subprocess.run([str(part) for part in cmd], cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_dir / path).resolve()


def ncnn_config_path(ncnn_root: Path, arch: str) -> Path:
    return ncnn_root / arch / "lib" / "cmake" / "ncnn" / "ncnnConfig.cmake"


def ncnn_config_version_path(ncnn_root: Path, arch: str) -> Path:
    return ncnn_root / arch / "lib" / "cmake" / "ncnn" / "ncnnConfigVersion.cmake"


def parse_cmake_set_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    pattern = re.compile(r"^\s*set\(\s*([A-Za-z0-9_]+)\s+\"?([^\"\)]+)\"?\s*\)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def detect_ncnn_info(ncnn_root: Path, arch: str) -> dict[str, str]:
    config_values = parse_cmake_set_values(ncnn_config_path(ncnn_root, arch))
    version_values = parse_cmake_set_values(ncnn_config_version_path(ncnn_root, arch))
    shared = config_values.get("NCNN_SHARED_LIB", "unknown")
    return {
        "version": config_values.get("NCNN_VERSION")
        or version_values.get("PACKAGE_VERSION")
        or DEFAULT_NCNN_VERSION,
        "vulkan": config_values.get("NCNN_VULKAN", "unknown"),
        "shared": shared,
        "linkage": "shared" if shared == "ON" else "static" if shared == "OFF" else "unknown",
    }


def package_filename_from_url(url: str) -> str:
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return filename or f"ncnn-{DEFAULT_NCNN_VERSION}-windows-vs2022.zip"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"[INFO] Using cached ncnn package: {destination}")
        return

    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    if temp_destination.exists():
        temp_destination.unlink()

    print(f"[INFO] Downloading ncnn package: {url}")
    progress = {"next_percent": 0}

    def report_progress(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = int(downloaded * 100 / total_size)
        if percent >= progress["next_percent"]:
            print(f"[INFO] ncnn download {percent}%")
            progress["next_percent"] += 10

    try:
        urllib.request.urlretrieve(url, temp_destination, reporthook=report_progress)
        temp_destination.replace(destination)
    except Exception as exc:
        if temp_destination.exists():
            temp_destination.unlink()
        raise RuntimeError(f"Failed to download ncnn package: {exc}") from exc


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination_root / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe path in ncnn archive: {member.filename}")
        archive.extractall(destination_root)


def find_extracted_ncnn_root(extract_dir: Path) -> Path | None:
    for config in extract_dir.rglob("ncnnConfig.cmake"):
        try:
            arch_dir = config.parents[3]
            package_root = config.parents[4]
        except IndexError:
            continue

        if (
            arch_dir.name in NCNN_PACKAGE_ARCHITECTURES
            and config.parent.name == "ncnn"
            and config.parent.parent.name == "cmake"
            and config.parent.parent.parent.name == "lib"
        ):
            return package_root
    return None


def install_ncnn_package(package_root: Path, ncnn_root: Path) -> None:
    ncnn_root.mkdir(parents=True, exist_ok=True)
    installed_arches: list[str] = []

    for arch in NCNN_PACKAGE_ARCHITECTURES:
        source = package_root / arch
        if not source.is_dir():
            continue
        target = ncnn_root / arch
        print(f"[INFO] Installing ncnn {arch}: {target}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        installed_arches.append(arch)

    if not installed_arches:
        raise RuntimeError(f"No ncnn architecture directories found in {package_root}")


def ensure_ncnn_package(args: argparse.Namespace, ncnn_root: Path) -> None:
    config_path = ncnn_config_path(ncnn_root, args.arch)
    if config_path.is_file():
        info = detect_ncnn_info(ncnn_root, args.arch)
        print(
            "[INFO] ncnn found: "
            f"version={info['version']} arch={args.arch} "
            f"linkage={info['linkage']} vulkan={info['vulkan']}"
        )
        print(f"[INFO] ncnn config: {config_path}")
        return

    if args.no_ncnn_download:
        print(f"[WARN] ncnn config is missing and auto-download is disabled: {config_path}")
        return

    print(f"[INFO] ncnn package is missing for arch={args.arch}: {config_path}")

    url = DEFAULT_NCNN_URL
    archive_path = ncnn_root / "_downloads" / package_filename_from_url(url)
    download_file(url, archive_path)

    with tempfile.TemporaryDirectory(prefix="ncnn-extract-") as temp_dir:
        extract_dir = Path(temp_dir)
        safe_extract_zip(archive_path, extract_dir)
        package_root = find_extracted_ncnn_root(extract_dir)
        if package_root is None:
            raise RuntimeError(f"Could not locate ncnnConfig.cmake inside {archive_path}")
        install_ncnn_package(package_root, ncnn_root)

    if not config_path.is_file():
        raise RuntimeError(f"ncnn package was installed, but config is still missing: {config_path}")

    info = detect_ncnn_info(ncnn_root, args.arch)
    print(
        "[INFO] ncnn ready: "
        f"version={info['version']} arch={args.arch} "
        f"linkage={info['linkage']} vulkan={info['vulkan']}"
    )
    print(f"[INFO] ncnn config: {config_path}")


def main() -> int:
    detected_default_generator = default_generator_key()
    parser = argparse.ArgumentParser(description="Build op_ocr_engine with Visual Studio CMake generators.")
    parser.add_argument("-g", "--generator", choices=GENERATORS.keys(), default=detected_default_generator)
    parser.add_argument("-a", "--arch", choices=ARCHITECTURES, default="x64")
    parser.add_argument("-t", "--type", choices=BUILD_TYPES, default="Release")
    parser.add_argument("--target", default="ocr_server", help="CMake target to build. Use empty string for all.")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before configuring.")
    parser.add_argument("--with-tesseract", action="store_true", help="Enable optional Tesseract targets.")
    parser.add_argument("--with-tests", action="store_true", help="Enable CTest/GTest targets.")
    parser.add_argument(
        "--ncnn-root",
        default="3rd_party/ncnn",
        help="Path to the prebuilt ncnn package root.",
    )
    parser.add_argument(
        "--no-ncnn-download",
        action="store_true",
        help="Do not auto-download the default prebuilt ncnn package when it is missing.",
    )
    args = parser.parse_args()

    ensure_cmake_on_path()

    project_dir = Path(__file__).parent.resolve()
    ncnn_root = resolve_path(project_dir, args.ncnn_root)
    try:
        ensure_ncnn_package(args, ncnn_root)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    generator = GENERATORS[args.generator]["cmake"]
    vs_arch = ARCH_TO_VS[args.arch]
    build_dir = project_dir / f"build-{args.generator}-{args.arch}"

    if args.clean and build_dir.exists():
        print(f"[INFO] Removing build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Build Configuration")
    print(f"    Generator:  {args.generator} ({generator})")
    print(f"    Arch:       {args.arch}")
    print(f"    Type:       {args.type}")
    print(f"    Target:     {args.target or 'ALL'}")
    print(f"    ncnn root:  {ncnn_root}")
    print(f"    Tesseract:  {'ON' if args.with_tesseract else 'OFF'}")
    print(f"    Tests:      {'ON' if args.with_tests else 'OFF'}")
    print("=" * 60)

    configure_cmd = [
        "cmake",
        "-S",
        str(project_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
        "-A",
        vs_arch,
        f"-DCMAKE_BUILD_TYPE={args.type}",
        f"-DNCNN_ROOT={ncnn_root}",
        f"-DBUILD_TESSERACT_SERVER={'ON' if args.with_tesseract else 'OFF'}",
        f"-DBUILD_TESTING={'ON' if args.with_tests else 'OFF'}",
    ]
    run(configure_cmd, cwd=project_dir)

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        args.type,
    ]
    if args.target:
        build_cmd.extend(["--target", args.target])
    run(build_cmd, cwd=project_dir)

    print("\n" + "=" * 60)
    print(f"  Build completed: {args.generator} | {args.arch} | {args.type}")
    print(f"  Build directory: {build_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
