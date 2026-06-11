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
import shutil
import subprocess
import sys
from pathlib import Path


GENERATORS = {
    "vs2022": {"cmake": "Visual Studio 17 2022"},
    "vs2026": {"cmake": "Visual Studio 18 2026"},
}

BUILD_TYPES = ("Debug", "Release", "RelWithDebInfo")
ARCHITECTURES = ("x86", "x64")
ARCH_TO_VS = {"x86": "Win32", "x64": "x64"}


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
    args = parser.parse_args()

    ensure_cmake_on_path()

    project_dir = Path(__file__).parent.resolve()
    ncnn_root = resolve_path(project_dir, args.ncnn_root)
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
