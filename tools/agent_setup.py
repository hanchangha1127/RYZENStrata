#!/usr/bin/env python3
"""Prepare RYZENStrata from a source checkout and verify the local chat API."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "terminal"
WINDOWS = sys.platform == "win32"
MODEL = "qwen3.8-flash-next-q2_0"
_WINDOWS_ENGINE_TAG = "ryzen-v0.3.0"
_WINDOWS_ENGINE_ASSET = "strata-windows-x64-hip.zip"
_WINDOWS_ENGINE_SHA256 = "a68c818f9fc7c1adb63a5e2d8b300f5e3ca2bd8647ed74c68d945493650e7658"
_WINDOWS_ENGINE_BYTES = 140224446
_WINDOWS_ENGINE_URL = f"https://github.com/hanchangha1127/RYZENStrata/releases/download/{_WINDOWS_ENGINE_TAG}/{_WINDOWS_ENGINE_ASSET}"
_WINDOWS_ENGINE_DEFAULT = ROOT / "dist" / "strata-windows-x64-hip"
_WINDOWS_ENGINE_REQUIRED = frozenset({
    "build.json", "strata.exe", "strata-device.exe", "amdhip64_7.dll", "amd_comgr.dll", "rocm_kpack.dll",
    "rocm/bin/amdhip64_7.dll", "rocm/bin/hipblas.dll", "rocm/bin/libhipblaslt.dll", "rocm/bin/rocblas.dll",
    "rocm/.kpack/blas_lib_gfx1151.kpack",
})
_WINDOWS_DEVICE_NAMES = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", re.IGNORECASE)


def run(argv: list[str | Path], *, extra_env: dict[str, str] | None = None) -> None:
    process = subprocess.run(
        [str(arg) for arg in argv],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1", **(extra_env or {})},
    )
    if process.returncode:
        raise SystemExit(process.returncode)


def windows_engine_usable(directory: Path) -> bool:
    import setup
    from strata_control.ryzen_policy import engine_has_profile

    try:
        meta = json.loads((directory / "BUILD.json").read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or meta.get("backend") != "hip" or meta.get("archs") != ["gfx1151"]:
            return False
        if meta.get("platform") != "windows-x64" or meta.get("lib_dirs") != ["rocm/bin"]:
            return False
        if not all((directory / name).is_file() for name in _WINDOWS_ENGINE_REQUIRED if name != "build.json"):
            return False
    except (OSError, ValueError):
        return False
    return engine_has_profile(directory / "strata.exe", env=setup.engine_env(directory))


def windows_release_cache() -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{64}", _WINDOWS_ENGINE_SHA256) or _WINDOWS_ENGINE_BYTES <= 0:
        return None
    cache = ROOT / "dist" / "windows-engine" / f"{_WINDOWS_ENGINE_TAG}-{_WINDOWS_ENGINE_SHA256[:12]}"
    if os.name != "nt":
        return cache
    # ROCm's nested kernel filenames can exceed MAX_PATH during extraction.
    path = str(cache.resolve())
    if path.startswith("\\\\?\\"):
        return Path(path)
    if path.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + path[2:])
    return Path("\\\\?\\" + path)


@contextmanager
def release_cache_lock(path: Path):
    import msvcrt

    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another RYZENStrata setup is preparing the Windows engine. Rerun after it finishes.") from None
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def windows_engine_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Accept the flat package_windows.py archive root and its rocm/ and tuning/ trees."""
    files: dict[str, zipfile.ZipInfo] = {}
    seen: dict[str, tuple[str, bool]] = {}
    members = archive.infolist()
    if len(members) > 4096 or sum(info.file_size for info in members) > (2 << 30):
        raise ValueError("the Windows engine ZIP exceeds the package size limit")
    for info in members:
        # orig_filename retains a NUL or backslash that ZipInfo may normalize.
        name = info.orig_filename
        is_directory = info.is_dir()
        parts = (name[:-1] if is_directory else name).split("/")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (not all(re.fullmatch(r"[A-Za-z0-9._()+ -]+", part) and not part.endswith((".", " "))
                    and not _WINDOWS_DEVICE_NAMES.fullmatch(part) for part in parts)
                or mode not in (0, stat.S_IFDIR if is_directory else stat.S_IFREG)
                or (len(parts) > 1 or is_directory) and parts[0] not in {"rocm", "tuning"}):
            raise ValueError(f"unsafe or unexpected path in the Windows engine ZIP: {name!r}")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            key = prefix.lower()
            directory = i < len(parts) or is_directory
            old = seen.setdefault(key, (prefix, directory))
            if old != (prefix, directory):
                raise ValueError(f"conflicting path in the Windows engine ZIP: {name!r}")
        if not is_directory:
            if key in files:
                raise ValueError(f"duplicate file in the Windows engine ZIP: {name!r}")
            files[key] = info
    if missing := _WINDOWS_ENGINE_REQUIRED - files.keys():
        raise ValueError(f"the Windows engine ZIP is missing {', '.join(sorted(missing))}")
    return files


def installed_release_matches(directory: Path, archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo]) -> bool:
    """Recheck an unpacked cache against the contents of its pinned archive."""
    from strata_control.transfer import sha256_file

    found = set()
    for path in itertools.chain((directory,), directory.rglob("*")):
        info = path.lstat()
        if (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))):
            return False
        if stat.S_ISREG(info.st_mode):
            key = path.relative_to(directory).as_posix().lower()
            member = files.get(key)
            if member is None or info.st_size != member.file_size:
                return False
            digest = hashlib.sha256()
            with archive.open(member) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            if sha256_file(path) != digest.hexdigest():
                return False
            found.add(key)
    return found == files.keys()


def verified_windows_archive(path: Path) -> bool:
    from strata_control.transfer import sha256_file

    return path.is_file() and path.stat().st_size == _WINDOWS_ENGINE_BYTES and sha256_file(path) == _WINDOWS_ENGINE_SHA256


def cached_windows_release() -> Path | None:
    cache = windows_release_cache()
    if cache is None:
        return None
    folder, archive = cache / "engine", cache / _WINDOWS_ENGINE_ASSET
    try:
        if folder.is_dir() and verified_windows_archive(archive):
            with zipfile.ZipFile(archive) as bundle:
                if installed_release_matches(folder, bundle, windows_engine_members(bundle)):
                    return folder
    except (OSError, ValueError, zipfile.BadZipFile):
        pass
    return None


def prepare_windows_release() -> Path:
    from strata_control import transfer

    cache = windows_release_cache()
    if cache is None:
        raise SystemExit("This checkout has no pinned Windows engine release. Use --build-engine or set STRATA_RYZEN_ENGINE_DIR.")
    cache.mkdir(parents=True, exist_ok=True)
    folder, archive = cache / "engine", cache / _WINDOWS_ENGINE_ASSET
    with release_cache_lock(cache / "setup.lock"):
        try:
            if not verified_windows_archive(archive):
                print(f"Downloading the RYZENStrata Windows HIP engine ({_WINDOWS_ENGINE_TAG}).", flush=True)
                part = archive.with_suffix(".zip.part")
                try:
                    transfer.download(_WINDOWS_ENGINE_URL, part, size=_WINDOWS_ENGINE_BYTES, sha256=_WINDOWS_ENGINE_SHA256,
                                      what="the pinned RYZENStrata Windows HIP engine", say=lambda line: print(line, flush=True))
                except transfer.ChecksumMismatch:
                    raise SystemExit("The Windows engine ZIP failed its pinned SHA-256 check. Rerun after checking the release asset.") from None
                part.replace(archive)
            with zipfile.ZipFile(archive) as bundle:
                files = windows_engine_members(bundle)
                if folder.exists():
                    if installed_release_matches(folder, bundle, files) and windows_engine_usable(folder):
                        print(f"Reusing the verified RYZENStrata Windows HIP engine in {folder}.", flush=True)
                        return folder
                    raise ValueError(f"the existing engine in {folder} failed validation; move that folder aside or use --build-engine")
                with tempfile.TemporaryDirectory(prefix=".extract-", dir=cache) as tmp:
                    stage = Path(tmp) / "engine"
                    stage.mkdir()
                    for info in files.values():
                        out = stage.joinpath(*info.filename.split("/"))
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(info) as source, out.open("xb") as target:
                            shutil.copyfileobj(source, target)
                    if not windows_engine_usable(stage):
                        raise ValueError("the extracted engine did not report the Ryzen gfx1151 profile with its packaged ROCm runtime")
                    stage.rename(folder)
        except (transfer.TransferError, OSError, ValueError, zipfile.BadZipFile) as error:
            raise SystemExit(f"Could not prepare the pinned Windows engine: {error}. Use --build-engine for an explicit local build.") from None
    return folder


def windows_engine(python: Path, *, build_engine: bool = False) -> Path:
    if override := os.environ.get("STRATA_RYZEN_ENGINE_DIR"):
        directory = Path(override).expanduser().resolve()
        if not windows_engine_usable(directory):
            raise SystemExit(f"No usable Ryzen HIP engine in STRATA_RYZEN_ENGINE_DIR={directory}")
        return directory
    if not build_engine:
        if windows_engine_usable(_WINDOWS_ENGINE_DEFAULT):
            return _WINDOWS_ENGINE_DEFAULT
        return prepare_windows_release()

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise SystemExit("Install Visual Studio Build Tools with the C++ workload before building the Windows HIP engine.")
    search_path = str(python.parent) + os.pathsep + os.environ.get("PATH", "")
    if not all(shutil.which(tool, path=search_path) for tool in ("cmake", "ninja")):
        run([python, "-m", "pip", "install", "-r", TERMINAL / "requirements-build.txt"])
    parent = ROOT / "dist"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="local-build-", dir=parent))
    directory = output / _WINDOWS_ENGINE_DEFAULT.name
    print("Building the RYZENStrata HIP engine.", flush=True)
    run(["cmd.exe", "/d", "/c", r"terminal\tools\hip\build_windows.bat"],
        extra_env={"PATH": search_path, "DIST_DIR": str(output)})
    if not windows_engine_usable(directory):
        raise SystemExit(f"The engine build did not produce a usable HIP engine in {directory}.")
    return directory


def is_ssd(folder: Path) -> bool:
    try:
        if WINDOWS:
            if not re.fullmatch(r"[A-Za-z]:", folder.drive):
                return False
            command = (
                f"$d = Get-Partition -DriveLetter '{folder.drive[0]}' -ErrorAction Stop | Get-Disk; "
                "if ($d.BusType -eq 'NVMe' -or (Get-PhysicalDisk | Where-Object { "
                "$_.MediaType -eq 'SSD' -and ($_.SerialNumber -eq $d.SerialNumber -or $_.FriendlyName -eq $d.FriendlyName) "
                "})) { Write-Output SSD }"
            )
            return subprocess.check_output(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], text=True, timeout=20
            ).strip() == "SSD"

        existing = next(path for path in (folder, *folder.parents) if path.exists())
        device = subprocess.check_output(
            ["findmnt", "--noheadings", "--output", "SOURCE", "--target", str(existing)], text=True, timeout=10
        ).strip().partition("[")[0]
        if not device.startswith("/dev/"):
            return False
        devices = subprocess.check_output(
            ["lsblk", "--inverse", "--raw", "--noheadings", "--output", "TYPE,ROTA", device], text=True, timeout=10
        )
        disks = [row.split()[1] for row in devices.splitlines() if row.split()[0] == "disk"]
        return bool(disks) and all(value == "0" for value in disks)
    except (OSError, subprocess.SubprocessError):
        return False


def verify_api(port: int) -> None:
    base = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    message = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8}
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(message).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with client.open(request, timeout=120) as response:
            answer = json.load(response)
        if not answer.get("choices"):
            raise ValueError("empty completion response")
    except (OSError, ValueError) as error:
        raise SystemExit(f"The local chat API check failed at {base}: {error}") from None
    print(f"RYZENStrata answered at {base}/v1")


def main() -> None:
    sys.path.insert(0, str(TERMINAL))
    from strata_control import ryzen_policy
    from strata_control.discovery import default_data_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="check hardware and print the launch plan without setup")
    parser.add_argument("--build-engine", action="store_true", help="build the Windows HIP engine from source")
    parser.add_argument("--data-dir", type=Path, help="model directory on an SSD")
    parser.add_argument("--port", type=int, default=8080, help="local API port (default: 8080)")
    parser.add_argument("--parallel", type=int, choices=ryzen_policy.PARALLEL_CHOICES,
                        help="manual concurrent GPU requests (default: 1)")
    parser.add_argument("--context", type=int,
                        help="maximum context tokens per request (default: 262144)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.context is not None and not 1024 <= args.context <= ryzen_policy.DEFAULT_CONTEXT:
        parser.error(f"--context must be between 1024 and {ryzen_policy.DEFAULT_CONTEXT} tokens")
    if (platform.machine().lower() not in {"amd64", "x86_64"}
            or not (WINDOWS or sys.platform == "linux" and "microsoft" not in platform.release().lower())):
        parser.error("RYZENStrata needs Windows x64 or native x86-64 Linux.")
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")

    if args.build_engine and (not WINDOWS or os.environ.get("STRATA_RYZEN_ENGINE_DIR")):
        parser.error("--build-engine is a Windows option and requires STRATA_RYZEN_ENGINE_DIR to be unset.")

    data_dir = (args.data_dir or default_data_dir()).resolve()
    data = ["--data-dir", str(data_dir)]
    if args.parallel is not None:
        data += ["--parallel", str(args.parallel)]
    if args.context is not None:
        data += ["--context", str(args.context)]
    run([sys.executable, TERMINAL / "setup.py", "--check"])
    launch = [TERMINAL / "strata.py", "launch", "--access", "local", "--port", str(args.port), *data]
    if args.plan:
        release, local = None, False
        if WINDOWS and not args.build_engine and not os.environ.get("STRATA_RYZEN_ENGINE_DIR"):
            local = windows_engine_usable(_WINDOWS_ENGINE_DEFAULT)
            if not local:
                release = cached_windows_release()
        engine_env = {"STRATA_RYZEN_ENGINE_DIR": str(release)} if release is not None else None
        run([sys.executable, *launch, "--plan", "--output", "json"], extra_env=engine_env)
        if WINDOWS and not os.environ.get("STRATA_RYZEN_ENGINE_DIR"):
            if args.build_engine:
                print("Windows engine plan: build locally.", flush=True)
            elif local:
                print("Windows engine plan: reuse the local Ryzen HIP engine.", flush=True)
            elif release is not None:
                print(f"Windows engine plan: use the verified {_WINDOWS_ENGINE_TAG} cache.", flush=True)
            elif windows_release_cache() is not None:
                print(f"Windows engine plan: download {_WINDOWS_ENGINE_ASSET} from {_WINDOWS_ENGINE_TAG} ({_WINDOWS_ENGINE_BYTES / 1e6:.0f} MB).", flush=True)
        print(f"SSD check: {'SSD' if is_ssd(data_dir) else 'SSD could not be verified'} ({data_dir})", flush=True)
        return
    if not is_ssd(data_dir):
        raise SystemExit(f"Could not verify that {data_dir} is on an SSD. Set --data-dir to a local SSD directory.")

    environment = TERMINAL / ".venv"
    python = environment / ("Scripts/python.exe" if WINDOWS else "bin/python")
    if not python.is_file():
        venv.create(environment, with_pip=True)
    engine_env = None
    if WINDOWS:
        directory = windows_engine(python, build_engine=args.build_engine)
        engine_env = {"STRATA_RYZEN_ENGINE_DIR": str(directory)}
    run([python, TERMINAL / "setup.py", "--yes", "--no-start", "--port", str(args.port), *data], extra_env=engine_env)
    run([python, *launch, "--yes", "--no-tui"], extra_env=engine_env)
    verify_api(args.port)


if __name__ == "__main__":
    main()
