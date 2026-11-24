#!/usr/bin/env python3
"""Prepare RYZENStrata from a source checkout and verify the local chat API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "terminal"
WINDOWS = sys.platform == "win32"
MODEL = "qwen3.8-flash-next-q2_0"


def run(argv: list[str | Path], *, extra_env: dict[str, str] | None = None) -> None:
    process = subprocess.run(
        [str(arg) for arg in argv],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1", **(extra_env or {})},
    )
    if process.returncode:
        raise SystemExit(process.returncode)


def windows_engine(python: Path) -> None:
    import setup
    from strata_control.ryzen_policy import engine_has_profile

    override = os.environ.get("STRATA_RYZEN_ENGINE_DIR")
    directory = Path(override).resolve() if override else ROOT / "dist" / "strata-windows-x64-hip"
    if engine_has_profile(directory / "strata.exe", env=setup.engine_env(directory)):
        return
    if override:
        raise SystemExit(f"No RYZENStrata HIP engine in STRATA_RYZEN_ENGINE_DIR={directory}")
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise SystemExit("Install Visual Studio Build Tools with the C++ workload before building the Windows HIP engine.")
    search_path = str(python.parent) + os.pathsep + os.environ.get("PATH", "")
    if not all(shutil.which(tool, path=search_path) for tool in ("cmake", "ninja")):
        run([python, "-m", "pip", "install", "-r", TERMINAL / "requirements-build.txt"])
    print("Building the RYZENStrata HIP engine.", flush=True)
    run(["cmd.exe", "/d", "/c", r"terminal\tools\hip\build_windows.bat"], extra_env={"PATH": search_path})
    if not engine_has_profile(directory / "strata.exe", env=setup.engine_env(directory)):
        raise SystemExit(f"The engine build did not produce a usable HIP engine in {directory}.")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="check hardware and print the launch plan without setup")
    parser.add_argument("--data-dir", type=Path, help="model directory on an SSD")
    parser.add_argument("--port", type=int, default=8080, help="local API port (default: 8080)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if (platform.machine().lower() not in {"amd64", "x86_64"}
            or not (WINDOWS or sys.platform == "linux" and "microsoft" not in platform.release().lower())):
        parser.error("RYZENStrata needs Windows x64 or native x86-64 Linux.")
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")

    sys.path.insert(0, str(TERMINAL))
    from strata_control.discovery import default_data_dir

    data_dir = (args.data_dir or default_data_dir()).resolve()
    data = ["--data-dir", str(data_dir)]
    run([sys.executable, TERMINAL / "setup.py", "--check"])
    launch = [TERMINAL / "strata.py", "launch", "--access", "local", "--port", str(args.port), *data]
    if args.plan:
        run([sys.executable, *launch, "--plan", "--output", "json"])
        print(f"SSD check: {'SSD' if is_ssd(data_dir) else 'SSD could not be verified'} ({data_dir})", flush=True)
        return
    if not is_ssd(data_dir):
        raise SystemExit(f"Could not verify that {data_dir} is on an SSD. Set --data-dir to a local SSD directory.")

    environment = TERMINAL / ".venv"
    python = environment / ("Scripts/python.exe" if WINDOWS else "bin/python")
    if not python.is_file():
        venv.create(environment, with_pip=True)
    if WINDOWS:
        windows_engine(python)
    run([python, TERMINAL / "setup.py", "--yes", "--no-start", "--port", str(args.port), *data])
    run([python, *launch, "--yes", "--no-tui"])
    verify_api(args.port)


if __name__ == "__main__":
    main()
