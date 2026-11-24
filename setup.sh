#!/bin/sh
# RYZENStrata for native x86-64 Linux with an AMD Ryzen AI Max+ 395 or 495.
# Python (with venv) is installed through apt/dnf if it is missing (asks for sudo).
# NixOS (or STRATA_UV=1) with uv installed: uv makes the project venv; every other Linux uses the standard venv.
cd "$(dirname "$0")" || exit 1
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ] || grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "Use native x86-64 Linux or run START-HERE.bat on Windows." >&2
  exit 1
fi
# the Terminal's sources and everything setup writes live in terminal/ (0.2.16 layout); an older clone's .venv moves there once
if [ -f terminal/setup.py ]; then
  [ -e terminal/.venv ] || { [ -x .venv/bin/python ] && mv .venv terminal/.venv; }
  cd terminal || exit 1
fi
# A bundled Python runtime (the python/ folder of the terminal archive) is used when it is there: nothing to find,
# install or download. (./setup.sh --setup, --update)
if [ -x python/bin/python3 ]; then
  PYTHONNOUSERSITE=1; export PYTHONNOUSERSITE
  unset PYTHONHOME PYTHONPATH
  exec python/bin/python3 setup.py "$@"
fi
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  # uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: STRATA_UV=1.
  # Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
  UV=""
  if [ "${STRATA_UV:-}" = "1" ] || grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
    UV="$(command -v uv 2>/dev/null || true)"
  fi
  PY=""
  if [ -n "$UV" ]; then
    # Prefer uv's own interpreter: unlike Nix's Python, it works naturally with pip wheels using dlopen.
    PY=3.12
  else
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then
        PY=$c; break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    fi
    PY=python3
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again. On NixOS, add uv to environment.systemPackages."
      exit 1
    fi
  fi
  if [ -n "$UV" ]; then
    # Force an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store.
    "$UV" venv --managed-python --seed --python "$PY" .venv || { echo "Could not create the uv environment in .venv."; exit 1; }
  else
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
  fi
fi
exec .venv/bin/python setup.py "$@"
