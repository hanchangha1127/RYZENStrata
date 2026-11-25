# Set up RYZENStrata

Use a Ryzen AI Max+ 395 or 495 system with 64 GiB or more installed. A 48 GiB or larger graphics memory allocation is recommended. Keep the model data on an SSD connected to the machine that runs Strata.

## Setup command

Clone `hanchangha1127/RYZENStrata` and run the command for your system from the repository root:

- Windows x64 uses `py -3 tools/agent_setup.py`. Install the AMD graphics driver and Python 3.10 or later. Setup reuses a compatible fork build or downloads the pinned Windows HIP engine with per-request DFlash2 vocabularies. The release path requires no compiler. [RYZENStrata release](https://github.com/hanchangha1127/RYZENStrata/releases/tag/ryzen-v0.2.0)
- Native x86-64 Linux uses `python3 tools/agent_setup.py`. Install Python 3.10 or later with `venv` support. The system needs the AMD `amdgpu` driver and permission to use the GPU. The upstream [Strix Halo guide](STRIX_HALO.md#1-the-toolchain-no-root-needed) documents the compiler and driver setup.

The Windows ZIP contains the executables and packaged ROCm runtime at the archive root and under `rocm/`. Setup resumes interrupted downloads, verifies the ZIP against the SHA-256 in the checkout, and caches the verified engine under `dist/windows-engine/`. The same directory is used for model preparation, launch, and later starts.

Windows source builds are available with `py -3 tools/agent_setup.py --build-engine`. Install Visual Studio Build Tools with the C++ workload first. The script installs the pinned CMake and Ninja if needed and stages a separate build under `dist/`. To use a compatible local Windows HIP engine, set `STRATA_RYZEN_ENGINE_DIR` to the directory that contains `strata.exe` and `BUILD.json`.

The script uses a local SSD for the model data. Append `--data-dir <directory>` if the default folder is on a different drive. It reuses verified model files on subsequent runs. A first run downloads about 67.6 GB for the Q2_0 and DFlash2 weights, plus engine build dependencies where needed. The SSD also needs room for the prepared model files.

Append `--plan` to check the machine, available engine, expected downloads, and SSD without installing anything. A normal run finishes by submitting a chat request to the loopback API. The default address is `http://127.0.0.1:8080/v1`. Set `--port` if port 8080 is already in use.

The launch defaults to 262,144 context tokens and Q8 KV. The engine uses direct reads for the N-gram table. A smaller context can be selected if the driver cannot allocate enough memory. On Windows, Windows-visible usable system memory excludes the dedicated GPU allocation, so check the amount installed for the 64 GiB threshold.

The [technical reference](AI_REFERENCE.md) lists the expected runtime settings and API endpoints.
