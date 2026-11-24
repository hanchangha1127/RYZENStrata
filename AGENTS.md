# AGENTS.md

Version 0.2.21, engine 0.1.67. These are the upstream base versions.

RYZENStrata serves ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO Q2_0 text on the
integrated gfx1151 GPU of AMD Ryzen AI Max+ 395 and 495 systems. It targets
Windows x64 and native x86-64 Linux, with at least 64 GiB installed memory.
At least 48 GiB of graphics memory is recommended. See [docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)
for product details and [docs/AI_SETUP.md](docs/AI_SETUP.md) for setup from this fork.

The default context is 262,144 tokens with Q8 KV and fused GPU attention.
The N-gram table uses direct SSD reads. DFlash2 Q4_0 is the supported drafter.
Keep model numerical work on the GPU. Do not add a fallback that computes it on
the CPU. Tokenization, file reads, GPU scheduling, and API handling are host work.

The desktop app lives in `app/desktop` and `app/host`. The Terminal and Control
live in `terminal`. The HIP engine lives in `engine`. Source setup starts at
`START-HERE.bat` or `setup.sh` and requires an engine from this fork. Use
`python terminal/strata.py launch --plan` to view the chosen settings before
downloading models. The default local API listens on `127.0.0.1`.

## When a user gives you this repository URL to set up RYZENStrata

1. Work in a shell on the user's target Ryzen PC. Clone this repository if it is
   missing. Reuse an existing checkout of `hanchangha1127/RYZENStrata`.
2. Run `py -3 tools/agent_setup.py --plan` on Windows or
   `python3 tools/agent_setup.py --plan` on native x86-64 Linux. Check the
   installed memory and SSD result. Pass `--data-dir` to select a local SSD if
   the default directory is unsuitable.
3. Run the same command without `--plan`. Windows setup downloads the pinned
   release engine or reuses a compatible local engine. Linux setup builds the
   HIP engine. Setup reads the routed experts from the original Q2_0 GGUF,
   prepares the DFlash2 Q4_0 drafter and starts the local server.
   See [docs/AI_SETUP.md](docs/AI_SETUP.md).
4. Wait for `RYZENStrata answered at http://127.0.0.1:8080/v1` or the selected
   `--port`. Report the verified local address and any failure that prevented
   the test chat request from completing.

## Working on the code

- Run the Terminal and Control tests with `python -m pytest terminal/tests/terminal`.
- Run the desktop host tests with `npm test` in `app/host`.
- Use the `cpu-check` CMake preset for host-side tests, and `hip-compile` for the
  gfx1151 engine. Use hardware measurements for performance claims.
- The upstream [AMD HIP guide](docs/AMD_HIP.md) and [Strix Halo guide](docs/STRIX_HALO.md)
  document the engine's implementation and previously measured configurations.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
