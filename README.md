# RYZENStrata

RYZENStrata is a text inference fork of [Strata](https://github.com/Niko1221/Strata) for AMD Ryzen AI Max+ 395 and 495 systems. It runs one main model, [ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

## Set up with an AI agent

Give this repository URL to an AI agent with shell access on the machine that will run the model. The setup steps are in [AGENTS.md](AGENTS.md). One command checks the PC and SSD, prepares the HIP engine, downloads the pinned model and drafter, starts the local server, and sends a test chat request.

| System | Command from this checkout |
| --- | --- |
| Windows x64 | `py -3 tools/agent_setup.py` |
| Native x86-64 Linux | `python3 tools/agent_setup.py` |

Append `--plan` to check the hardware, SSD, and planned downloads without installing anything. On Windows, setup can use the checksum-verified release engine without a compiler. The [setup guide](docs/AI_SETUP.md) lists prerequisites and shows how to use a different SSD.

## Hardware and defaults

| Setting | RYZENStrata |
| --- | --- |
| Processor | Ryzen AI Max+ 395 or Ryzen AI Max+ PRO 495, including the PRO 395 |
| GPU | Integrated Radeon 8060S or 8065S, using HIP (`gfx1151`) |
| OS | Windows x64 or native x86-64 Linux |
| Installed memory | At least 64 GiB of shared system memory |
| Graphics memory | At least 48 GiB recommended |
| Context window | 262,144 tokens by default |
| Concurrent requests | 1 by default. Select 1, 2, 4, 8 or 16 with `--parallel`. Multiple slots require DFlash2 and enough GPU memory for each slot's context. |
| K and V cache | Q8 by default |
| Attention | Fused GPU attention on the native QSA path |
| Draft model | DFlash2 Q4_0 |
| Draft vocabulary | Automatically follows English, Korean, Japanese, Chinese or French user messages. Each profile includes English and code. |
| N-gram table | Direct SSD reads |
| Inference | GPU only |

The model publisher lists 37.6 GB of transformer weights and a 28.8 GB N-gram table. Keep the model on an SSD. The context window, drafter, engine scratch buffers, and operating system need additional memory. The CPU and GPU share one memory pool, so count that memory once. Actual maximum context depends on how much memory the engine and driver can allocate. [Model files and memory requirements](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF#memory-requirements)

The Windows release and the measured before-and-after results are available here. [Windows HIP engine](https://github.com/hanchangha1127/RYZENStrata/releases/tag/ryzen-v0.3.0) · [Ryzen AI Max+ 395 benchmark](docs/bench-results/2026-11-24-ryzen395-q2/README.md)

## Run from source

On Windows, append `--build-engine` to build the HIP engine from this checkout. Linux setup builds the engine locally. The [setup guide](docs/AI_SETUP.md) covers the native Windows and Linux entry points, model storage, and launch checks. The [technical reference](docs/AI_REFERENCE.md) lists the runtime settings and local API addresses.

## License

RYZENStrata retains Strata's [MIT license](LICENSE). Model weights and the drafter have their own licenses. [Main model](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) · [DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
