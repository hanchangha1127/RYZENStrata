"""The supported model, hardware and run configuration for this Strata fork."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

FAMILY = "qwen"
MODEL = "Q2_0"
MODEL_NAME = "qwen3.8-flash-next-q2_0"
TITLE = "ISTA-DASLab Qwen3.8-Flash-Next Q2_0"
ARCH = "gfx1151"
DEFAULT_CONTEXT = 262144
CONTEXTS = (8192, 32768, 65536, 131072, 204800, DEFAULT_CONTEXT)
KV = "int8"
DRAFTER = "dflash2"
DFLASH_WINDOW = 6 if os.name == "nt" else 4
DFLASH_CONFIDENCE = 0.35
MIN_RAM_GIB = 64
RECOMMENDED_VRAM_GIB = 48
GPU_ONLY_FLAG = "--gpu-experts-only"
MODEL_ERROR = f"This edition supports only {TITLE}."
GPU_ERROR = "This edition requires the integrated AMD Radeon gfx1151 GPU."
ENGINE_PROFILE = "RYZENStrata HIP profile v2"
ENGINE_ERROR = f"This edition requires {ENGINE_PROFILE}. Build the engine from this checkout."
CPU_NAME = re.compile(r"\bAMD Ryzen AI Max\+\s+(?:PRO\s+)?(?:395|495)\b", re.IGNORECASE)


def model_matches(family: str | None, model: str | None) -> bool:
    return str(family or "").lower() == FAMILY and str(model or "").upper() == MODEL


def target_cards(cards: list[dict]) -> list[dict]:
    return [g for g in cards if str(g.get("arch", "")).partition(":")[0].lower() == ARCH]


def installed_memory_gib(visible_gib: float, cards: list[dict]) -> float | None:
    """Count firmware-installed memory, including the gfx1151 BIOS carve-out."""
    if os.name == "nt":
        import ctypes

        total_kib = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total_kib)):
            return total_kib.value / 2**20
    if visible_gib <= 0:
        return None
    carved_gib = max((float(g.get("dedicated_gb") or 0) for g in target_cards(cards)), default=0.0)
    total_gib = visible_gib + carved_gib
    # Linux MemTotal excludes firmware and kernel reservations. The capacity of a
    # nominal 64 GiB Strix Halo board can read slightly below 64 GiB after adding VRAM.
    if os.name != "nt" and 63 <= total_gib < MIN_RAM_GIB:
        return float(MIN_RAM_GIB)
    return total_gib


def hardware_problem(hw: dict, gpu: int | None = None) -> str | None:
    cpu = str((hw.get("cpu") or {}).get("name") or "unknown CPU")
    if not CPU_NAME.search(cpu):
        return f"This edition requires AMD Ryzen AI Max+ 395 or 495. Detected {cpu}."
    cards = target_cards(hw.get("gpus") or [])
    if not any(gpu is None or str(g.get("index")) == str(gpu) for g in cards):
        return GPU_ERROR
    installed = hw.get("installed_ram_gb")
    if installed is None:
        installed = installed_memory_gib(float(hw.get("ram_gb") or 0), cards)
    if installed is None:
        return f"Could not verify the {MIN_RAM_GIB} GiB installed-memory requirement."
    if installed < MIN_RAM_GIB:
        return f"This edition requires {MIN_RAM_GIB} GiB of installed memory. Detected {installed:.1f} GiB."
    return None


def vram_note(cards: list[dict]) -> str | None:
    available = max((float(g.get("vram_gb") or 0) for g in target_cards(cards)), default=0.0)
    if 0 < available < RECOMMENDED_VRAM_GIB:
        return f"{RECOMMENDED_VRAM_GIB} GiB of GPU-accessible memory is recommended. Detected {available:.1f} GiB."
    return None


def host_hardware_problem(gpu: int | None = None) -> str | None:
    import setup

    cpu, _, _ = setup.cpu_info()
    cards = setup.amd_gpus()
    visible = setup.ram_gb()
    return hardware_problem({"cpu": {"name": cpu}, "gpus": cards, "ram_gb": visible}, gpu)


def launch_problem(s) -> str | None:
    if (s.family is not None or s.model is not None) and not model_matches(s.family or FAMILY, s.model or MODEL):
        return MODEL_ERROR
    if s.context is not None and not 0 < s.context <= DEFAULT_CONTEXT:
        return "Context must be at most 256K tokens."
    if s.kv not in (None, KV):
        return "This edition uses the Q8 KV cache (--kv int8)."
    if s.vision not in (None, "off", "no", "none"):
        return "This edition serves text only."
    if s.drafting not in (None, "auto", DRAFTER, "off") or s.mtp == "on":
        return "This edition supports DFlash2 Q4_0 drafting or drafting off."
    if (s.gpus and len(s.gpus) > 1) or s.gpu_roles or s.split_engine == "on":
        return "This edition runs on one gfx1151 GPU."
    if s.parallel not in (None, 1):
        return "This edition serves one request at a time."
    return None


def set_launch_defaults(s) -> None:
    s.family, s.model = FAMILY, MODEL
    s.context = s.context or DEFAULT_CONTEXT
    s.kv, s.vision, s.parallel = KV, "off", 1
    s.drafting = "off" if s.drafting == "off" else DRAFTER
    s.mtp, s.split_engine, s.tune = None, "off", "skip"


def engine_has_profile(exe: Path, *, env: dict | None = None) -> bool:
    """Probe the same binary that will load the model. The probe loads no weights."""
    if not exe.is_file():
        return False
    try:
        probe = subprocess.run([str(exe), "generate", "--help"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return ENGINE_PROFILE in probe.stdout or ENGINE_PROFILE in probe.stderr


def runtime_problem(cfg: dict) -> str | None:
    if str(cfg.get("model_name") or "").lower() != MODEL_NAME:
        return MODEL_ERROR
    args = cfg.get("args") or []
    if cfg.get("ryzen_profile") is not True or cfg.get("backend") != "hip" or GPU_ONLY_FLAG not in args:
        return ENGINE_ERROR + " Run setup again to generate the fork's model configuration."
    def values(name):
        return [args[i + 1] for i, option in enumerate(args[:-1]) if option == name]

    if values("--kv") != [KV] or values("--ple-io") != ["direct"]:
        return "This edition requires the Q8 KV cache and direct SSD n-gram reads. Run setup again."
    context = values("--max-context")
    if len(context) != 1 or not str(context[0]).isdigit() or not 0 < int(context[0]) <= DEFAULT_CONTEXT:
        return "Context must be at most 256K tokens. Run setup again to update the configuration."
    if len(values("--native")) != 1 or len(values("--pack")) != 1:
        return "Run setup again to prepare the native ISTA-DASLab Q2_0 pack."
    if cfg.get("vision") or cfg.get("images"):
        return "This edition serves text only."
    if cfg.get("drafting") not in (None, DRAFTER, "off") or "--mtp" in args:
        return "This edition supports DFlash2 Q4_0 drafting or drafting off. Run setup again."
    if (isinstance(cfg.get("gpu"), list) or cfg.get("split_engine") or cfg.get("split_serve")
            or cfg.get("split_fallback") or cfg.get("replicas") or cfg.get("gpu_roles")
            or cfg.get("gpu_none") or cfg.get("parallel", 1) not in (None, 1) or "--cpu" in args):
        return "This edition serves one request on one gfx1151 GPU."
    return None


def apply_runtime(cfg: dict, context: int, drafting: str) -> None:
    """Write the runtime settings after setup has carried over the previous config."""
    from strata_control import draft_layer, user_args

    blocked = {"--kv", "--ple-io", "--max-context", "--kv-resident", "--mtp", "--mmap-experts",
               "--batch", "--batch-groups", "--vision", "--prefill", "--spec",
               "--resident-budget-gib", "--resident-experts", "--resident-cpu-experts",
               "--cpu", "--native-flash-attn-short", "--draft", "--draft-fallback", "--draft-model"}
    for key in ("args", user_args.USER_KEY):
        groups = user_args.groups(cfg.get(key))
        cfg[key] = user_args.flat([g for g in groups if g[0].partition("=")[0] not in blocked | {GPU_ONLY_FLAG}])
    cfg["args"] += [GPU_ONLY_FLAG, "--max-context", str(context), "--kv", KV, "--ple-io", "direct",
                    "--mmap-experts", "--prefill", "auto", "--spec", str(DFLASH_WINDOW if drafting == DRAFTER else 4)]
    cfg.update(backend="hip", parallel=1, mtp=False, drafting=drafting, ryzen_profile=True)
    for key in ("vision", "images", "split_engine", "split_serve", "split_fallback", "replicas", "gpu_roles", "gpu_none"):
        cfg.pop(key, None)
    draft_layer.apply(cfg)
