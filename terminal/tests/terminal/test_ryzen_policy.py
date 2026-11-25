"""Checks for the launch defaults and strict runtime configuration of RYZENStrata."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata_cli.launch.settings import LaunchSettings
from strata_control import ryzen_policy as policy


def hardware(cpu="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", installed=64, arch="gfx1151"):
    return {"cpu": {"name": cpu}, "ram_gb": 15.65, "installed_ram_gb": installed,
            "gpus": [{"index": 0, "arch": arch, "dedicated_gb": 48, "vram_gb": 55.8}]}


def test_64_gib_with_48_gib_carveout_is_supported():
    assert policy.hardware_problem(hardware()) is None
    assert policy.hardware_problem(hardware("AMD Ryzen AI Max+ PRO 495 w/ Radeon 8065S")) is None
    assert "64 GiB" in policy.hardware_problem(hardware(installed=32))
    assert "395 or 495" in policy.hardware_problem(hardware("AMD Ryzen AI Max 390"))
    assert "gfx1151" in policy.hardware_problem(hardware(arch="gfx1150"))
    assert policy.hardware_problem(hardware(), gpu=1) == policy.GPU_ERROR


def test_launch_defaults_and_unsupported_choices():
    launch = LaunchSettings()
    assert policy.launch_problem(launch) is None
    policy.set_launch_defaults(launch)
    assert launch.parallel == 1
    assert (launch.family, launch.model) == ("qwen", "Q2_0")
    assert (launch.context, launch.kv, launch.drafting, launch.vision) == (262144, "int8", "dflash2", "off")
    assert policy.launch_problem(LaunchSettings(family="qwen", model="IQ3_S")) == policy.MODEL_ERROR
    assert policy.launch_problem(LaunchSettings(vision="cpu"))
    assert policy.launch_problem(LaunchSettings(drafting="mtp"))


@pytest.mark.parametrize("slots", policy.PARALLEL_CHOICES)
def test_requested_gpu_slots_reach_native_engine(monkeypatch, tmp_path, slots):
    from serve import server
    from strata_control import drafting

    launch = LaunchSettings(parallel=slots)
    policy.set_launch_defaults(launch)
    assert launch.parallel == slots and policy.launch_problem(launch) is None
    cfg = {"model_name": policy.MODEL_NAME, "args": ["--pack", str(tmp_path / "packs" / "q2_0"), "--native", "q.gguf"]}
    policy.apply_runtime(cfg, launch.context, launch.drafting, launch.parallel)
    monkeypatch.setattr(drafting, "stock_q4_ok", lambda _: True)
    assert policy.runtime_problem(cfg) is None
    args = server.engine_args(cfg, announce=False)
    if slots > 1:
        assert args.count("--batch") == 1 and args[args.index("--batch") + 1] == str(slots)
        assert args[args.index("--draft") + 1] == "dflash2"
    else:
        assert "--batch" not in args


def test_parallel_limits_and_unassisted_decoding():
    for n in (True, 0, 3, 16.0, "auto", 32):
        assert policy.launch_problem(LaunchSettings(parallel=n))
    assert policy.launch_problem(LaunchSettings(parallel=2, drafting="off"))
    assert policy.launch_problem(LaunchSettings(parallel=1, drafting="off")) is None


def test_runtime_settings_replace_overrides_and_validate_the_native_profile():
    cfg = {"model_name": policy.MODEL_NAME, "args": ["--pack", "packs/q2_0", "--native", "shard1.gguf",
           "--ple-io", "ram", "--kv", "fp16", "--resident-experts", "--batch", "2", "--mtp", "old"],
           "user_args": ["--ple-io", "mmap", "--cpu"], "vision": {"exe": "cpu.exe"}, "gpu": 0}
    assert policy.runtime_problem(cfg)
    policy.apply_runtime(cfg, 262144, "off")
    assert policy.runtime_problem(cfg) is None
    assert cfg["user_args"] == []
    assert cfg["args"][4:] == ["--gpu-experts-only", "--max-context", "262144", "--kv", "int8", "--ple-io", "direct",
                               "--mmap-experts", "--prefill", "auto", "--spec", "4"]
    assert "--mtp" not in cfg["args"] and "--resident-experts" not in cfg["args"]
    assert policy.runtime_problem({**cfg, "gpu": [0, 1]})
    assert policy.runtime_problem({**cfg, "drafting": "mtp"})


@pytest.mark.parametrize("os_name, default_flags", [("nt", ["--ssd-experts"]), ("posix", [])])
def test_saved_ryzen_configs_use_the_platform_expert_io_default(monkeypatch, os_name, default_flags):
    from serve import server

    monkeypatch.setattr(server, "os", SimpleNamespace(name=os_name))
    saved_flags = ["--mmap-experts"]
    cfg = {"ryzen_profile": True, "drafting": "off", "args": saved_flags}
    assert server.engine_args(cfg, announce=False) == ["--mmap-experts", *default_flags]
    assert saved_flags == ["--mmap-experts"]

    saved_flags.append("--ssd-experts")
    assert server.engine_args(cfg, announce=False) == saved_flags


def test_stock_drafter_is_verified_and_has_no_fallback(monkeypatch):
    from serve import server
    from strata_control import drafting

    root = Path("models-test")
    cfg = {"model_name": policy.MODEL_NAME, "args": ["--pack", str(root / "packs" / "q2_0"),
                                                    "--native", "shard1.gguf"], "gpu": 0}
    policy.apply_runtime(cfg, 262144, "dflash2")
    monkeypatch.setattr(drafting, "vocab_path", lambda *_: root / "draft_vocab.bin")
    monkeypatch.setattr(drafting, "stock_q4_ok", lambda _: True)
    args = server.engine_args(cfg, announce=False)
    assert args[args.index("--draft-model") + 1] == str(drafting.target_path(root))
    assert args[args.index("--draft-fallback") + 1] == "none"
    assert args.count("--spec") == 1
    assert args[args.index("--spec") + 1] == str(policy.DFLASH_WINDOW)
    assert "--mtp" not in args


def test_request_vocab_uses_verified_profiles_and_respects_fixed_vocab(monkeypatch, tmp_path):
    from serve import server
    from strata_control import drafting

    cfg = {"model_name": policy.MODEL_NAME, "args": ["--pack", str(tmp_path / "packs" / "q2_0"),
                                                    "--native", "q2_0.gguf"], "gpu": 0}
    policy.apply_runtime(cfg, policy.DEFAULT_CONTEXT, policy.DRAFTER)
    monkeypatch.setattr(drafting, "stock_q4_ok", lambda _: True)
    monkeypatch.setattr(server, "engine_has_text", lambda *_: True)
    args = server.engine_args(cfg, "strata.exe", announce=False)
    assert args[args.index("--draft-vocab-dir") + 1] == str(drafting.shipped_vocab().parent)
    fixed = {"user_args": ["--mtp-draft-vocab", str(drafting.shipped_vocab("fr"))]}
    args = server.engine_args({**cfg, **fixed}, "strata.exe", announce=False)
    assert "--draft-vocab-dir" not in args
    assert args[args.index("--mtp-draft-vocab") + 1] == str(drafting.shipped_vocab("fr"))
    monkeypatch.setattr(server, "engine_has_text", lambda *_: False)
    assert "--draft-vocab-dir" not in server.engine_args(cfg, "legacy.exe", announce=False)
