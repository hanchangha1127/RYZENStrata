"""Checks for the launch defaults and strict runtime configuration of RYZENStrata."""
from pathlib import Path

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
    assert (launch.family, launch.model) == ("qwen", "Q2_0")
    assert (launch.context, launch.kv, launch.drafting, launch.vision) == (262144, "int8", "dflash2", "off")
    assert policy.launch_problem(LaunchSettings(family="qwen", model="IQ3_S")) == policy.MODEL_ERROR
    assert policy.launch_problem(LaunchSettings(vision="cpu"))
    assert policy.launch_problem(LaunchSettings(drafting="mtp"))


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
    assert args[args.index("--spec") + 1] == "4"
    assert "--mtp" not in args
