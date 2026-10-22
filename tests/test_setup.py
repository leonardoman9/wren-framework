"""python -m wren setup: precondition checks and messages, symlink replacement, idempotency
(contracts/cli-setup.md). Nothing is installed: the install steps are replaced by stand-ins.
Usage: .venv-torch/bin/python -m pytest tests/test_setup.py"""
import sys

import pytest

from wren import setup as S


def test_platforms():
    assert S.platform_name("linux", "x86_64") == "linux-x86_64"
    assert S.platform_name("darwin", "arm64", "15.1") == "macos-arm64"
    with pytest.raises(S.Stop, match="Intel processor.*Docker"):
        S.platform_name("darwin", "x86_64")
    with pytest.raises(S.Stop, match="macOS 14 or later"):
        S.platform_name("darwin", "arm64", "13.6")
    with pytest.raises(S.Stop, match="not supported directly.*Docker"):
        S.platform_name("linux", "aarch64")


def test_python_version_message(monkeypatch):
    monkeypatch.setattr(S.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(S.shutil, "which", lambda exe: None)
    with pytest.raises(S.Stop, match=r"Python 3\.11 is needed \(found 3\.12\)"):
        S.find_python()


def test_disk(monkeypatch):
    with pytest.raises(S.Stop, match=r"About 6 GB of free disk space is needed; 4 GB are free"):
        S.check_disk(free_gb=4)
    S.check_disk(free_gb=40)
    monkeypatch.setenv("WREN_MIN_DISK_GB", "3")  # CI runners may lower it
    S.check_disk(free_gb=4)
    monkeypatch.setenv("WREN_MIN_DISK_GB", "100")  # but never raise it
    S.check_disk(free_gb=7)


def test_missing_tool_hint():
    with pytest.raises(S.Stop) as e:
        S.check_tools(which=lambda exe: None if exe == "wget" else "/usr/bin/" + exe, system="darwin")
    assert "wget" in str(e.value) and "brew install make wget mtools qemu" in str(e.value)
    with pytest.raises(S.Stop, match="sudo apt install"):
        S.check_tools(which=lambda exe: None if exe == "mformat" else "/usr/bin/" + exe, system="linux")
    with pytest.raises(S.Stop, match="GNU make 4"):  # macOS: the system make (3.81) cannot build TFLM
        S.check_tools(which=lambda exe: None if exe == "gmake" else "/usr/bin/" + exe, system="darwin")
    S.check_tools(which=lambda exe: None if exe == "gmake" else "/usr/bin/" + exe, system="linux")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repository root with .in files and matching locks."""
    (tmp_path / "envs" / "lock").mkdir(parents=True)
    for env in S.ENVS:
        spec = tmp_path / "envs" / f"requirements-{env}.in"
        spec.write_text(f"{env}-package==1.0\n")
        (tmp_path / "envs" / "lock" / f"{env}-linux-x86_64.txt").write_text(f"# input-sha256: {S.sha256(spec)}\n")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "LOG", tmp_path / "runs" / "_setup" / "logs" / "setup.log")
    return tmp_path


def test_stale_lock(repo):
    S.lock_file("torch", "linux-x86_64")
    (repo / "envs" / "requirements-torch.in").write_text("torch-package==2.0\n")
    with pytest.raises(S.Stop, match="out of date"):
        S.lock_file("torch", "linux-x86_64")


def test_symlinked_env_replaced_target_untouched(repo, tmp_path_factory, monkeypatch):
    target = tmp_path_factory.mktemp("elsewhere")
    (target / "keep.txt").write_text("not ours")
    (repo / ".venv-tf").symlink_to(target)
    calls = []

    def fake_step(cmd, what, env=None):
        calls.append(cmd)
        if "venv" in [str(c) for c in cmd]: (repo / ".venv-tf" / "bin").mkdir(parents=True)

    monkeypatch.setattr(S, "step", fake_step)
    lock = S.lock_file("tf", "linux-x86_64")
    S.install_env("tf", lock, sys.executable)
    assert not (repo / ".venv-tf").is_symlink() and (repo / ".venv-tf").is_dir()
    assert (target / "keep.txt").read_text() == "not ours"
    assert S.env_ready("tf", lock) and len(calls) == 2


def test_second_run_installs_nothing(repo, monkeypatch):
    monkeypatch.setattr(S, "platform_name", lambda: "linux-x86_64")
    monkeypatch.setattr(S, "find_python", lambda given=None: sys.executable)
    monkeypatch.setattr(S, "check_tools", lambda: None)
    monkeypatch.setattr(S, "check_disk", lambda: None)
    monkeypatch.setattr(S, "tflm_ready", lambda: True)
    monkeypatch.setattr(S, "versions", lambda plat: [])
    installs = []
    monkeypatch.setattr(S, "install_env", lambda env, lock, py: (installs.append(env),
                        (repo / f".venv-{env}").mkdir(exist_ok=True), (repo / f".venv-{env}" / ".wren-lock").write_text(S.sha256(lock))))
    assert S.main() == 0 and installs == ["torch", "tf"]
    installs.clear()
    assert S.main() == 0 and installs == []
