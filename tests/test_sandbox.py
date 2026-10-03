import json
import subprocess
import time

import pytest

from sandbox.runner import LABEL, Sandbox, SandboxConfig, container_name, destroy_run
from sandbox.security import SandboxLimits
from tools import filesystem


def inspect(sandbox: Sandbox) -> dict:
    out = subprocess.run(["docker", "inspect", sandbox.name], capture_output=True, text=True, check=True).stdout
    return json.loads(out)[0]


def run(sandbox: Sandbox, script: str, timeout: int | None = None):
    return sandbox.exec(["bash", "-c", script], timeout)


def test_runs_as_non_root(sandbox):
    assert run(sandbox, "id -u").output.strip() == "10001"


def test_hardening_is_applied(sandbox):
    host = inspect(sandbox)["HostConfig"]
    assert host["ReadonlyRootfs"] is True
    assert host["CapDrop"] == ["ALL"]
    assert host["SecurityOpt"] == ["no-new-privileges"]
    assert host["NanoCpus"] == 1_000_000_000
    assert host["Memory"] == host["MemorySwap"] == 1024**3
    assert host["PidsLimit"] == 256
    assert host["NetworkMode"] == "none"


def test_workspace_and_root_filesystem_are_read_only(sandbox):
    for path in ("/workspace/injected.py", "/etc/injected", "/usr/local/lib/injected"):
        result = run(sandbox, f"touch {path}")
        assert result.exit_code != 0 and "Read-only file system" in result.output


def test_no_network(sandbox):
    result = sandbox.exec(["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"])
    assert result.exit_code != 0 and "Network is unreachable" in result.output


def test_host_secrets_never_reach_the_container(repo, sandbox_image, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret")
    box = Sandbox.start(repo, f"secrets-{tmp_path.name[-20:]}")
    try:
        env = run(box, "env").output
    finally:
        box.destroy()
    assert "sk-secret" not in env and "lf-secret" not in env


def test_timeout_kills_the_command(sandbox):
    started = time.monotonic()
    result = run(sandbox, "sleep 30", timeout=2)
    assert result.timed_out and result.exit_code == 124
    assert time.monotonic() - started < 10
    # Nothing left running: only PID 1 (the container's sleep) and this check's shell.
    assert "sleep 30" not in run(sandbox, "cat /proc/[0-9]*/cmdline | tr '\\0' ' '").output


def test_memory_limit_kills_without_looking_like_a_timeout(repo, sandbox_image, tmp_path):
    box = Sandbox.start(repo, f"mem-{tmp_path.name[-20:]}", SandboxConfig(limits=SandboxLimits(memory="128m")))
    try:
        result = box.exec(["python", "-c", "x = bytearray(512 * 1024 * 1024)"])
    finally:
        box.destroy()
    assert result.exit_code == 137
    assert not result.timed_out


def test_writable_space_is_capped(repo, sandbox_image, tmp_path):
    box = Sandbox.start(repo, f"disk-{tmp_path.name[-20:]}", SandboxConfig(limits=SandboxLimits(tmp_size="8m")))
    try:
        result = run(box, "head -c 20000000 /dev/zero > /tmp/big")
    finally:
        box.destroy()
    assert result.exit_code != 0 and "No space left on device" in result.output


def test_output_is_capped(sandbox):
    result = run(sandbox, "head -c 1000000 /dev/zero | tr '\\0' 'x'")
    assert len(result.output) <= 200_000 + 100


def test_network_is_cut_after_dependency_install(repo, sandbox_image, tmp_path):
    # No network-free way to install a real package in a test, so the file only proves the install-then-cut path.
    filesystem.write_file(repo, "requirements.txt", "# nothing to install\n")
    box = Sandbox.start(repo, f"deps-{tmp_path.name[-20:]}")
    try:
        networks = inspect(box)["NetworkSettings"]["Networks"]
        unreachable = box.exec(["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"])
    finally:
        box.destroy()
    assert "bridge" not in networks
    assert unreachable.exit_code != 0


def test_destroy_removes_the_container(sandbox):
    sandbox.destroy()
    assert not sandbox.is_running()


def test_destroy_run_cleans_up_by_label(repo, sandbox_image):
    box = Sandbox.start(repo, "label-cleanup")
    assert box.name == container_name("label-cleanup")
    destroy_run("label-cleanup")
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}=label-cleanup"],
                         capture_output=True, text=True, check=True).stdout
    assert ids.strip() == ""


def test_containers_remove_themselves_when_their_lifetime_ends(repo, sandbox_image, tmp_path):
    box = Sandbox.start(repo, f"life-{tmp_path.name[-20:]}", SandboxConfig(lifetime_seconds=2))
    deadline = time.monotonic() + 15
    while box.is_running() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not box.is_running()


def test_missing_image_fails_clearly(repo):
    with pytest.raises(RuntimeError, match="docker build"):
        Sandbox.start(repo, "no-image", SandboxConfig(image="patchpilot-does-not-exist:latest"))
