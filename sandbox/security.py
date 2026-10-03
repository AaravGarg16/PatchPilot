"""The container hardening applied to every sandbox. Kept in one place so it is easy to audit."""
from pydantic import BaseModel

SANDBOX_UID = 10001


class SandboxLimits(BaseModel):
    cpus: float = 1.0
    memory: str = "1g"
    pids: int = 256
    # Writable space is only these tmpfs mounts (root fs is read-only), so their sizes are the disk limit.
    tmp_size: str = "256m"
    deps_size: str = "1g"


def hardening_flags(limits: SandboxLimits) -> list[str]:
    uid = SANDBOX_UID
    return [
        "--user", f"{uid}:{uid}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={limits.tmp_size},mode=1777",
        "--tmpfs", f"/deps:rw,nosuid,nodev,size={limits.deps_size},uid={uid},gid={uid},mode=0755",
        "--cpus", str(limits.cpus),
        "--memory", limits.memory,
        "--memory-swap", limits.memory,  # equal to memory: no swap on top
        "--pids-limit", str(limits.pids),
    ]
