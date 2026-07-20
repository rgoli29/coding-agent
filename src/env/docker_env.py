"""DockerEnvironment — a per-task sandbox on the HOST docker daemon.

We shell out to the docker CLI rather than the SDK: the commands below are
exactly what you would type to debug a stuck task, which matters a lot while
learning the mechanics.

The container is a sibling, not a child — this process talks to the host daemon
through the mounted /var/run/docker.sock.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import uuid

from src.config import SandboxConfig
from src.dataset import Instance
from src.env.base import Environment, ExecResult

log = logging.getLogger("env")

WORKDIR = "/testbed"

# Commands run through a NON-login bash, so the image's `conda activate testbed`
# in /root/.bashrc is not sourced automatically. BASH_ENV forces it. Without
# this every test runs against the base env and fails.
EXEC_ENV = {
    "BASH_ENV": "/root/.bashrc",
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


def instance_image(instance_id: str) -> str:
    """SWE-bench's published per-instance image for x86_64.

    Mirrors the harness naming: lowercased, and `__` -> `_1776_` because docker
    tags disallow a double underscore.
    """
    name = f"sweb.eval.x86_64.{instance_id}".lower().replace("__", "_1776_")
    return f"swebench/{name}:latest"


class DockerEnvironment(Environment):
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.container_id: str | None = None
        self.image: str | None = None
        self.base_commit: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def reset(self, instance: Instance) -> None:
        """Start a fresh container for `instance`, replacing any previous one."""
        self.close()
        self.image = instance_image(instance.instance_id)
        self.base_commit = instance.base_commit

        if not self._image_present(self.image):
            log.info("pulling %s (this is the slow part)", self.image)
            pull = self._run_docker(
                ["pull", self.image], timeout=self.config.pull_timeout_s
            )
            if pull.returncode != 0:
                raise RuntimeError(
                    f"could not pull {self.image}: {pull.stderr.strip()[:300]}"
                )

        name = f"agent-{instance.instance_id.replace('__', '-')}-{uuid.uuid4().hex[:6]}"
        started = self._run_docker(
            [
                "run", "-d", "--rm",
                "--name", name,
                "-w", WORKDIR,
                self.image,
                "sleep", "infinity",
            ],
            timeout=120,
        )
        if started.returncode != 0:
            raise RuntimeError(f"could not start sandbox: {started.stderr.strip()[:300]}")
        self.container_id = started.stdout.strip()
        log.info("sandbox %s up (%s)", name, self.container_id[:12])

        # Make the working tree clean and pinned to the base commit, so that
        # get_diff() returns exactly the agent's changes.
        self.exec("git config --global --add safe.directory /testbed", timeout=30)
        self.exec(
            f"git checkout {shlex.quote(instance.base_commit)} -- . && git status --porcelain",
            timeout=120,
        )

    def close(self) -> None:
        if not self.container_id:
            return
        self._run_docker(["kill", self.container_id], timeout=60)
        log.info("sandbox %s down", self.container_id[:12])
        self.container_id = None

    # -- Environment interface --------------------------------------------

    def exec(self, cmd: str, timeout: int) -> ExecResult:
        if not self.container_id:
            raise RuntimeError("exec before reset(): no sandbox is running")

        args = ["exec", "-w", WORKDIR]
        for key, value in EXEC_ENV.items():
            args += ["-e", f"{key}={value}"]
        args += [self.container_id, "bash", "-c", cmd]

        try:
            proc = self._run_docker(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout}s",
                exit_code=124,
                timed_out=True,
            )
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)

    def read_file(self, path: str) -> str:
        result = self.exec(f"cat {shlex.quote(path)}", timeout=60)
        if result.exit_code != 0:
            raise FileNotFoundError(f"{path}: {result.stderr.strip()[:200]}")
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        # Heredoc with a quoted delimiter: no expansion, so content is literal.
        script = f"mkdir -p $(dirname {shlex.quote(path)}) && cat > {shlex.quote(path)} <<'__EOF__'\n{content}\n__EOF__"
        result = self.exec(script, timeout=60)
        if result.exit_code != 0:
            raise OSError(f"could not write {path}: {result.stderr.strip()[:200]}")

    def get_diff(self) -> str:
        """The submitted patch: everything the agent changed under /testbed.

        Diffed against the instance's base commit, because that is what the
        SWE-bench harness applies the patch to. The images carry an extra
        commit on top of base, so diffing HEAD is not the same thing.
        """
        self.exec("git add -A", timeout=60)  # stage new files so they appear in the diff
        base = shlex.quote(self.base_commit) if self.base_commit else "HEAD"
        result = self.exec(f"git diff --cached {base}", timeout=120)
        return result.stdout if result.exit_code == 0 else ""

    # -- helpers -----------------------------------------------------------

    def _image_present(self, image: str) -> bool:
        return self._run_docker(["image", "inspect", image], timeout=60).returncode == 0

    @staticmethod
    def _run_docker(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout
        )
