"""Find and stop running sgen servers.

Starting the app used to leave whatever was already running in place. Because a
second instance cannot bind the same port, it would quietly land on another one
or fail — and after a few days five servers were listening, most of them on code
from before the last edit. They served the current HTML and JavaScript straight
from disk while answering with stale API handlers, which is the worst kind of
mismatch to debug: the page looks right and behaves wrong.

So `sgen ui` now finds the previous instances and stops them first.

Identification is deliberately strict. A process is only killed if the port
answers as sgen — `/api/health` on a current server, `/api/meta` on one started
before this module existed. Something else on 8420 is reported and left alone;
this must never become a tool that kills a stranger's process because it liked
the number.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

log = logging.getLogger(__name__)

# Ports to look at. 8420 is the default; the rest cover instances started with
# --port while experimenting, which are exactly the ones that get forgotten.
PORT_SCAN = range(8420, 8440)

HOST = "127.0.0.1"


@dataclass
class Instance:
    """A server found listening on a local port."""

    port: int
    pid: int | None = None
    is_sgen: bool = False
    detail: str = ""

    def describe(self) -> str:
        what = "sgen" if self.is_sgen else (self.detail or "another program")
        where = f"pid {self.pid}" if self.pid else "pid unknown"
        return f":{self.port} ({what}, {where})"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def port_is_open(port: int, timeout: float = 0.15) -> bool:
    """Is anything listening? A refused connection on loopback is immediate."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((HOST, port)) == 0


def _get_json(port: int, path: str, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}{path}", timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def identify(port: int) -> Instance:
    """Work out what owns a port, without assuming it is ours."""
    health = _get_json(port, "/api/health")
    if health and health.get("app") == "sgen":
        return Instance(port=port, pid=health.get("pid"), is_sgen=True)

    # Servers started before /api/health existed. /api/meta is distinctive
    # enough: profiles plus a work_dir is not a shape other apps return.
    meta = _get_json(port, "/api/meta")
    if meta and "profiles" in meta and "work_dir" in meta:
        return Instance(
            port=port, pid=pid_for_port(port), is_sgen=True,
            detail="sgen, started before this version",
        )

    return Instance(port=port, pid=pid_for_port(port), is_sgen=False)


def pid_for_port(port: int) -> int | None:
    """Ask the OS which process holds a port. Used only for older servers."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            pattern = re.compile(rf"^\s*TCP\s+\S+:{port}\s+\S+\s+LISTENING\s+(\d+)", re.M)
        else:
            out = subprocess.run(
                ["ss", "-ltnp"], capture_output=True, text=True, timeout=15
            ).stdout
            pattern = re.compile(rf":{port}\s.*?pid=(\d+)")
        match = pattern.search(out)
        return int(match.group(1)) if match else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def discover(ports: Iterable[int] = PORT_SCAN) -> list[Instance]:
    """Every listening port in range, with what owns it."""
    return [identify(port) for port in ports if port_is_open(port)]


# --------------------------------------------------------------------------- #
# stopping
# --------------------------------------------------------------------------- #

def _kill(pid: int) -> None:
    if os.name == "nt":
        # /T takes the tree: `--reload` runs the app in a child process, and
        # killing only the parent would leave the child holding the port.
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True, timeout=20)
    else:
        import signal

        os.kill(pid, signal.SIGTERM)


def stop(instance: Instance, timeout: float = 6.0) -> bool:
    """Stop one instance and wait for its port to actually free up.

    Returning before the port is free would make the caller's own bind fail with
    "address in use", which is the problem this is meant to remove.
    """
    if not instance.is_sgen:
        log.warning("%s is not sgen; leaving it alone", instance.describe())
        return False
    if instance.pid is None:
        log.warning("could not find the process behind %s", instance.describe())
        return False
    if instance.pid == os.getpid():
        return False

    try:
        _kill(instance.pid)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not stop %s: %s", instance.describe(), exc)
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_is_open(instance.port):
            return True
        time.sleep(0.15)
    return not port_is_open(instance.port)


def stop_running(
    ports: Iterable[int] = PORT_SCAN, keep_port: int | None = None
) -> tuple[list[Instance], list[Instance]]:
    """Stop every sgen server found. Returns (stopped, left alone).

    `keep_port` spares one port, for `sgen stop` invoked from inside a session
    that wants to keep its own server up.
    """
    stopped: list[Instance] = []
    skipped: list[Instance] = []
    for instance in discover(ports):
        if keep_port is not None and instance.port == keep_port:
            continue
        if instance.is_sgen and stop(instance):
            stopped.append(instance)
        else:
            skipped.append(instance)
    return stopped, skipped
