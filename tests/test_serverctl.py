"""Finding and stopping previous servers.

Nothing here opens a port or kills a process: discovery and termination are both
stubbed. The behaviour that matters is which processes are considered ours, and
the answer must stay conservative — a port number is not a licence to terminate
somebody else's process.
"""

import pytest

from sgen import serverctl


@pytest.fixture
def fake_world(monkeypatch):
    """A controllable set of ports, each either sgen, foreign, or closed."""
    world = {"open": {}, "killed": [], "json": {}}

    def port_is_open(port, timeout=0.15):
        return port in world["open"]

    def get_json(port, path, timeout=1.5):
        return world["json"].get((port, path))

    def kill(pid):
        world["killed"].append(pid)
        for port, owner in list(world["open"].items()):
            if owner == pid:
                del world["open"][port]      # the port frees up, as it would

    monkeypatch.setattr(serverctl, "port_is_open", port_is_open)
    monkeypatch.setattr(serverctl, "_get_json", get_json)
    monkeypatch.setattr(serverctl, "_kill", kill)
    monkeypatch.setattr(serverctl, "pid_for_port", lambda port: world["open"].get(port))
    return world


def add_sgen(world, port, pid):
    world["open"][port] = pid
    world["json"][(port, "/api/health")] = {"app": "sgen", "pid": pid}


def add_legacy_sgen(world, port, pid):
    """A server started before /api/health existed."""
    world["open"][port] = pid
    world["json"][(port, "/api/meta")] = {"profiles": ["home-video"], "work_dir": "work"}


def add_foreign(world, port, pid):
    world["open"][port] = pid    # answers nothing recognisable


# --------------------------------------------------------------------------- #
# identification
# --------------------------------------------------------------------------- #

def test_identifies_a_current_server_by_its_own_pid(fake_world):
    add_sgen(fake_world, 8420, 4242)
    found = serverctl.identify(8420)
    assert found.is_sgen and found.pid == 4242


def test_identifies_a_server_started_before_this_version(fake_world):
    """The pile-up being fixed consists mostly of servers predating /api/health."""
    add_legacy_sgen(fake_world, 8424, 777)
    found = serverctl.identify(8424)
    assert found.is_sgen and found.pid == 777


def test_a_foreign_program_is_not_mistaken_for_sgen(fake_world):
    add_foreign(fake_world, 8421, 999)
    assert not serverctl.identify(8421).is_sgen


def test_something_answering_json_but_not_sgen_is_still_foreign(fake_world):
    """A JSON API on 8420 must not be enough to get killed."""
    fake_world["open"][8420] = 500
    fake_world["json"][(8420, "/api/health")] = {"app": "other-tool", "pid": 500}
    fake_world["json"][(8420, "/api/meta")] = {"something": "else"}
    assert not serverctl.identify(8420).is_sgen


def test_discover_only_reports_open_ports(fake_world):
    add_sgen(fake_world, 8420, 1)
    add_sgen(fake_world, 8431, 2)
    ports = {i.port for i in serverctl.discover()}
    assert ports == {8420, 8431}


# --------------------------------------------------------------------------- #
# stopping
# --------------------------------------------------------------------------- #

def test_stops_every_sgen_server_it_finds(fake_world):
    add_sgen(fake_world, 8420, 11)
    add_legacy_sgen(fake_world, 8425, 22)
    stopped, others = serverctl.stop_running()
    assert {i.port for i in stopped} == {8420, 8425}
    assert fake_world["killed"] == [11, 22]
    assert not others


def test_never_kills_a_foreign_process(fake_world):
    add_foreign(fake_world, 8433, 99)
    stopped, others = serverctl.stop_running()
    assert not stopped
    assert [i.port for i in others] == [8433]
    assert fake_world["killed"] == []


def test_refuses_to_kill_itself(fake_world, monkeypatch):
    """A server asked to replace running servers must not stop itself."""
    import os

    add_sgen(fake_world, 8420, os.getpid())
    stopped, _ = serverctl.stop_running()
    assert not stopped
    assert fake_world["killed"] == []


def test_keep_port_spares_one_server(fake_world):
    add_sgen(fake_world, 8420, 11)
    add_sgen(fake_world, 8431, 22)
    stopped, _ = serverctl.stop_running(keep_port=8420)
    assert [i.port for i in stopped] == [8431]
    assert 8420 in fake_world["open"]


def test_stop_waits_for_the_port_to_free_up(fake_world, monkeypatch):
    """Returning early would make the caller's own bind fail with 'in use'."""
    add_sgen(fake_world, 8420, 11)
    delays = {"n": 0}
    real_open = serverctl.port_is_open

    def slow_close(port, timeout=0.15):
        # Still held for the first two checks, as a dying process would be.
        delays["n"] += 1
        return True if delays["n"] <= 2 else real_open(port, timeout)

    monkeypatch.setattr(serverctl, "port_is_open", slow_close)
    monkeypatch.setattr(serverctl.time, "sleep", lambda s: None)
    assert serverctl.stop(serverctl.Instance(port=8420, pid=11, is_sgen=True))
    assert delays["n"] > 2, "must poll until the port is released"


def test_stop_gives_up_rather_than_hanging(fake_world, monkeypatch):
    add_sgen(fake_world, 8420, 11)
    monkeypatch.setattr(serverctl, "port_is_open", lambda port, timeout=0.15: True)
    monkeypatch.setattr(serverctl, "_kill", lambda pid: None)   # refuses to die
    monkeypatch.setattr(serverctl.time, "sleep", lambda s: None)
    assert not serverctl.stop(
        serverctl.Instance(port=8420, pid=11, is_sgen=True), timeout=0.3
    )


def test_an_unknown_pid_is_not_stopped(fake_world):
    assert not serverctl.stop(serverctl.Instance(port=8420, pid=None, is_sgen=True))


def test_describe_is_readable():
    text = serverctl.Instance(port=8420, pid=42, is_sgen=True).describe()
    assert "8420" in text and "42" in text and "sgen" in text


# --------------------------------------------------------------------------- #
# the CLI contract
# --------------------------------------------------------------------------- #

def test_ui_refuses_a_port_held_by_something_else(fake_world):
    """Better to stop than to silently start on a different port."""
    import typer

    from sgen import cli

    add_foreign(fake_world, 8420, 99)
    with pytest.raises(typer.Exit) as exc:
        cli._replace_running(8420)
    assert exc.value.exit_code == 1


def test_ui_replaces_previous_servers_without_complaint(fake_world):
    from sgen import cli

    add_sgen(fake_world, 8420, 11)
    cli._replace_running(8420)          # must not raise
    assert fake_world["killed"] == [11]


def test_replace_is_on_by_default_in_settings():
    from sgen import settings

    assert settings.ServerSettings().replace_running is True


def test_health_endpoint_identifies_the_process():
    pytest.importorskip("fastapi")
    import os

    from fastapi.testclient import TestClient

    from sgen.server.app import create_app

    with TestClient(create_app()) as client:
        body = client.get("/api/health").json()
    assert body["app"] == "sgen"
    assert body["pid"] == os.getpid()
