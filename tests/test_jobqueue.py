"""Stopping and pausing the file being processed.

Both are cooperative: the worker checks between decoded segments rather than
being killed. Tearing down a decode mid-CUDA-call is what leaves the card in a
state the next job inherits, and a subtitle file half-written is worse than none
— though atomic writes mean that cannot happen anyway.

The pipeline is stubbed here: what is under test is the queue's control flow, not
transcription.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from sgen.server.jobs import Cancelled, JobQueue


class FakePipeline:
    """Reports progress in small steps, like a real decode reporting segments."""

    def __init__(self, steps: int = 40, on_step=None):
        self.steps = steps
        self.on_step = on_step
        self.completed = []
        self.cfg = None

    def process(self, source, *, out_dir=None, overwrite=False, progress=None):
        for i in range(self.steps):
            if progress:
                progress("transcribe", i / self.steps)
            if self.on_step:
                self.on_step(i)
            time.sleep(0.01)
        self.completed.append(source)
        return Result(source)

    def close(self):
        pass


class Result:
    def __init__(self, source):
        self.source = source
        self.cues = []
        self.language = "en"
        self.language_probability = 1.0
        self.duration = 1.0
        self.suppressed_count = 0
        self.gate_summary = ""
        self.outputs = []
        self.content_id = "fake"
        self.sidecar = None
        self.audio = None
        self.verdict = None


@pytest.fixture
def queue(monkeypatch):
    q = JobQueue()
    pipeline = FakePipeline()
    monkeypatch.setattr(q, "_pipeline_for", lambda options: pipeline)
    q.pipeline = pipeline
    yield q
    q.shutdown()


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def status_of(queue, job_id):
    return next(j["status"] for j in queue.list() if j["id"] == job_id)


# --------------------------------------------------------------------------- #
# stopping
# --------------------------------------------------------------------------- #

def test_a_queued_job_is_dropped(queue):
    first = queue.submit(Path("a.mp4"), None, {})
    second = queue.submit(Path("b.mp4"), None, {})
    assert queue.cancel(second.id)
    assert status_of(queue, second.id) == "cancelled"
    assert wait_for(lambda: status_of(queue, first.id) == "done")
    assert Path("b.mp4") not in queue.pipeline.completed


def test_the_running_job_can_be_stopped(queue):
    """The case that was missing: a long file queued by mistake had to finish."""
    job = queue.submit(Path("long.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, job.id) == "running")

    assert queue.cancel(job.id)
    assert wait_for(lambda: status_of(queue, job.id) == "cancelled"), queue.list()
    assert Path("long.mp4") not in queue.pipeline.completed, "it should not finish"


def test_stopping_one_lets_the_next_start(queue):
    first = queue.submit(Path("a.mp4"), None, {})
    second = queue.submit(Path("b.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, first.id) == "running")
    queue.cancel(first.id)
    assert wait_for(lambda: status_of(queue, second.id) == "done"), queue.list()


def test_a_finished_job_cannot_be_stopped(queue):
    job = queue.submit(Path("a.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, job.id) == "done")
    assert not queue.cancel(job.id)


def test_stopping_an_unknown_job_is_false(queue):
    assert not queue.cancel("nope")


# --------------------------------------------------------------------------- #
# pausing
# --------------------------------------------------------------------------- #

def test_pause_holds_the_file_being_processed(queue):
    job = queue.submit(Path("a.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, job.id) == "running")

    queue.pause()
    assert queue.paused
    assert wait_for(lambda: status_of(queue, job.id) == "paused"), queue.list()

    # And it stays held: no progress, and certainly not finished.
    held = next(j for j in queue.list() if j["id"] == job.id)
    time.sleep(0.4)
    after = next(j for j in queue.list() if j["id"] == job.id)
    assert after["status"] == "paused"
    assert after["progress"] == held["progress"], "it should not creep forward"


def test_resume_continues_rather_than_restarting(queue):
    job = queue.submit(Path("a.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, job.id) == "running")
    assert wait_for(lambda: next(
        j for j in queue.list() if j["id"] == job.id)["progress"] > 0.05)
    queue.pause()
    assert wait_for(lambda: status_of(queue, job.id) == "paused")
    at = next(j for j in queue.list() if j["id"] == job.id)["progress"]

    queue.resume()
    assert not queue.paused
    assert wait_for(lambda: status_of(queue, job.id) == "done"), queue.list()
    assert at > 0, "progress was kept, so it carried on from there"
    assert queue.pipeline.completed == [Path("a.mp4")], "processed once, not twice"


def test_pause_also_stops_the_next_file_starting(queue):
    first = queue.submit(Path("a.mp4"), None, {})
    second = queue.submit(Path("b.mp4"), None, {})
    queue.pause()
    assert wait_for(lambda: status_of(queue, first.id) in ("paused", "running"))
    time.sleep(0.5)
    assert status_of(queue, second.id) == "queued", "nothing new should start"
    queue.resume()
    assert wait_for(lambda: status_of(queue, second.id) == "done", timeout=8)


def test_a_paused_job_can_still_be_stopped(queue):
    """Otherwise pausing would be a trap: held, and no way out but resuming."""
    job = queue.submit(Path("a.mp4"), None, {})
    assert wait_for(lambda: status_of(queue, job.id) == "running")
    queue.pause()
    assert wait_for(lambda: status_of(queue, job.id) == "paused")

    assert queue.cancel(job.id)
    assert wait_for(lambda: status_of(queue, job.id) == "cancelled"), queue.list()


def test_shutdown_wakes_a_paused_worker(queue):
    """A paused queue must not hold the process open on exit."""
    queue.submit(Path("a.mp4"), None, {})
    queue.pause()
    time.sleep(0.2)
    start = time.time()
    queue.shutdown()
    assert time.time() - start < 5, "shutdown should not wait on the pause"


def test_pausing_an_idle_queue_is_harmless(queue):
    queue.pause()
    job = queue.submit(Path("a.mp4"), None, {})
    time.sleep(0.3)
    assert status_of(queue, job.id) == "queued"
    queue.resume()
    assert wait_for(lambda: status_of(queue, job.id) == "done")
