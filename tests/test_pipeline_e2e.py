"""End-to-end test against a synthetic fixture.

Requires the GPU and downloaded models, so it is skipped unless the fixture
exists (build it with tests/make_fixture.ps1) and the model resolves locally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sgen.config import Config, enforce_offline

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"

# Content words that must appear in the transcript of the fixture script.
EXPECTED_TERMS = ("river", "bridge", "rain", "jacket", "afternoon")


def _models_present() -> bool:
    enforce_offline()
    from sgen import models

    try:
        models.ct2_path("large-v3")
        return True
    except models.ModelMissing:
        return False


pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not built"),
    pytest.mark.skipif(not _models_present(), reason="models not downloaded"),
]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    enforce_offline()
    from sgen.pipeline import Pipeline

    cfg = Config.load("home-video")
    cfg.asr.language = "en"  # fixture is English; keeps the test deterministic
    out = tmp_path_factory.mktemp("out")
    with Pipeline(cfg) as pipeline:
        return pipeline.process(FIXTURE, out_dir=out, overwrite=True)


def test_produces_cues(result):
    assert result.cues, "no cues produced"


def test_transcribes_expected_content(result):
    text = " ".join(c.flat for c in result.cues).lower()
    missing = [t for t in EXPECTED_TERMS if t not in text]
    assert not missing, f"missing terms {missing} in: {text!r}"


def test_writes_srt_and_vtt(result):
    suffixes = {p.suffix for p in result.outputs}
    assert suffixes == {".srt", ".vtt"}
    for path in result.outputs:
        assert path.exists() and path.stat().st_size > 0


def test_srt_is_well_formed(result):
    srt = next(p for p in result.outputs if p.suffix == ".srt")
    # utf-8-sig: output carries a BOM so Windows tools don't guess cp1252.
    body = srt.read_text(encoding="utf-8-sig")
    # index / timecode / text blocks
    assert re.search(r"^1\r?\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", body)
    assert body.count("-->") == len(result.cues)


def test_srt_carries_a_utf8_bom(result):
    srt = next(p for p in result.outputs if p.suffix == ".srt")
    assert srt.read_bytes().startswith(b"\xef\xbb\xbf")


def test_respects_line_and_reading_limits(result):
    cfg = Config.load("home-video").cues
    for cue in result.cues:
        assert len(cue.lines) <= cfg.max_lines, cue.lines
        for line in cue.lines:
            assert len(line) <= cfg.max_chars_per_line or "overlong" in cue.warnings


def test_cues_monotonic_and_non_overlapping(result):
    for a, b in zip(result.cues, result.cues[1:]):
        assert a.end <= b.start, (a.text, b.text)


def test_noise_passage_produces_no_subtitle(result):
    """The 3 s pink-noise block must not be transcribed into words.

    This is the behaviour the gating stage exists for: the noise starts right
    after the first spoken pass ends.
    """
    # min(), not max(): the fixture speaks the script twice, and we want the end
    # of the *first* pass, which is where the noise block begins. Using max()
    # would point past the end of the file and pass vacuously.
    speech_end = min(
        (c.end for c in result.cues if "afternoon" in c.flat.lower()), default=None
    )
    if speech_end is None:
        pytest.skip("could not locate end of first spoken pass")
    in_noise = [c for c in result.cues if speech_end < c.start < speech_end + 3.0]
    assert not in_noise, f"noise transcribed as: {[c.flat for c in in_noise]}"


def test_sidecar_round_trips(result, tmp_path):
    from sgen.pipeline import reformat_from_sidecar

    assert result.sidecar and result.sidecar.exists()
    cfg = Config.load("home-video")
    cfg.cues.max_chars_per_line = 30  # narrower than the run that produced it
    outputs = reformat_from_sidecar(result.sidecar, cfg)
    assert outputs
    for path in outputs:
        assert path.exists()
