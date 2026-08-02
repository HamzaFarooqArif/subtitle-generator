"""Output-name collision handling.

Two sources sharing a stem (clip.mp4 and clip.wav) written into one --out-dir
must not overwrite each other's subtitles.
"""

from __future__ import annotations

from pathlib import Path

from sgen.config import Config
from sgen.pipeline import Pipeline


def claim(pipeline, base: Path, source: Path, content_id: str) -> Path:
    return pipeline._claim(base, source, content_id)


def test_distinct_stems_are_left_alone(tmp_path):
    pipeline = Pipeline(Config())
    a = claim(pipeline, tmp_path / "one", Path("one.mp4"), "aaa")
    b = claim(pipeline, tmp_path / "two", Path("two.mp4"), "bbb")
    assert a.name == "one"
    assert b.name == "two"


def test_same_stem_different_source_is_disambiguated(tmp_path):
    pipeline = Pipeline(Config())
    first = claim(pipeline, tmp_path / "clip", Path("clip.mp4"), "aaa")
    second = claim(pipeline, tmp_path / "clip", Path("clip.wav"), "bbb")
    assert first.name == "clip"
    assert second.name == "clip_wav"
    assert first != second


def test_three_way_collision(tmp_path):
    pipeline = Pipeline(Config())
    names = [
        claim(pipeline, tmp_path / "clip", Path(f"clip{ext}"), cid).name
        for ext, cid in ((".mp4", "a"), (".wav", "b"), (".mkv", "c"))
    ]
    assert len(set(names)) == 3, names


def test_same_content_reclaims_the_same_stem(tmp_path):
    """Re-processing the same file must be idempotent, not append suffixes."""
    pipeline = Pipeline(Config())
    first = claim(pipeline, tmp_path / "clip", Path("clip.mp4"), "aaa")
    again = claim(pipeline, tmp_path / "clip", Path("clip.mp4"), "aaa")
    assert first == again


def test_repeated_same_suffix_collision_gets_a_counter(tmp_path):
    pipeline = Pipeline(Config())
    a = claim(pipeline, tmp_path / "clip", Path("x/clip.wav"), "a")
    b = claim(pipeline, tmp_path / "clip", Path("y/clip.wav"), "b")
    c = claim(pipeline, tmp_path / "clip", Path("z/clip.wav"), "c")
    assert len({a, b, c}) == 3, (a, b, c)
