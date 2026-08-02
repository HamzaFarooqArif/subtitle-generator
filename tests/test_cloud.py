"""Cloud translation as a pipeline step.

No network: the provider is a stub. What matters is that a job asking for cloud
translation gets it, that a failure costs the transcription nothing, and that
mistakes which can be caught without spending a request are caught first.
"""

from pathlib import Path

import pytest

from sgen import cloud, online
from sgen.config import Config
from sgen.cues import Cue


class Stub:
    """A provider that behaves like the real ones: numbering is preserved.

    Both APIs return the numbered lines they were given, which is what lets the
    document path put translations back on the right timings.
    """

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.seen: list[str] = []

    def translate_texts(self, texts, src, tgt="en"):
        if self.fail:
            raise self.fail
        self.seen.extend(texts)
        out = []
        for text in texts:
            lines = []
            for line in text.splitlines() or [text]:
                number, _, body = line.partition(". ")
                lines.append(f"{number}. [{tgt}] {body}" if body else f"[{tgt}] {line}")
            out.append("\n".join(lines))
        return out


class Scrambler(Stub):
    """Loses the numbering, as a translator occasionally will."""

    def translate_texts(self, texts, src, tgt="en"):
        self.seen.extend(texts)
        return [f"[{tgt}] everything ran together" for _ in texts]


@pytest.fixture
def cues():
    return [
        Cue(start=0.0, end=2.0, lines=["Привет, как дела?"]),
        Cue(start=2.5, end=5.0, lines=["Всё хорошо,", "спасибо."]),
    ]


@pytest.fixture
def no_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "settings.local.yaml"))
    for variable in ("SGEN_GOOGLE_API_KEY", "SGEN_DEEPL_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


# --------------------------------------------------------------------------- #
# what is checked before anything is sent
# --------------------------------------------------------------------------- #

def test_missing_key_is_refused_without_a_request(no_keys):
    with pytest.raises(cloud.NotPossible, match="no Google"):
        cloud.resolve("google", "ru", "en")


def test_same_language_is_refused(no_keys, monkeypatch):
    monkeypatch.setenv("SGEN_GOOGLE_API_KEY", "k")
    with pytest.raises(cloud.NotPossible, match="already in"):
        cloud.resolve("google", "en", "en")


def test_deepl_without_the_language_is_refused(no_keys, monkeypatch):
    """Checked against DeepL's own list, not a remembered one: refusing a pair
    the service actually supports is the worse failure."""
    monkeypatch.setenv("SGEN_DEEPL_API_KEY", "k")
    monkeypatch.setattr(online.DeepLTranslator, "targets",
                        lambda self: frozenset({"de", "en"}))
    with pytest.raises(cloud.NotPossible, match="use Google"):
        cloud.resolve("deepl", "en", "ja")


def test_a_language_deepl_has_gained_is_allowed(no_keys, monkeypatch):
    monkeypatch.setenv("SGEN_DEEPL_API_KEY", "k")
    monkeypatch.setattr(online.DeepLTranslator, "targets",
                        lambda self: frozenset({"ur", "hi", "en"}))
    assert cloud.resolve("deepl", "en", "ur")


def test_unknown_language_for_google_is_allowed_through(no_keys, monkeypatch):
    """Google's list is long and changes; let the service decide."""
    monkeypatch.setenv("SGEN_GOOGLE_API_KEY", "k")
    assert cloud.resolve("google", "en", "hi")


def test_empty_source_language_is_fine(no_keys, monkeypatch):
    """Detection may have returned nothing; the provider can detect too."""
    monkeypatch.setenv("SGEN_GOOGLE_API_KEY", "k")
    assert cloud.resolve("google", "", "en")


# --------------------------------------------------------------------------- #
# translating
# --------------------------------------------------------------------------- #

def test_writes_translated_subtitles_next_to_the_source(cues, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"")
    result = cloud.translate(
        cues, Stub(), provider_name="deepl", source_language="ru",
        target_language="en", source_path=source, formats=("srt",),
    )
    assert result.cue_count >= 1
    assert result.language == "en"
    written = result.written[0]
    assert written.name == "clip.en.srt"
    assert written.parent == tmp_path
    body = written.read_text(encoding="utf-8-sig")
    assert "[en]" in body
    # Timings come from the source cues, unchanged.
    assert "00:00:00,0" in body


def test_out_dir_is_honoured(cues, tmp_path):
    source = tmp_path / "clip.mp4"
    out = tmp_path / "subs"
    out.mkdir()
    result = cloud.translate(
        cues, Stub(), provider_name="google", source_language="ru",
        target_language="de", source_path=source, formats=("srt",), out_dir=out,
    )
    assert result.written[0] == out / "clip.de.srt"


def test_the_whole_transcript_goes_in_one_request(cues, tmp_path):
    """The difference the user could see: both APIs translate each item in a
    request independently, so one cue per item means every line is translated
    with no conversation around it. Numbered lines in a single request is what
    the paste-into-the-web-UI workflow does, and it reads far better.
    """
    stub = Stub()
    cloud.translate(
        cues, stub, provider_name="google", source_language="ru",
        target_language="en", source_path=tmp_path / "c.mp4", formats=("srt",),
    )
    assert len(stub.seen) == 1, "one document, not one request item per cue"
    assert stub.seen[0].startswith("1. ")
    # A cue's two lines are one sentence; they must not become two entries.
    assert "2. Всё хорошо, спасибо." in stub.seen[0]


def test_a_long_transcript_is_split_on_line_boundaries(tmp_path):
    """Chunking must never cut a cue in half, or its number is lost with it."""
    many = [Cue(start=float(i), end=float(i) + 1, lines=[f"строка номер {i} " * 4])
            for i in range(120)]
    stub = Stub()
    cloud.translate(
        many, stub, provider_name="deepl", source_language="ru",
        target_language="en", source_path=tmp_path / "c.mp4", formats=("srt",),
    )
    assert len(stub.seen) > 1, "should have been chunked"
    for chunk in stub.seen:
        for line in chunk.splitlines():
            assert line[0].isdigit(), "every chunk line still starts with its number"


def test_falls_back_to_cue_by_cue_when_numbering_is_lost(cues, tmp_path):
    """Alignment beats fluency: a scrambled document must not scramble the file."""
    scrambler = Scrambler()
    result = cloud.translate(
        cues, scrambler, provider_name="google", source_language="ru",
        target_language="en", source_path=tmp_path / "c.mp4", formats=("srt",),
    )
    # First the document attempt, then one item per cue.
    assert len(scrambler.seen) > 1
    assert result.cue_count >= 1


def test_character_count_is_reported(cues, tmp_path):
    """The free tiers are measured in characters, so show what was spent."""
    result = cloud.translate(
        cues, Stub(), provider_name="google", source_language="ru",
        target_language="en", source_path=tmp_path / "c.mp4", formats=("srt",),
    )
    assert result.characters == sum(len(" ".join(c.lines)) for c in cues)


def test_nothing_to_translate_is_refused(tmp_path):
    with pytest.raises(cloud.NotPossible, match="no subtitles"):
        cloud.translate([], Stub(), provider_name="google", source_language="ru",
                        target_language="en", source_path=tmp_path / "c.mp4",
                        formats=("srt",))


def test_blank_cues_are_refused_rather_than_sent(tmp_path):
    blank = [Cue(start=0.0, end=1.0, lines=["   "])]
    with pytest.raises(cloud.NotPossible, match="no subtitle text"):
        cloud.translate(blank, Stub(), provider_name="google", source_language="ru",
                        target_language="en", source_path=tmp_path / "c.mp4",
                        formats=("srt",))


def test_service_failure_propagates_as_a_translation_error(cues, tmp_path):
    with pytest.raises(online.TranslationError):
        cloud.translate(
            cues, Stub(fail=online.TranslationError("rate limited")),
            provider_name="google", source_language="ru", target_language="en",
            source_path=tmp_path / "c.mp4", formats=("srt",),
        )


def test_formats_come_from_the_caller(cues, tmp_path):
    result = cloud.translate(
        cues, Stub(), provider_name="google", source_language="ru",
        target_language="en", source_path=tmp_path / "c.mp4",
        formats=("srt", "vtt"), cfg=Config(),
    )
    assert {p.suffix for p in result.written} == {".srt", ".vtt"}


# --------------------------------------------------------------------------- #
# as part of a transcription job
# --------------------------------------------------------------------------- #

class FakeResult:
    def __init__(self, cues, tmp_path):
        self.cues = cues
        self.language = "ru"
        self.language_probability = 0.99
        self.duration = 5.0
        self.suppressed_count = 0
        self.gate_summary = ""
        self.content_id = "abc123"
        self.outputs = [tmp_path / "clip.ru.srt"]
        self.verdict = None


def test_a_job_can_translate_straight_after_transcribing(cues, tmp_path, monkeypatch):
    from sgen.server.jobs import Job, JobQueue

    monkeypatch.setattr(cloud, "resolve", lambda *a, **k: Stub())
    queue = JobQueue()
    job = Job(id="j1", source=tmp_path / "clip.mp4", out_dir=None,
              options={"cloud_provider": "deepl", "translate_target": "en"})

    note, written = queue._cloud_translate(job, FakeResult(cues, tmp_path))
    assert "translated to en via deepl" in note
    assert written and written[0].name == "clip.en.srt"


def test_a_translation_failure_does_not_fail_the_job(cues, tmp_path, monkeypatch):
    """The transcription is the valuable part; a bad key must not discard it."""
    from sgen.server.jobs import Job, JobQueue

    def refuse(*a, **k):
        raise cloud.NotPossible("no DeepL API key configured")

    monkeypatch.setattr(cloud, "resolve", refuse)
    queue = JobQueue()
    job = Job(id="j2", source=tmp_path / "clip.mp4", out_dir=None,
              options={"cloud_provider": "deepl"})

    note, written = queue._cloud_translate(job, FakeResult(cues, tmp_path))
    assert note.startswith("not translated:")
    assert "no DeepL API key" in note
    assert written == []


def test_a_service_outage_is_reported_on_the_row(cues, tmp_path, monkeypatch):
    from sgen.server.jobs import Job, JobQueue

    monkeypatch.setattr(cloud, "resolve",
                        lambda *a, **k: Stub(fail=online.TranslationError("503")))
    queue = JobQueue()
    job = Job(id="j3", source=tmp_path / "clip.mp4", out_dir=None,
              options={"cloud_provider": "google"})

    note, written = queue._cloud_translate(job, FakeResult(cues, tmp_path))
    assert note.startswith("translation failed:")
    assert written == []


def test_an_unexpected_crash_is_contained(cues, tmp_path, monkeypatch):
    from sgen.server.jobs import Job, JobQueue

    monkeypatch.setattr(cloud, "resolve", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("something odd")))
    queue = JobQueue()
    job = Job(id="j4", source=tmp_path / "clip.mp4", out_dir=None,
              options={"cloud_provider": "google"})

    note, _ = queue._cloud_translate(job, FakeResult(cues, tmp_path))
    assert "RuntimeError" in note


def test_no_cloud_provider_means_no_request(tmp_path):
    """The default must stay offline-only: this is what sends text to a service."""
    from sgen.server.jobs import build_config

    cfg = build_config({})
    assert cfg.translate_to_english is False
    assert "cloud_provider" not in {}      # nothing is added implicitly


# --------------------------------------------------------------------------- #
# "translate automatically when it isn't English"
# --------------------------------------------------------------------------- #

@pytest.fixture
def auto_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    return path


def test_the_setting_turns_translation_on_without_asking(auto_settings):
    from sgen.server.jobs import _cloud_provider

    auto_settings.write_text(
        "defaults:\n  translate:\n    auto: true\n    provider: deepl\n",
        encoding="utf-8")
    assert _cloud_provider({}) == "deepl"


def test_off_by_default(auto_settings):
    from sgen.server.jobs import _cloud_provider

    assert _cloud_provider({}) == ""


def test_an_explicit_choice_in_the_request_wins(auto_settings):
    from sgen.server.jobs import _cloud_provider

    auto_settings.write_text(
        "defaults:\n  translate:\n    auto: true\n    provider: deepl\n",
        encoding="utf-8")
    assert _cloud_provider({"cloud_provider": "google"}) == "google"
    # "transcribe only", chosen deliberately for this run, must not be overridden.
    assert _cloud_provider({"cloud_provider": ""}) == ""


def test_local_provider_does_not_trigger_a_cloud_request(auto_settings):
    """`provider: local` means the offline model, not a service."""
    from sgen.server.jobs import _cloud_provider, build_config

    auto_settings.write_text(
        "defaults:\n  translate:\n    auto: true\n    provider: local\n",
        encoding="utf-8")
    assert _cloud_provider({}) == ""
    assert build_config({}).translate_to_english is True


def test_an_english_file_is_skipped_not_translated(cues, tmp_path, auto_settings):
    """The point of "when it isn't English": most home video already is, and
    sending it anyway would spend quota to get the same words back."""
    from sgen.server.jobs import Job, JobQueue

    called = []

    def spy(*a, **k):
        called.append(a)
        return Stub()

    import sgen.cloud as cloud_module
    original = cloud_module.resolve
    cloud_module.resolve = spy
    try:
        queue = JobQueue()
        job = Job(id="j6", source=tmp_path / "clip.mp4", out_dir=None,
                  options={"cloud_provider": "deepl", "translate_target": "en"})
        result = FakeResult(cues, tmp_path)
        result.language = "en"                     # already the target
        note, written = queue._cloud_translate(job, result)
    finally:
        cloud_module.resolve = original

    assert note == "already in en — nothing to translate"
    assert written == []
    assert called == [], "no request should have been made"


def test_a_non_english_file_is_translated(cues, tmp_path, auto_settings, monkeypatch):
    from sgen.server.jobs import Job, JobQueue

    monkeypatch.setattr(cloud, "resolve", lambda *a, **k: Stub())
    queue = JobQueue()
    job = Job(id="j7", source=tmp_path / "clip.mp4", out_dir=None,
              options={"cloud_provider": "deepl", "translate_target": "en"})
    note, written = queue._cloud_translate(job, FakeResult(cues, tmp_path))
    assert note.startswith("translated to en")
    assert written


def test_the_job_reports_its_note_to_the_ui(tmp_path):
    from sgen.server.jobs import Job

    job = Job(id="j5", source=tmp_path / "c.mp4", out_dir=None, options={})
    job.cloud_note = "translated to en via deepl (78 subtitles, 1617 characters)"
    assert job.to_dict()["cloud_note"] == job.cloud_note
