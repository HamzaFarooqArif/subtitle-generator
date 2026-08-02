"""English translation output via Whisper's translate task.

The model-level behaviour needs a GPU, so those tests skip without one. The
wiring — config plumbing, stage weights, VAD inheritance — is tested without.
"""

from pathlib import Path

import pytest

from sgen.config import Config, enforce_offline


def test_config_flag_defaults_off():
    assert Config().translate_to_english is False


def test_profile_can_enable_translation(tmp_path, monkeypatch):
    import yaml
    from sgen import config as config_mod

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "xlate.yaml").write_text(
        yaml.safe_dump({"translate_to_english": True, "romanize": True}), encoding="utf-8"
    )
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    cfg = Config.load("xlate")
    assert cfg.translate_to_english is True
    assert cfg.romanize is True


def test_ui_option_maps_to_config():
    from sgen.server.jobs import build_config

    assert build_config({"translate": True}).translate_to_english is True
    assert build_config({"translate": False}).translate_to_english is False
    assert build_config({}).translate_to_english is False


def test_translate_stage_has_progress_weight():
    """Without a weight, the progress bar would stall through the whole pass."""
    from sgen.server.jobs import STAGES, STAGE_WEIGHTS, _overall

    assert "translate" in STAGES
    assert STAGE_WEIGHTS["translate"] > 0.1
    assert abs(sum(STAGE_WEIGHTS.values()) - 1.0) < 1e-6
    # Monotonic across the pipeline.
    assert _overall("transcribe", 1.0) < _overall("translate", 0.5) < _overall("write", 0.0)


def test_recognizer_accepts_task_without_touching_the_gpu():
    """The task parameter must reach the underlying kwargs."""
    from sgen.asr import Recognizer
    from sgen.config import AsrConfig

    # Build kwargs via the unbound method to avoid loading a model.
    kwargs = Recognizer._kwargs(
        type("Stub", (), {"cfg": AsrConfig()})(), "hi", task="translate"
    )
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "hi"


# --------------------------------------------------------------------------- #
# GPU end-to-end
# --------------------------------------------------------------------------- #

FIXTURE = Path(__file__).parent / "fixtures" / "german.mp4"


def _models_present() -> bool:
    enforce_offline()
    from sgen import models

    try:
        models.ct2_path("large-v3")
        return True
    except models.ModelMissing:
        return False


gpu = pytest.mark.skipif(
    not FIXTURE.exists() or not _models_present(),
    reason="needs the fixture and downloaded models",
)


@gpu
def test_translation_writes_english_file(tmp_path):
    """German fixture -> both de and en subtitle files."""
    from sgen.pipeline import Pipeline

    enforce_offline()
    cfg = Config.load("home-video")
    cfg.asr.language = "de"
    cfg.translate_to_english = True
    cfg.formats = ("srt",)

    with Pipeline(cfg) as pipeline:
        result = pipeline.process(FIXTURE, out_dir=tmp_path, overwrite=True)

    names = sorted(p.name for p in result.outputs)
    assert any(n.endswith(".de.srt") for n in names), names
    assert any(n.endswith(".en.srt") for n in names), names

    english = next(p for p in result.outputs if p.name.endswith(".en.srt"))
    body = english.read_text(encoding="utf-8-sig")
    # The fixture says "Guten Morgen ... Fluss ... Bruder Thomas".
    lowered = body.lower()
    assert any(w in lowered for w in ("morning", "river", "brother")), body[:400]
    # And it must NOT still be German.
    assert "guten morgen" not in lowered


@gpu
def test_english_audio_skips_the_translation_pass(tmp_path):
    """Translating English to English is wasted work; it must be skipped."""
    from sgen.pipeline import Pipeline

    fixture = Path(__file__).parent / "fixtures" / "sample.mp4"
    if not fixture.exists():
        pytest.skip("English fixture not built")

    enforce_offline()
    cfg = Config.load("home-video")
    cfg.asr.language = "en"
    cfg.translate_to_english = True
    cfg.formats = ("srt",)

    with Pipeline(cfg) as pipeline:
        result = pipeline.process(fixture, out_dir=tmp_path, overwrite=True)

    # Exactly one output: the English transcription, no duplicate translation.
    assert len(result.outputs) == 1, [p.name for p in result.outputs]
