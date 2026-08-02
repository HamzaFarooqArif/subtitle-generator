"""Online translation providers (Google, DeepL).

HTTP is mocked throughout: these tests must never make a network call or need a
key. What matters is the contract the surrounding pipeline depends on — one
translation per input, in order — plus clear errors and no key leakage.
"""

import json
import urllib.error
from unittest.mock import patch

import pytest

from sgen import online


# --------------------------------------------------------------------------- #
# key storage
# --------------------------------------------------------------------------- #

@pytest.fixture
def temp_keys(tmp_path, monkeypatch):
    """Point the settings file at a temp path; keys live there now."""
    path = tmp_path / "settings.local.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    monkeypatch.delenv("SGEN_GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("SGEN_DEEPL_API_KEY", raising=False)
    return path


def test_no_keys_by_default(temp_keys):
    keys = online.load_keys()
    assert keys.google == "" and keys.deepl == ""
    assert online.configured() == {"google": False, "deepl": False}


def test_keys_round_trip(temp_keys):
    online.save_keys(online.Keys(google="g-key", deepl="d-key", deepl_plan="pro"))
    keys = online.load_keys()
    assert keys.google == "g-key"
    assert keys.deepl == "d-key"
    assert keys.deepl_plan == "pro"
    assert online.configured() == {"google": True, "deepl": True}


def test_corrupt_keys_file_does_not_crash(temp_keys):
    temp_keys.write_text("api_keys: [not, a, mapping", encoding="utf-8")
    assert online.load_keys().google == ""


def test_keys_file_is_gitignored():
    from pathlib import Path

    ignore = (Path(__file__).parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "settings.local.yaml" in ignore, "API keys must never be committable"


def test_missing_key_raises_before_any_request(temp_keys):
    with pytest.raises(online.TranslationError, match="no Google"):
        online.get_translator("google")
    with pytest.raises(online.TranslationError, match="no DeepL"):
        online.get_translator("deepl")


def test_unknown_provider_rejected():
    with pytest.raises(online.TranslationError, match="unknown"):
        online.get_translator("bing")


# --------------------------------------------------------------------------- #
# Google
# --------------------------------------------------------------------------- #

def google_reply(texts):
    return {"data": {"translations": [{"translatedText": f"EN::{t}"} for t in texts]}}


def test_google_returns_one_translation_per_input():
    captured = {}

    def fake_post(url, *, data=None, headers=None, **kw):
        captured["url"] = url
        captured["body"] = data.decode()
        from urllib.parse import parse_qs
        qs = parse_qs(captured["body"])
        return google_reply(qs["q"])

    with patch.object(online, "_post", fake_post):
        out = online.GoogleTranslator("k").translate_texts(["one", "two"], "ru", "en")

    assert out == ["EN::one", "EN::two"]
    assert "target=en" in captured["body"]
    assert "source=ru" in captured["body"]


def test_google_batches_large_inputs():
    calls = []

    def fake_post(url, *, data=None, headers=None, **kw):
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        calls.append(len(qs["q"]))
        return google_reply(qs["q"])

    with patch.object(online, "_post", fake_post):
        out = online.GoogleTranslator("k").translate_texts([f"s{i}" for i in range(250)], "ru")

    assert len(out) == 250
    assert len(calls) == 3 and max(calls) <= online.GoogleTranslator.max_batch


def test_google_unescapes_html_entities():
    """Google returns entities even with format=text."""
    with patch.object(online, "_post",
                      lambda *a, **k: {"data": {"translations": [
                          {"translatedText": "it&#39;s fine &amp; good"}]}}):
        out = online.GoogleTranslator("k").translate_texts(["x"], "ru")
    assert out == ["it's fine & good"]


def test_google_count_mismatch_is_an_error():
    with patch.object(online, "_post",
                      lambda *a, **k: {"data": {"translations": [{"translatedText": "only one"}]}}):
        with pytest.raises(online.TranslationError, match="1 translations for 2"):
            online.GoogleTranslator("k").translate_texts(["a", "b"], "ru")


def test_google_unexpected_shape_is_an_error():
    with patch.object(online, "_post", lambda *a, **k: {"oops": True}):
        with pytest.raises(online.TranslationError, match="unexpected Google"):
            online.GoogleTranslator("k").translate_texts(["a"], "ru")


# --------------------------------------------------------------------------- #
# DeepL
# --------------------------------------------------------------------------- #

def test_deepl_returns_one_translation_per_input():
    captured = {}

    def fake_post(url, *, data=None, headers=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        captured["target"] = qs["target_lang"][0]
        return {"translations": [{"text": f"EN::{t}"} for t in qs["text"]]}

    with patch.object(online, "_post", fake_post):
        out = online.DeepLTranslator("k").translate_texts(["one", "two"], "ru", "en")

    assert out == ["EN::one", "EN::two"]
    assert "api-free" in captured["url"]
    assert captured["headers"]["Authorization"].startswith("DeepL-Auth-Key ")
    assert captured["target"] == "EN-US"      # DeepL wants a regional English


def test_deepl_pro_uses_the_paid_host():
    assert online.DeepLTranslator("k", plan="pro")._url == online.DEEPL_PRO


def test_deepl_free_key_suffix_overrides_the_plan_setting():
    """Free and Pro are different hosts, and the wrong one returns a bare 403
    that reads like a bad key. DeepL suffixes free keys with ":fx", so the host
    is chosen from the key rather than from a dropdown the user may misread.
    """
    translator = online.DeepLTranslator("abc-123:fx", plan="pro")
    assert translator._url == online.DEEPL_FREE


def test_deepl_pro_key_without_suffix_respects_pro():
    assert online.DeepLTranslator("abc-123", plan="pro")._url == online.DEEPL_PRO


def test_deepl_asks_deepl_which_languages_it_has():
    """A hardcoded list went two years stale — it claimed no Hindi and no Urdu
    when the service reports 110 target languages including both. Refusing a
    pair the service supports is worse than sending it.
    """
    online._deepl_target_cache.clear()
    reported = [{"language": c} for c in ("EN-US", "UR", "HI", "DE", "ZH-HANS")]

    class Reply:
        def read(self): return json.dumps(reported).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(online.urllib.request, "urlopen", lambda *a, **k: Reply()):
        translator = online.DeepLTranslator("live-key")
        assert translator.can_target("ur"), "Urdu is in the list DeepL returns"
        assert translator.can_target("hi")
        assert translator.can_target("en"), "en-us also means en"
        assert not translator.can_target("xx")


def test_the_language_list_is_fetched_once_per_key():
    online._deepl_target_cache.clear()
    calls = []

    class Reply:
        def read(self): return json.dumps([{"language": "UR"}]).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def spy(*a, **k):
        calls.append(a)
        return Reply()

    with patch.object(online.urllib.request, "urlopen", spy):
        translator = online.DeepLTranslator("k2")
        translator.can_target("ur")
        translator.can_target("de")
        translator.can_target("ur")
    assert len(calls) == 1, "the list changes on DeepL's release schedule, not per cue"


def test_a_failed_lookup_falls_back_instead_of_blocking():
    """No network or a bad key must not stop a translation that might work."""
    online._deepl_target_cache.clear()

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    with patch.object(online.urllib.request, "urlopen", boom):
        translator = online.DeepLTranslator("k3")
        assert translator.can_target("de"), "the built-in list still has German"
        assert not translator.can_target("ur"), "and is honest about not knowing"


def test_an_unreachable_language_is_refused_before_sending():
    online._deepl_target_cache.clear()

    class Reply:
        def read(self): return json.dumps([{"language": "DE"}]).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(online.urllib.request, "urlopen", lambda *a, **k: Reply()):
        with pytest.raises(online.TranslationError, match="use Google"):
            online.DeepLTranslator("k4").translate_texts(["x"], "de", "ja")


def test_deepl_sends_surrounding_cues_as_context():
    """Each cue is translated on its own — that is what keeps it aligned to its
    timing — so the scene has to be supplied separately or a one-word cue is
    translated with nothing around it.
    """
    captured = {}

    def fake_post(url, *, data=None, headers=None, **kw):
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        captured.setdefault("contexts", []).append(qs.get("context", [""])[0])
        captured.setdefault("models", []).append(qs.get("model_type", [""])[0])
        return {"translations": [{"text": t} for t in qs["text"]]}

    texts = [f"line {i}" for i in range(60)]
    with patch.object(online, "_post", fake_post):
        online.DeepLTranslator("k").translate_texts(texts, "ru", "en")

    first = captured["contexts"][0]
    assert "line 25" in first, "must carry the lines that follow this batch"
    assert "line 0" not in first.split("line 1")[0], "own batch is not the context"
    second = captured["contexts"][1]
    assert "line 15" in second and "line 50" in second, "both sides"
    # The next-generation model, which is what handles idiom and register.
    assert set(captured["models"]) == {"prefer_quality_optimized"}


def test_a_short_file_uses_itself_as_context():
    """One batch has nothing around it, and that is exactly when the rest of the
    file is the missing context."""
    captured = {}

    def fake_post(url, *, data=None, headers=None, **kw):
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        captured["context"] = qs.get("context", [""])[0]
        return {"translations": [{"text": t} for t in qs["text"]]}

    with patch.object(online, "_post", fake_post):
        online.DeepLTranslator("k").translate_texts(
            ["At work? Yes.", "мусора", "not like that"], "ru", "en")

    assert "At work?" in captured["context"] and "not like that" in captured["context"]


def test_context_is_bounded():
    """Free to send, but it still has to fit in a request."""
    captured = {}

    def fake_post(url, *, data=None, headers=None, **kw):
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        captured["context"] = qs.get("context", [""])[0]
        return {"translations": [{"text": t} for t in qs["text"]]}

    with patch.object(online, "_post", fake_post):
        online.DeepLTranslator("k").translate_texts(["x" * 500] * 40, "ru", "en")

    assert len(captured["context"]) <= online.DeepLTranslator.max_context_chars


def test_deepl_batches_at_twenty_five():
    calls = []

    def fake_post(url, *, data=None, headers=None, **kw):
        from urllib.parse import parse_qs
        qs = parse_qs(data.decode())
        calls.append(len(qs["text"]))
        return {"translations": [{"text": t} for t in qs["text"]]}

    with patch.object(online, "_post", fake_post):
        out = online.DeepLTranslator("k").translate_texts([f"s{i}" for i in range(120)], "ru")

    assert len(out) == 120
    assert max(calls) <= online.DeepLTranslator.max_batch


# --------------------------------------------------------------------------- #
# HTTP error handling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code,needle", [
    (403, "rejected the key"),
    (401, "rejected the key"),
    (429, "rate limited"),
    (456, "quota"),
    (400, "unsupported language"),
])
def test_http_errors_explained_in_plain_language(code, needle):
    assert needle in online._explain(code, "body")


def test_retries_then_gives_up(monkeypatch):
    import urllib.error

    attempts = {"n": 0}

    def always_503(*a, **k):
        attempts["n"] += 1
        raise urllib.error.HTTPError("u", 503, "busy", {}, None)

    monkeypatch.setattr(online.urllib.request, "urlopen", always_503)
    monkeypatch.setattr(online.time, "sleep", lambda s: None)
    with pytest.raises(online.TranslationError):
        online._post("https://example.invalid", data=b"x")
    assert attempts["n"] > 1, "must retry transient failures"


def test_key_never_appears_in_an_error_message():
    """A raised error may be shown in the UI; it must not carry the key."""
    msg = online._explain(403, "some provider body")
    assert "AIza" not in msg


# --------------------------------------------------------------------------- #
# integration with the shared pipeline
# --------------------------------------------------------------------------- #

def test_provider_plugs_into_the_shared_translate_pipeline():
    """Online providers must satisfy the same interface as the local model."""
    from sgen import translate as mt
    from sgen.asr import Segment, Word
    from sgen.config import CueConfig

    class Stub:
        def translate_texts(self, texts, src, tgt="en"):
            return [f"translated {i}" for i, _ in enumerate(texts, 1)]

    segments = [
        Segment(start=float(i), end=float(i) + 2.0, text=f"line {i}.",
                words=[Word(float(i), float(i) + 2.0, f" line {i}.", 1.0)])
        for i in range(3)
    ]
    out = mt.translate_segments(Stub(), segments, "ru", "en", cue_cfg=CueConfig())
    assert len(out) == 3
    assert out[0].text.startswith("translated")
    # Timings preserved from the source.
    assert out[0].start == 0.0
