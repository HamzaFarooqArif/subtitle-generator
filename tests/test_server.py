"""API tests for the local web UI.

These exercise the HTTP surface without touching the GPU: job submission is
tested for validation and queueing only, not for completion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sgen.config import WORK_DIR  # noqa: E402
from sgen.server.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "sgen" in res.text
    # The two things the page is for: queue files, translate them.
    assert 'id="listing"' in res.text
    assert 'id="translate-panel"' in res.text


def test_library_lists_transcripts_from_disk(client):
    """The queue is in-memory, so a restart lost every finished file — and with
    it the Translate button, which only appears on a finished file. The library
    is read from the sidecars so nothing becomes unreachable.
    """
    res = client.get("/api/library")
    assert res.status_code == 200
    items = res.json()["items"]
    assert isinstance(items, list)
    for item in items:
        assert item["content_id"]
        assert item["cue_count"] > 0
        assert "language" in item
        assert "source_exists" in item
    # Newest first, so the file just worked on is at the top.
    stamps = [i["modified"] for i in items]
    assert stamps == sorted(stamps, reverse=True)


def test_library_is_surfaced_in_the_page(client):
    body = client.get("/").text
    assert 'id="library"' in body
    assert "Already transcribed" in body


def test_editor_tab_is_gone(client):
    """The cue editor and waveform player were removed to simplify the page."""
    body = client.get("/")
    assert 'id="waveform"' not in body.text
    assert 'data-view="review"' not in body.text


def test_external_translation_carries_a_visible_privacy_warning(client):
    """The one action that sends data off the machine must say so plainly.

    A faint hint inside a collapsed section is not enough for a tool built around
    keeping personal recordings local.
    """
    body = client.get("/").text
    # A real callout, not hint-styled body text, inside the translate panel.
    assert "sends your transcript" in body
    assert 'class="notice"' in body
    assert "never leaves this machine" in body

    css = client.get("/static/style.css").text
    assert ".notice {" in css


def test_index_is_never_cached(client):
    """A stale index.html against fresh app.js killed the whole frontend.

    New JS looking up an element that only exists in the new HTML returned null,
    threw at module scope, and took every other listener down with it — the file
    browser stopped responding for reasons that had nothing to do with it.
    """
    res = client.get("/")
    cache = res.headers.get("cache-control", "")
    assert "no-store" in cache, cache


def test_asset_urls_are_version_stamped(client):
    """A changed asset must be a changed URL, or browsers keep the old one."""
    import re

    body = client.get("/").text
    stamped = re.findall(r"/static/(?:app\.js|style\.css)\?v=\d+", body)
    assert len(stamped) == 2, stamped
    # And the stamped URL must actually serve.
    url = re.search(r"(/static/app\.js\?v=\d+)", body).group(1)
    assert client.get(url).status_code == 200


def test_frontend_survives_a_missing_element(client):
    """$() must return an inert node rather than null."""
    js = client.get("/static/app.js").text
    assert "document.createElement" in js
    assert "no element matches" in js


def test_static_assets_served(client):
    for asset in ("/static/app.js", "/static/style.css"):
        res = client.get(asset)
        assert res.status_code == 200, asset
        assert res.content


def test_meta_reports_profiles_and_defaults(client):
    data = client.get("/api/meta").json()
    assert "home-video" in data["profiles"]
    assert "gating" in data["defaults"]
    assert "max_no_speech_prob" in data["defaults"]["gating"]
    assert "max_chars_per_line" in data["defaults"]["cues"]


def test_scan_reports_what_a_folder_still_needs(client, tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_subtitles

    (tmp_path / "one.mp4").write_bytes(b"x")
    (tmp_path / "two.mp4").write_bytes(b"x")
    write_subtitles([Cue(start=0.0, end=1.0, lines=["hi"])],
                    tmp_path / "one", ("srt",), "en")

    res = client.post("/api/scan", json={
        "folder": str(tmp_path), "options": {"formats": ["srt"]}})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert data["counts"]["done"] == 1
    assert data["counts"]["pending"] == 1
    assert "already done" in data["summary"]


def test_scan_rejects_a_path_that_is_not_a_folder(client, tmp_path):
    lonely = tmp_path / "clip.mp4"
    lonely.write_bytes(b"x")
    assert client.post("/api/scan", json={"folder": str(lonely)}).status_code == 400


def test_submitting_a_folder_skips_finished_files(client, tmp_path):
    """The resumable path: queue a folder, and files that already have subtitles
    for these settings are not queued again."""
    from sgen.cues import Cue
    from sgen.write import write_subtitles

    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (tmp_path / name).write_bytes(b"x")
    write_subtitles([Cue(start=0.0, end=1.0, lines=["hi"])],
                    tmp_path / "a", ("srt",), "en")

    res = client.post("/api/jobs", json={
        "paths": [str(tmp_path)],
        "options": {"formats": ["srt"], "model": "", "overwrite": False},
        "skip_done": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["skipped_count"] == 1
    assert {Path(j["path"]).name for j in data["jobs"]} == {"b.mp4", "c.mp4"}

    for job in data["jobs"]:
        client.delete(f"/api/jobs/{job['id']}")


def test_overwrite_beats_skip_done(client, tmp_path):
    """"Redo completed work" has to mean it, even in folder mode."""
    from sgen.cues import Cue
    from sgen.write import write_subtitles

    (tmp_path / "a.mp4").write_bytes(b"x")
    write_subtitles([Cue(start=0.0, end=1.0, lines=["hi"])],
                    tmp_path / "a", ("srt",), "en")

    res = client.post("/api/jobs", json={
        "paths": [str(tmp_path)],
        "options": {"formats": ["srt"], "overwrite": True},
        "skip_done": True,
    })
    data = res.json()
    assert data["skipped_count"] == 0
    assert len(data["jobs"]) == 1
    for job in data["jobs"]:
        client.delete(f"/api/jobs/{job['id']}")


def test_folder_mode_is_in_the_page(client):
    body = client.get("/").text
    assert 'id="btn-scan-folder"' in body
    assert 'id="btn-queue-folder"' in body
    assert 'id="opt-recursive"' in body


def test_providers_report_which_languages_each_engine_reaches(tmp_path, monkeypatch):
    """The UI builds its language list from this, so it has to be the truth.

    With no key there is nobody to ask, so the built-in list is used — old and
    conservative on purpose. The live list is covered in test_online.py.
    """
    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "s.yaml"))
    for variable in ("SGEN_GOOGLE_API_KEY", "SGEN_DEEPL_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    with TestClient(create_app()) as c:
        targets = c.get("/api/translate/providers").json()["targets"]

    assert "de" in targets["deepl"], "the fallback list still has German"
    assert "ur" in targets["local"], "NLLB has Urdu"
    assert targets["google"] == [], "empty means no restriction"


def test_deepls_own_list_is_used_when_there_is_a_key(tmp_path, monkeypatch):
    """A written-down list claimed DeepL had no Urdu; the service reports 110
    target languages including it. Asking beats remembering."""
    from sgen import online

    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "s.yaml"))
    monkeypatch.setenv("SGEN_DEEPL_API_KEY", "test-key")
    online._deepl_target_cache.clear()
    monkeypatch.setattr(
        online.DeepLTranslator, "targets",
        lambda self: frozenset({"en", "de", "ur", "hi"}),
    )

    with TestClient(create_app()) as c:
        targets = c.get("/api/translate/providers").json()["targets"]

    assert "ur" in targets["deepl"]
    assert "hi" in targets["deepl"]


def test_always_translate_can_be_saved_and_survives_a_restart(tmp_path, monkeypatch):
    """A setting kept in browser storage is a setting you cannot find again, so
    this goes into the same file the user edits by hand."""
    path = tmp_path / "settings.local.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    with TestClient(create_app()) as c:
        assert c.get("/api/meta").json()["defaults"]["translate_auto"] is False

        res = c.post("/api/translate/default",
                     json={"auto": True, "provider": "deepl", "target": "en"})
        assert res.status_code == 200 and res.json()["auto"] is True
        assert "auto: true" in path.read_text(encoding="utf-8")

    # A new server reads it back — this is what "automatically" has to mean.
    with TestClient(create_app()) as c:
        defaults = c.get("/api/meta").json()["defaults"]
        assert defaults["translate_auto"] is True
        assert defaults["translate_provider"] == "deepl"


def test_an_unknown_provider_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "s.yaml"))
    with TestClient(create_app()) as c:
        assert c.post("/api/translate/default",
                      json={"auto": True, "provider": "bing"}).status_code == 400


def test_a_combination_that_cannot_run_is_not_saved(tmp_path, monkeypatch):
    """Storing DeepL+Urdu would fail on every file from now on."""
    path = tmp_path / "s.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    with TestClient(create_app()) as c:
        res = c.post("/api/translate/default",
                     json={"auto": True, "provider": "deepl", "target": "ur"})
        assert res.status_code == 400
        assert "use Google" in res.json()["detail"]
        assert not path.exists(), "nothing should have been written"
        # The same language through Google is fine.
        assert c.post("/api/translate/default",
                      json={"auto": True, "provider": "google", "target": "ur"}
                      ).status_code == 200


def test_the_always_translate_control_is_on_the_page(client):
    body = client.get("/").text
    assert 'id="opt-translate-remember"' in body
    assert "isn't already in that language" in body


def test_cloud_translation_is_offered_in_settings(client):
    """It used to be reachable only from a panel two screens down."""
    body = client.get("/").text
    assert 'id="opt-translate-mode"' in body
    assert 'value="google"' in body and 'value="deepl"' in body
    # And a way to get a key from where the choice is made.
    assert 'id="btn-open-keys"' in body


def test_cloud_translation_is_off_unless_chosen(client):
    """The default must not send anything anywhere."""
    body = client.get("/").text
    mode = body[body.index('id="opt-translate-mode"'):][:400]
    default = mode[mode.index("<option"):mode.index("</option>")]
    assert 'value="none"' in default and "selected" in default


def test_vtt_checkbox_starts_unchecked(client):
    """The markup must agree with defaults.formats, or the box flashes checked
    for the moment before settings arrive and then quietly changes itself."""
    body = client.get("/").text
    srt = body[body.index('id="fmt-srt"'):][:60]
    vtt = body[body.index('id="fmt-vtt"'):][:60]
    assert "checked" in srt
    assert "checked" not in vtt


def test_translation_is_reachable_from_the_top_of_the_page(client):
    """Google/DeepL translation was invisible: its only entry point was a button
    ~1300px down the page, so the feature read as missing. It now has a tab.
    """
    body = client.get("/").text
    assert 'data-view="translate"' in body
    assert 'id="translate-picker"' in body
    # The providers are named where someone looking for them would look.
    assert "Google Translate" in body and "DeepL" in body


def test_page_can_announce_that_the_backend_is_gone(client):
    """A tab whose server was replaced looks fine and silently drops clicks.

    That is indistinguishable from a broken page — it cost a debugging session
    already — so the markup for saying it out loud has to be present.
    """
    body = client.get("/").text
    assert 'id="offline"' in body
    assert 'id="btn-reload"' in body
    assert "Lost contact with the app" in body


def test_favicon_is_served(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")


def test_meta_points_at_the_settings_file(client):
    """The page has to be able to tell the user where to put their keys."""
    data = client.get("/api/meta").json()
    assert data["settings"]["path"].endswith((".yaml", ".yml"))
    assert "language" in data["defaults"]
    assert "out_dir" in data["defaults"]
    assert "translate_provider" in data["defaults"]


def test_settings_file_is_named_in_the_page(client):
    body = client.get("/").text
    assert 'id="settings-path"' in body
    assert "sgen config --init" in body


def test_hand_edited_settings_reach_the_ui(tmp_path, monkeypatch):
    """Editing the file must not need a server restart: it is re-read per call."""
    path = tmp_path / "settings.local.yaml"
    path.write_text(
        "defaults:\n  profile: verbatim\n  hotwords: Oaxaca\n  romanize: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    with TestClient(create_app()) as c:
        defaults = c.get("/api/meta").json()["defaults"]
        assert defaults["profile"] == "verbatim"
        assert defaults["hotwords"] == "Oaxaca"
        assert defaults["romanize"] is True

        path.write_text("defaults:\n  profile: music\n", encoding="utf-8")
        assert c.get("/api/meta").json()["defaults"]["profile"] == "music"


def test_an_unknown_profile_in_settings_does_not_break_the_page(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.yaml"
    path.write_text("defaults:\n  profile: nonexistent\n", encoding="utf-8")
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    with TestClient(create_app()) as c:
        data = c.get("/api/meta").json()
        assert data["settings"]["error"]           # says what is wrong
        assert data["defaults"]["profile"] == "home-video"   # still usable


def test_saving_keys_writes_the_settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.local.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(path))
    monkeypatch.delenv("SGEN_GOOGLE_API_KEY", raising=False)
    with TestClient(create_app()) as c:
        res = c.post("/api/translate/keys", json={"google": "AIza-test"})
        assert res.status_code == 200
        assert res.json()["configured"]["google"] is True
        assert res.json()["path"] == str(path)
        assert "AIza-test" in path.read_text(encoding="utf-8")
        # The key itself must never come back out of the API.
        assert "AIza-test" not in c.get("/api/translate/providers").text


def test_out_dir_defaults_to_the_setting(tmp_path, monkeypatch):
    """A collect-everything-here folder set once should apply to new jobs."""
    from sgen.server import app as app_module

    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "s.yaml"))
    (tmp_path / "s.yaml").write_text(
        f'defaults:\n  out_dir: "{tmp_path.as_posix()}/subs"\n', encoding="utf-8"
    )
    assert app_module._out_dir(None) == Path(tmp_path / "subs")
    assert app_module._out_dir("D:/elsewhere") == Path("D:/elsewhere")


def test_drives_lists_something(client):
    data = client.get("/api/drives").json()
    assert data["drives"]
    assert Path(data["home"]).exists()


def test_browse_lists_media_and_dirs(client, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")

    data = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert [d["name"] for d in data["dirs"]] == ["sub"]
    names = [f["name"] for f in data["files"]]
    assert "clip.mp4" in names
    assert "notes.txt" not in names, "non-media files must be filtered out"
    assert data["parent"] == str(tmp_path.parent)


def test_browse_rejects_a_file_path(client, tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    res = client.get("/api/browse", params={"path": str(target)})
    assert res.status_code == 400


def test_submit_rejects_selection_with_no_media(client, tmp_path):
    res = client.post("/api/jobs", json={"paths": [str(tmp_path / "nope.mp4")], "options": {}})
    assert res.status_code == 400
    assert "no media" in res.json()["detail"].lower()


def test_jobs_list_is_empty_initially(client):
    assert isinstance(client.get("/api/jobs").json()["jobs"], list)


def test_cancel_unknown_job_is_a_conflict(client):
    assert client.delete("/api/jobs/deadbeef").status_code == 409


def test_result_for_unknown_id_is_404(client):
    assert client.get("/api/result/nosuchid").status_code == 404
    assert client.get("/api/audio/nosuchid").status_code == 404


def test_profile_rejects_unknown_option(client):
    res = client.post("/api/profile/home-video", json={"gating": {"not_a_real_key": 1}})
    assert res.status_code == 400
    assert "unknown" in res.json()["detail"].lower()


def test_profile_rejects_unknown_profile(client):
    res = client.post("/api/profile/doesnotexist", json={"gating": {}})
    assert res.status_code == 404


def test_profile_roundtrip_preserves_file(client):
    """Saving the current values back must not corrupt the profile."""
    import yaml

    path = Path(__file__).parent.parent / "profiles" / "home-video.yaml"
    before = yaml.safe_load(path.read_text(encoding="utf-8"))
    original = path.read_text(encoding="utf-8")
    try:
        res = client.post(
            "/api/profile/home-video",
            json={"gating": {"max_no_speech_prob": before["gating"]["max_no_speech_prob"]}},
        )
        assert res.status_code == 200
        after = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert after["asr"]["model"] == before["asr"]["model"]
        assert after["gating"]["max_no_speech_prob"] == before["gating"]["max_no_speech_prob"]
    finally:
        path.write_text(original, encoding="utf-8")


# --------------------------------------------------------------------------- #
# regate — needs a real sidecar from a prior run
# --------------------------------------------------------------------------- #

def _any_sidecar() -> Path | None:
    if not WORK_DIR.exists():
        return None
    for path in WORK_DIR.glob("*/transcript.sgen.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("segments"):
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


SIDECAR = _any_sidecar()
needs_sidecar = pytest.mark.skipif(SIDECAR is None, reason="no transcript in work/ yet")


@needs_sidecar
def test_library_includes_a_known_transcript(client):
    """A transcript on disk must be reachable even with an empty job queue."""
    content_id = SIDECAR.parent.name
    ids = [i["content_id"] for i in client.get("/api/library").json()["items"]]
    assert content_id in ids


@needs_sidecar
def test_result_returns_segments_and_cues(client):
    content_id = SIDECAR.parent.name
    data = client.get(f"/api/result/{content_id}").json()
    assert data["segments"]
    assert data["cues"]
    assert data["language"]


@needs_sidecar
def test_regate_runs_without_gpu_and_returns_stats(client):
    content_id = SIDECAR.parent.name
    res = client.post(f"/api/result/{content_id}/regate", json={"gating": {}, "cues": {}})
    assert res.status_code == 200
    body = res.json()
    assert "kept" in body["stats"]
    assert body["stats"]["total"] > 0
    assert isinstance(body["cues"], list)


@needs_sidecar
def test_stricter_thresholds_suppress_more(client):
    """The point of the tuner: moving a threshold must visibly change the result."""
    content_id = SIDECAR.parent.name

    lenient = client.post(f"/api/result/{content_id}/regate", json={
        "gating": {"min_mean_word_prob": 0.0, "hard_avg_logprob": -3.0,
                   "hard_no_speech_prob": 1.0, "max_compression_ratio": 5.0},
    }).json()
    strict = client.post(f"/api/result/{content_id}/regate", json={
        "gating": {"min_mean_word_prob": 0.99},
    }).json()

    assert strict["stats"]["suppressed"] > lenient["stats"]["suppressed"]
    assert lenient["stats"]["suppressed"] == 0


@needs_sidecar
def test_narrower_line_limit_produces_more_cues(client):
    content_id = SIDECAR.parent.name
    wide = client.post(f"/api/result/{content_id}/regate",
                       json={"cues": {"max_chars_per_line": 60}}).json()
    narrow = client.post(f"/api/result/{content_id}/regate",
                         json={"cues": {"max_chars_per_line": 22}}).json()
    assert len(narrow["cues"]) >= len(wide["cues"])


@needs_sidecar
def test_save_writes_subtitles(client, tmp_path):
    content_id = SIDECAR.parent.name
    res = client.post(f"/api/result/{content_id}/save", json={
        "cues": [{"start": 1.0, "end": 3.0, "lines": ["Hand edited line."]}],
        "formats": ["srt"],
        "out_dir": str(tmp_path),
    })
    assert res.status_code == 200
    written = [Path(p) for p in res.json()["written"]]
    assert written and written[0].exists()
    body = written[0].read_text(encoding="utf-8-sig")
    assert "Hand edited line." in body


@needs_sidecar
def test_edits_are_remembered(client, tmp_path):
    """A saved edit must come back on reload, not be overwritten by the original."""
    content_id = SIDECAR.parent.name
    edits_file = WORK_DIR / content_id / "edits.json"
    backup = edits_file.read_text(encoding="utf-8") if edits_file.exists() else None
    try:
        client.post(f"/api/result/{content_id}/save", json={
            "cues": [{"start": 2.0, "end": 4.0, "lines": ["Remembered."]}],
            "formats": ["srt"],
            "out_dir": str(tmp_path),
        })
        data = client.get(f"/api/result/{content_id}").json()
        assert data["edited"]
        assert data["edited"][0]["lines"] == ["Remembered."]
    finally:
        if backup is None:
            edits_file.unlink(missing_ok=True)
        else:
            edits_file.write_text(backup, encoding="utf-8")
