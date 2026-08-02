"""Per-file settings, stored in the folder they apply to.

A folder of home video is rarely uniform: two of the files are songs and want the
`music` profile, the Hindi ones want Latin-script subtitles, and one is already
in English and wants no translation. One settings panel cannot express that.

The overrides live in `sgen.folder.yaml` **next to the videos**, for the same
reason resumability reads the subtitle files: state kept in the app cannot
survive a restart, and state kept in a database can disagree with the disk. A
file in the folder travels with the videos, survives anything, and can be edited
by hand — which for fifty files is faster than fifty clicks.

    # sgen.folder.yaml
    files:
      "Full Song - KHAIRIYAT.mp4":
        profile: music
        romanize: true
      "beach 2019.mp4":
        translate: none          # already English, nothing to translate

Anything a file does not mention comes from the app's own settings, so this file
stays as short as the exceptions actually are.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CONFIG_NAME = "sgen.folder.yaml"

# Only settings that plausibly differ per file. Model, batch size and output
# folder are properties of the machine or the run, not of one video.
ALLOWED = {
    "profile": str,          # home-video | music | verbatim
    "language": str,         # "" means detect
    "hotwords": str,
    "romanize": bool,
    "translate": str,        # none | deepl | google | local | "" (use settings)
    "translate_target": str,
}
TRANSLATE_CHOICES = {"", "none", "deepl", "google", "local"}

HEADER = """\
# Per-file settings for this folder, read by sgen.
#
# Anything not listed here uses the app's own settings. Kept beside the videos on
# purpose: it survives restarts, moves with the folder, and is quicker to edit by
# hand than to click through for a large batch.
#
# profile:          home-video | music | verbatim
# language:         auto detects; or a code like hi, de, ru
# hotwords:         names in this file, comma-separated
# romanize:         true also writes Latin-script subtitles (नमस्ते -> namaste)
# translate:        none | deepl | google | local
# translate_target: language code, default en
"""


class FolderConfigError(ValueError):
    """The folder file names something that is not a per-file setting."""


def config_path(folder: Path) -> Path:
    return folder / CONFIG_NAME


def load(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the overrides. A missing or unreadable file means "no overrides".

    Never raises: a batch of fifty videos must not fail because of a typo in an
    optional file. Problems are logged and surfaced by `problems()`.
    """
    path = config_path(folder)
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    files = raw.get("files") if isinstance(raw, dict) else None
    if not isinstance(files, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for name, values in files.items():
        if not isinstance(values, dict):
            continue
        clean = {}
        for key, value in values.items():
            if key not in ALLOWED:
                log.warning("%s: ignoring unknown per-file setting %r", path.name, key)
                continue
            if ALLOWED[key] is bool and not isinstance(value, bool):
                log.warning("%s: %s must be true or false", path.name, key)
                continue
            if ALLOWED[key] is str:
                value = "" if value is None else str(value).strip()
            clean[key] = value
        if clean:
            out[_normalise(str(name))] = clean
    return out


def problems(folder: Path) -> list[str]:
    """Human-readable complaints about the folder file, for the UI to show."""
    path = config_path(folder)
    if not path.exists():
        return []
    issues: list[str] = []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"{CONFIG_NAME} could not be read: {exc}"]
    if not isinstance(raw, dict) or not isinstance(raw.get("files", {}), dict):
        return [f"{CONFIG_NAME} should contain a 'files:' mapping"]
    for name, values in (raw.get("files") or {}).items():
        if not isinstance(values, dict):
            issues.append(f"{name}: expected a mapping of settings")
            continue
        for key, value in values.items():
            if key not in ALLOWED:
                issues.append(f"{name}: unknown setting {key!r}")
            elif key == "translate" and str(value).lower() not in TRANSLATE_CHOICES:
                issues.append(
                    f"{name}: translate must be one of "
                    f"{', '.join(sorted(c for c in TRANSLATE_CHOICES if c))}"
                )
    return issues


def _normalise(name: str) -> str:
    """Folder-relative, forward slashes, so a subfolder entry is unambiguous."""
    return name.replace("\\", "/").strip("/")


def key_for(folder: Path, source: Path) -> str:
    try:
        return _normalise(str(source.relative_to(folder)))
    except ValueError:
        return _normalise(source.name)


def for_file(folder: Path, source: Path) -> dict[str, Any]:
    """Overrides for one file, matched by relative path then by bare name."""
    overrides = load(folder)
    key = key_for(folder, source)
    if key in overrides:
        return overrides[key]
    return overrides.get(_normalise(source.name), {})


def apply_to_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge one file's overrides into the run's options.

    `translate` is the one that needs translating itself: the folder file says
    what to use in plain terms ("none", "deepl"), while the job options carry the
    two separate flags the pipeline actually reads.
    """
    options = dict(base)
    for key in ("profile", "language", "hotwords", "translate_target"):
        if key in override:
            options[key] = override[key]
    # An empty value cannot be stored — it is how "no override" is spelled — so
    # "auto" is what lets one file detect its language while the app has one
    # pinned for everything else.
    if str(options.get("language", "")).lower() in ("auto", "detect"):
        options["language"] = ""
    if "romanize" in override:
        options["romanize"] = override["romanize"]

    choice = str(override.get("translate", "")).lower()
    if choice in ("none", ""):
        if choice == "none":
            options["translate"] = False
            options["cloud_provider"] = ""
    elif choice == "local":
        options["translate"] = True
        options["cloud_provider"] = ""
    elif choice in ("deepl", "google"):
        options["translate"] = False
        options["cloud_provider"] = choice
    return options


def save(folder: Path, overrides: dict[str, dict[str, Any]]) -> Path:
    """Write the whole file. Empty overrides removes it rather than leaving a
    file that says nothing."""
    path = config_path(folder)
    cleaned = {name: values for name, values in overrides.items() if values}
    if not cleaned:
        path.unlink(missing_ok=True)
        return path
    body = HEADER + "\n" + yaml.safe_dump(
        {"files": cleaned}, allow_unicode=True, sort_keys=True, default_flow_style=False
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
    return path


def clear_all(folder: Path) -> tuple[Path, int]:
    """Remove every per-file override in this folder.

    Returns the path and how many files had settings, so the caller can say what
    it undid rather than reporting a silent success.
    """
    count = len(load(folder))
    path = config_path(folder)
    path.unlink(missing_ok=True)
    return path, count


def set_for_file(
    folder: Path, source: Path, values: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Set (or with an empty dict, clear) one file's overrides."""
    for key, value in values.items():
        if key not in ALLOWED:
            raise FolderConfigError(
                f"{key!r} is not a per-file setting "
                f"(allowed: {', '.join(sorted(ALLOWED))})"
            )
        if key == "translate" and str(value).lower() not in TRANSLATE_CHOICES:
            raise FolderConfigError(f"unknown translate choice {value!r}")

    overrides = load(folder)
    key = key_for(folder, source)
    # Drop a duplicate bare-name entry, so the two cannot disagree later.
    overrides.pop(_normalise(source.name), None)
    kept = {k: v for k, v in values.items() if v not in ("", None)}
    if kept:
        overrides[key] = kept
    else:
        overrides.pop(key, None)
    save(folder, overrides)
    return overrides
