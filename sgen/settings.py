"""User settings — one hand-editable file for keys and machine defaults.

`profiles/*.yaml` say *how to transcribe*. This file says how **this machine**
should behave: API keys for online translation, which profile the UI starts on,
where subtitles go, what port to bind. The two are deliberately separate,
because profiles are shared and committed while this holds secrets and local
paths that must never be.

Everything is optional. Every property has a default, so a missing file or a
file with three lines in it is normal, not an error. The file is re-read on
every access rather than cached, so editing it while the UI is running takes
effect on the next request — no restart, which matters most for the property
most likely to be wrong on the first attempt: an API key.

Precedence, highest first:

1. an explicit command-line flag or UI control
2. environment variables (``SGEN_GOOGLE_API_KEY``, ``SGEN_DEEPL_API_KEY``) —
   for keys you would rather not write to disk at all
3. ``settings.local.yaml``
4. the defaults in this module

Writes preserve the file. Saving a key from the UI edits the one line it owns
and leaves your comments, ordering and other settings untouched, because a
hand-maintained file that a program rewrites wholesale stops being worth
maintaining by hand.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT

log = logging.getLogger(__name__)

SETTINGS_NAME = "settings.local.yaml"
TEMPLATE_PATH = REPO_ROOT / "settings.example.yaml"

# Keys may come from the environment instead of the file.
ENV_KEYS = {"google": "SGEN_GOOGLE_API_KEY", "deepl": "SGEN_DEEPL_API_KEY"}


class SettingsError(ValueError):
    """The settings file is malformed, or names something that doesn't exist.

    Raised rather than ignored: a typo that is silently dropped looks exactly
    like a setting that doesn't work.
    """


def settings_path() -> Path:
    """Where settings live. ``SGEN_SETTINGS`` overrides, mainly for tests."""
    override = os.environ.get("SGEN_SETTINGS")
    return Path(override).expanduser() if override else REPO_ROOT / SETTINGS_NAME


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

@dataclass
class ApiKeys:
    """Credentials for the optional online translators."""

    google: str = ""
    deepl: str = ""
    # Free and Pro are different hostnames. Free keys end in ":fx" and are
    # detected from the key itself, so this only matters for Pro.
    deepl_plan: str = "free"


@dataclass
class TranslateDefaults:
    provider: str = "google"      # google | deepl | local
    target: str = "en"
    # Translate every file whose language is not `target`, without being asked
    # each time. Files already in the target language are left alone — there is
    # nothing to translate and it would spend quota to say so.
    auto: bool = False


@dataclass
class Defaults:
    """What the UI and CLI start from when nothing is specified."""

    profile: str = "home-video"
    language: str = ""            # "" => detect per file
    hotwords: str = ""            # names and places in your footage
    romanize: bool = False
    keep_suppressed: bool = False
    # SRT only. WebVTT matters for browser players; for watching in VLC/MPC and
    # editing in Subtitle Edit, the second file is just clutter next to every
    # video. Add "vtt" here to get both.
    formats: tuple[str, ...] = ("srt",)
    out_dir: str = ""             # "" => next to each source file
    translate: TranslateDefaults = field(default_factory=TranslateDefaults)


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"       # localhost only; do not expose this
    port: int = 8420
    open_browser: bool = True
    # Stop servers left over from a previous launch instead of adding to them.
    # Without this they pile up: the old one keeps the port, the new one lands
    # elsewhere, and stale API handlers get served alongside current assets.
    replace_running: bool = True


@dataclass
class Settings:
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    defaults: Defaults = field(default_factory=Defaults)
    server: ServerSettings = field(default_factory=ServerSettings)

    # Provenance, for `sgen config` and the UI — not settings themselves.
    path: Path = field(default_factory=settings_path)
    exists: bool = False
    error: str = ""
    key_source: dict[str, str] = field(default_factory=dict)
    # Dotted names the file actually mentions. Needed to tell "the user asked
    # for [srt]" from "nobody said anything", which decides whether a setting
    # overrides the profile or defers to it.
    provided: set[str] = field(default_factory=set)

    def given(self, dotted: str) -> bool:
        return dotted in self.provided

    def to_dict(self) -> dict[str, Any]:
        """Settings only, with keys redacted. Safe to send to the browser."""
        return {
            "api_keys": {
                "google": _redact(self.api_keys.google),
                "deepl": _redact(self.api_keys.deepl),
                "deepl_plan": self.api_keys.deepl_plan,
            },
            "defaults": dataclasses.asdict(self.defaults) | {
                "formats": list(self.defaults.formats)
            },
            "server": dataclasses.asdict(self.server),
        }


SECTIONS = ("api_keys", "defaults", "server")


def _redact(value: str) -> str:
    """Never hand a full key back out — the UI only needs to know it is set."""
    if not value:
        return ""
    return f"…{value[-4:]}" if len(value) > 4 else "…"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def load(path: Path | None = None) -> Settings:
    """Read settings, raising SettingsError on anything malformed."""
    path = path or settings_path()
    result = Settings(path=path, exists=path.exists())

    data: dict[str, Any] = {}
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SettingsError(f"{path.name} is not valid YAML: {exc}") from exc
        except OSError as exc:
            raise SettingsError(f"could not read {path}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise SettingsError(f"{path.name} must contain a mapping of sections")
        data = raw

    for section in data:
        if section not in SECTIONS:
            raise SettingsError(
                f"{path.name}: unknown section {section!r} "
                f"(expected {', '.join(SECTIONS)})"
            )
    for section in SECTIONS:
        if section in data:
            _apply(
                getattr(result, section), data[section],
                where=f"{path.name}: {section}", prefix=section,
                provided=result.provided,
            )

    _apply_environment(result)
    return result


def load_or_default(path: Path | None = None) -> Settings:
    """Never raises. A broken file must not stop a transcription from running.

    The problem is reported on ``.error`` so the caller can surface it — the CLI
    prints it, the UI shows it — rather than failing silently.
    """
    try:
        return load(path)
    except SettingsError as exc:
        log.warning("%s", exc)
        broken = Settings(path=path or settings_path(), exists=True, error=str(exc))
        _apply_environment(broken)
        return broken


def _apply(target: Any, data: Any, where: str, prefix: str, provided: set[str]) -> None:
    if not isinstance(data, dict):
        raise SettingsError(f"{where} must be a mapping, not {type(data).__name__}")
    valid = {f.name for f in dataclasses.fields(target)}
    for key, value in data.items():
        if key not in valid:
            raise SettingsError(
                f"{where}: unknown option {key!r} (valid: {', '.join(sorted(valid))})"
            )
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value, f"{where}.{key}", f"{prefix}.{key}", provided)
        else:
            setattr(target, key, _coerce(current, value, f"{where}.{key}"))
            provided.add(f"{prefix}.{key}")


def _coerce(current: Any, value: Any, where: str) -> Any:
    """Keep the declared type. YAML is loose; the rest of the code is not."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise SettingsError(f"{where} must be true or false, got {value!r}")
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{where} must be a whole number, got {value!r}")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            raise SettingsError(f"{where} must be a list, got {value!r}")
        return tuple(str(v) for v in value)
    # str
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise SettingsError(f"{where} must be text, got {value!r}")
    return str(value).strip()


def apply_defaults(cfg: Any, user: Settings | None = None) -> Any:
    """Overlay the settings file onto a Config already loaded from a profile.

    Only properties the file actually mentions are applied. A profile that sets
    `formats: [vtt]` therefore keeps it unless the settings file names formats
    too — settings fill gaps and state machine-wide preferences; they do not
    silently reach into every profile.
    """
    user = user or load_or_default()
    d = user.defaults
    if user.given("defaults.language"):
        cfg.asr.language = d.language or None          # "" => detect per file
    if user.given("defaults.hotwords"):
        cfg.asr.hotwords = d.hotwords or None
    if user.given("defaults.romanize"):
        cfg.romanize = d.romanize
    if user.given("defaults.keep_suppressed"):
        cfg.gating.drop_suppressed = not d.keep_suppressed
    if user.given("defaults.formats") and d.formats:
        cfg.formats = tuple(d.formats)
    if user.given("defaults.translate.target"):
        cfg.translate_target = d.translate.target
    return cfg


def _apply_environment(result: Settings) -> None:
    for name, variable in ENV_KEYS.items():
        from_env = (os.environ.get(variable) or "").strip()
        if from_env:
            setattr(result.api_keys, name, from_env)
            result.key_source[name] = variable
            result.provided.add(f"api_keys.{name}")
        elif getattr(result.api_keys, name):
            result.key_source[name] = str(result.path)
        else:
            result.key_source[name] = "unset"


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

TEMPLATE_FALLBACK = """\
# sgen settings. Gitignored: safe for API keys.
# Every property is optional; delete a line to fall back to the default.

api_keys:
  google: ""
  deepl: ""
  deepl_plan: free

defaults:
  profile: home-video
  language: ""
  hotwords: ""
  romanize: false
  keep_suppressed: false
  formats: [srt]
  out_dir: ""
  translate:
    provider: google
    target: en

server:
  host: 127.0.0.1
  port: 8420
  open_browser: true
  replace_running: true
"""


def _write(path: Path, body: str) -> None:
    """Write with LF endings, matching profiles/*.yaml and this file's template.

    `write_text` would translate to CRLF on Windows, so a file created here and
    a file created by an editor would differ in every line.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)


def ensure_file(path: Path | None = None) -> tuple[Path, bool]:
    """Create the settings file from the template if it is missing.

    Returns the path and whether it was created, so callers can say which.
    """
    path = path or settings_path()
    if path.exists():
        return path, False
    body = (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        if TEMPLATE_PATH.exists()
        else TEMPLATE_FALLBACK
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, body)
    return path, True


def set_values(values: dict[str, Any], path: Path | None = None) -> Path:
    """Set dotted properties (``api_keys.google``) in place.

    Validated against the schema *before* the file is touched, so a typo is
    rejected instead of written. Comments, ordering and unrelated settings
    survive: the line for each property is rewritten and nothing else is.
    """
    path = path or settings_path()
    for dotted, value in values.items():
        _validate_path(dotted, value)

    if not path.exists():
        ensure_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for dotted, value in values.items():
        lines = _edit(lines, dotted.split("."), _render(value))
    _write(path, "\n".join(lines) + "\n")
    return path


def update_api_keys(
    google: str | None = None,
    deepl: str | None = None,
    deepl_plan: str | None = None,
    path: Path | None = None,
) -> Path:
    """Store keys. ``None`` leaves a key as it is; ``""`` clears it."""
    changes: dict[str, Any] = {}
    if google is not None:
        changes["api_keys.google"] = google
    if deepl is not None:
        changes["api_keys.deepl"] = deepl
    if deepl_plan is not None:
        changes["api_keys.deepl_plan"] = deepl_plan
    if not changes:
        return path or settings_path()
    return set_values(changes, path)


def _validate_path(dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target: Any = Settings()
    for depth, part in enumerate(parts):
        if not dataclasses.is_dataclass(target):
            raise SettingsError(f"{'.'.join(parts[:depth])} has no sub-properties")
        names = {f.name for f in dataclasses.fields(target)}
        if depth == 0:
            names &= set(SECTIONS)
        if part not in names:
            raise SettingsError(
                f"unknown property {dotted!r} — {part!r} is not one of "
                f"{', '.join(sorted(names))}"
            )
        target = getattr(target, part)
    if dataclasses.is_dataclass(target):
        raise SettingsError(f"{dotted!r} is a section, not a single property")
    _coerce(target, value, dotted)


def parse_assignment(text: str) -> tuple[str, Any]:
    """Parse ``defaults.profile=music`` from the command line.

    The value goes through the YAML scalar parser, so ``true``, ``8421`` and
    ``[srt]`` arrive as the types the schema expects.
    """
    if "=" not in text:
        raise SettingsError(f"expected property=value, got {text!r}")
    dotted, _, raw = text.partition("=")
    dotted, raw = dotted.strip(), raw.strip()
    try:
        value = yaml.safe_load(raw) if raw else ""
    except yaml.YAMLError:
        value = raw
    if value is None:
        value = ""
    _validate_path(dotted, value)
    return dotted, value


_PLAIN = re.compile(r"^[A-Za-z][A-Za-z0-9 _./+-]*$")
_YAML_KEYWORDS = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}


def _render(value: Any) -> str:
    """Render a scalar for a file a person also edits: quote only when needed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    text = str(value)
    if _PLAIN.match(text) and text.lower() not in _YAML_KEYWORDS:
        return text
    # Anything else gets quoted — DeepL free keys end in ":fx", which YAML would
    # otherwise read as a nested mapping.
    return json.dumps(text)


def _trailing_comment(line: str) -> str:
    """Keep a note the user wrote beside a value when rewriting that value."""
    at = line.find(" #")
    if at < 0:
        return ""
    if line[:at].count('"') % 2:      # inside a quoted string, not a comment
        return ""
    return "  " + line[at:].strip()


def _key_line(indent: str, key: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(indent)}{re.escape(key)}\s*:")


def _block_end(lines: list[str], start: int, indent: str) -> int:
    """Index just past the block owned by a header at `indent`."""
    end = start
    for i in range(start, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= len(indent):
            break
        end = i + 1
    return end


def _child_indent(lines: list[str], start: int, end: int, indent: str) -> str:
    for i in range(start, end):
        if lines[i].strip():
            return lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return indent + "  "


def _edit(lines: list[str], parts: list[str], rendered: str) -> list[str]:
    """Set one property, adding whatever part of the chain is missing."""
    indent, start, end = "", 0, len(lines)
    for depth, part in enumerate(parts):
        pattern = _key_line(indent, part)
        found = next((i for i in range(start, end) if pattern.match(lines[i])), None)
        if found is None:
            # Append the missing chain at the end of the enclosing block, which
            # `end` already points just past.
            block = _chain(parts[depth:], rendered, indent)
            return lines[:end] + block + lines[end:]
        if depth == len(parts) - 1:
            lines[found] = f"{indent}{part}: {rendered}{_trailing_comment(lines[found])}"
            return lines
        start, end = found + 1, _block_end(lines, found + 1, indent)
        indent = _child_indent(lines, start, end, indent)
    return lines


def _chain(parts: list[str], rendered: str, indent: str) -> list[str]:
    out = []
    for depth, part in enumerate(parts[:-1]):
        out.append(f"{indent}{'  ' * depth}{part}:")
    out.append(f"{indent}{'  ' * (len(parts) - 1)}{parts[-1]}: {rendered}")
    return out
