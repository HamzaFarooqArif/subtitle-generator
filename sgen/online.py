"""Online translation providers.

Local models hit a quality ceiling well below a production service, so this adds
Google Cloud Translation and DeepL as opt-in backends. They expose the same
`translate_texts` interface as the local NLLB translator, so the surrounding
machinery — sentence grouping, timing, line breaking — is shared and only the
translation step differs.

**These send transcript text to a third party.** Nothing here runs unless the
user explicitly asks for it: no key is read during a normal transcription, and
the pipeline never calls this on its own. Keys live in a gitignored file next to
the project, never in the sidecar or any subtitle output.

Only the standard library is used for HTTP, so no new dependency is introduced
for a feature most runs will not touch.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import settings

log = logging.getLogger(__name__)

GOOGLE_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
DEEPL_FREE = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO = "https://api.deepl.com/v2/translate"

DEEPL_LANGUAGES_FREE = "https://api-free.deepl.com/v2/languages"
DEEPL_LANGUAGES_PRO = "https://api.deepl.com/v2/languages"

# Which languages DeepL accepts is asked of DeepL rather than remembered here.
# The list below was written from its documentation and went stale: it claims no
# Hindi and no Urdu, when the service now reports 110 target languages including
# both. Refusing a pair the service supports is worse than sending it, so this is
# only the fallback for when the live list cannot be fetched — no key configured,
# or the lookup failed.
DEEPL_FALLBACK_TARGETS = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "id",
    "it", "ja", "ko", "lt", "lv", "nb", "no", "nl", "pl", "pt", "ro", "ru",
    "sk", "sl", "sv", "tr", "uk", "zh",
}
# Kept for callers that only need "probably supported" without a key.
DEEPL_TARGETS = DEEPL_FALLBACK_TARGETS

# Codes DeepL spells differently from Whisper. It has no plain "en" or "pt" —
# only regional variants — so those must be expanded or the request 400s.
_DEEPL_TARGET_ALIASES = {"en": "EN-US", "pt": "PT-PT", "no": "NB"}

# Live target list, fetched once per process and shared: it changes on DeepL's
# release schedule, not during a run.
_deepl_target_cache: dict[str, frozenset[str]] = {}


class TranslationError(RuntimeError):
    """A provider refused or failed in a way the user needs to see."""


# --------------------------------------------------------------------------- #
# key storage
# --------------------------------------------------------------------------- #

#
# Keys live in settings.local.yaml (see sgen/settings.py) or in the environment.
# This module keeps its own small view of them so nothing else needs to care
# where they came from.

@dataclass
class Keys:
    google: str = ""
    deepl: str = ""
    deepl_plan: str = "free"          # "free" or "pro" — different hostnames

    def to_dict(self) -> dict[str, Any]:
        return {"google": self.google, "deepl": self.deepl,
                "deepl_plan": self.deepl_plan}


def keys_path() -> Path:
    return settings.settings_path()


def load_keys() -> Keys:
    api = settings.load_or_default().api_keys
    return Keys(google=api.google, deepl=api.deepl,
                deepl_plan=api.deepl_plan or "free")


def save_keys(keys: Keys) -> Path:
    """Write the keys back, leaving the rest of the settings file alone."""
    return settings.update_api_keys(
        google=keys.google, deepl=keys.deepl, deepl_plan=keys.deepl_plan
    )


def configured() -> dict[str, bool]:
    keys = load_keys()
    return {"google": bool(keys.google), "deepl": bool(keys.deepl)}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _post(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
    attempts: int = 4,
) -> dict[str, Any]:
    """POST with backoff on rate limits and transient server errors."""
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                delay = 1.5 * (2 ** attempt)
                log.warning("provider returned %s; retrying in %.1fs", exc.code, delay)
                time.sleep(delay)
                last = exc
                continue
            raise TranslationError(_explain(exc.code, body)) from exc
        except urllib.error.URLError as exc:
            if attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                last = exc
                continue
            raise TranslationError(f"could not reach the service: {exc.reason}") from exc
    raise TranslationError(f"translation failed after {attempts} attempts: {last}")


def _explain(code: int, body: str) -> str:
    if code in (401, 403):
        return (
            f"the service rejected the key ({code}). Check the key is correct and, "
            "for Google, that the Cloud Translation API is enabled and billing is "
            f"active on the project. Response: {body}"
        )
    if code == 429:
        return f"rate limited ({code}). Wait and retry, or translate fewer files at once."
    if code == 456:
        return "DeepL quota exhausted for this billing period (456)."
    if code == 400:
        return f"the service rejected the request (400) — often an unsupported language. {body}"
    return f"the service returned {code}: {body}"


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #

class GoogleTranslator:
    """Google Cloud Translation v2.

    Batched: the API takes repeated `q` parameters and returns one translation
    per input in the same order, so cue alignment is exact and needs no
    line-number matching.
    """

    name = "google"
    max_batch = 100

    def __init__(self, api_key: str):
        if not api_key:
            raise TranslationError("no Google Translate API key configured")
        self._key = api_key

    def translate_texts(
        self, texts: Sequence[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        out: list[str] = []
        for start in range(0, len(texts), self.max_batch):
            chunk = list(texts[start : start + self.max_batch])
            fields: list[tuple[str, str]] = [
                ("target", target_language), ("format", "text"), ("key", self._key)
            ]
            if source_language:
                fields.append(("source", source_language))
            fields.extend(("q", t) for t in chunk)
            body = urllib.parse.urlencode(fields).encode("utf-8")
            payload = _post(
                GOOGLE_ENDPOINT,
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            try:
                translations = payload["data"]["translations"]
            except (KeyError, TypeError) as exc:
                raise TranslationError(f"unexpected Google response: {payload}") from exc
            if len(translations) != len(chunk):
                raise TranslationError(
                    f"Google returned {len(translations)} translations for {len(chunk)} inputs"
                )
            out.extend(_unescape(t.get("translatedText", "")) for t in translations)
        return out


class DeepLTranslator:
    """DeepL v2. Same contract: one translation per input, order preserved.

    Two things matter for subtitles specifically:

    **Context.** Each cue is translated as its own item, which is what keeps the
    result aligned to the timings — but an isolated cue has no conversation
    around it, and that is exactly when a translator guesses wrong. Russian
    "мусора" alone is "garbage"; in a scene where someone is hiding, it is
    "cops". DeepL's `context` field takes surrounding text that informs the
    translation without being translated or billed, so every request carries the
    neighbouring cues.

    **Model.** `prefer_quality_optimized` asks for DeepL's next-generation
    model, which is markedly better on idiom and register, and falls back on its
    own for language pairs that do not have it.
    """

    name = "deepl"
    # Smaller than the API allows: context is per request, so batches have to be
    # small enough for "the text around this" to still mean something.
    max_batch = 25
    # Cues of surrounding conversation sent as context on each side.
    context_window = 10
    # Context is free but still travels in the request body; keep it bounded.
    max_context_chars = 2000

    def __init__(self, api_key: str, plan: str = "free"):
        if not api_key:
            raise TranslationError("no DeepL API key configured")
        self._key = api_key
        # Free and Pro live on different hostnames, and using the wrong one
        # returns a bare 403 that looks like a bad key. DeepL suffixes free keys
        # with ":fx", so the right host can be chosen without asking.
        if api_key.endswith(":fx"):
            self._url = DEEPL_FREE
        elif plan == "pro":
            self._url = DEEPL_PRO
        else:
            self._url = DEEPL_FREE

    @staticmethod
    def supports(target_language: str) -> bool:
        """Fallback check, for callers with no key to ask with."""
        return (target_language or "").lower() in DEEPL_FALLBACK_TARGETS

    def targets(self) -> frozenset[str]:
        """What this account can translate into, according to DeepL.

        Cached per key for the life of the process. On any failure the stale
        built-in list is returned rather than blocking the translation — an
        unsupported pair will be refused by the service anyway, with a clearer
        message than a guess made here.
        """
        cached = _deepl_target_cache.get(self._key)
        if cached is not None:
            return cached

        url = DEEPL_LANGUAGES_PRO if self._url == DEEPL_PRO else DEEPL_LANGUAGES_FREE
        try:
            request = urllib.request.Request(
                f"{url}?type=target",
                headers={"Authorization": f"DeepL-Auth-Key {self._key}"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            reported = {str(item["language"]).lower() for item in payload}
        except Exception as exc:            # network, auth, shape — all non-fatal
            log.warning("could not read DeepL's language list (%s); using the "
                        "built-in one, which may be out of date", exc)
            return frozenset(DEEPL_FALLBACK_TARGETS)

        # "en-gb" also means English is available; keep both forms so a plain
        # Whisper code matches.
        codes = reported | {code.split("-")[0] for code in reported}
        result = frozenset(codes)
        _deepl_target_cache[self._key] = result
        log.info("DeepL reports %d target languages", len(reported))
        return result

    def can_target(self, target_language: str) -> bool:
        return (target_language or "").lower() in self.targets()

    @classmethod
    def _context(cls, texts: Sequence[str], start: int, end: int) -> str:
        """The conversation around this batch, for DeepL's `context` field.

        Not translated and not billed — it only tells the model what scene these
        lines belong to. Every item in a request is still translated on its own,
        which is what keeps cues aligned to their timings, so without this a cue
        is a sentence with no world around it.

        A file shorter than one batch has no "around" to send, and that is
        exactly the case where its own lines are the missing context, so they are
        used instead.
        """
        window = cls.context_window
        before = list(texts[max(0, start - window) : start])
        after = list(texts[end : end + window])
        surrounding = [t.strip() for t in (*before, *after) if t.strip()]
        if not surrounding:
            surrounding = [t.strip() for t in texts if t.strip()]
        context = " ".join(surrounding)
        return context[:cls.max_context_chars]

    def translate_texts(
        self, texts: Sequence[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        target = (target_language or "en").lower()
        if not self.can_target(target):
            raise TranslationError(
                f"DeepL does not translate into {target_language!r} — "
                "use Google for this language."
            )
        deepl_target = _DEEPL_TARGET_ALIASES.get(target, target.upper())

        out: list[str] = []
        for start in range(0, len(texts), self.max_batch):
            end = start + self.max_batch
            chunk = list(texts[start:end])
            fields: list[tuple[str, str]] = [
                ("target_lang", deepl_target),
                # Falls back by itself where the next-gen model has no pair.
                ("model_type", "prefer_quality_optimized"),
            ]
            source = (source_language or "").lower()
            # Only send source_lang when DeepL knows it; otherwise let it detect,
            # which it does well. Source languages are a subset of targets.
            if source and source in self.targets():
                fields.append(("source_lang", source.upper()))
            context = self._context(texts, start, end)
            if context:
                fields.append(("context", context))
            fields.extend(("text", t) for t in chunk)
            body = urllib.parse.urlencode(fields).encode("utf-8")
            payload = _post(
                self._url,
                data=body,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self._key}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            translations = payload.get("translations")
            if not isinstance(translations, list) or len(translations) != len(chunk):
                raise TranslationError(f"unexpected DeepL response: {payload}")
            out.extend(t.get("text", "") for t in translations)
        return out


def _unescape(text: str) -> str:
    """Google returns HTML entities even with format=text."""
    import html

    return html.unescape(text)


def get_translator(provider: str, keys: Keys | None = None):
    """Build a provider by name, raising a clear error if it isn't configured."""
    keys = keys or load_keys()
    if provider == "google":
        return GoogleTranslator(keys.google)
    if provider == "deepl":
        return DeepLTranslator(keys.deepl, keys.deepl_plan)
    raise TranslationError(f"unknown translation provider {provider!r}")
