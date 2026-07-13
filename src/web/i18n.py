"""Internationalisation for the web frontend.

The frontend is bilingual (German / English). Following the usual server-side
i18n split, this module owns three things and nothing else:

  * **Locale negotiation** — turning the request's `lang` query parameter into a
    supported `Locale`, defaulting to German when it's missing or unknown.
  * **Message catalogs** — the actual UI strings, kept out of the code and
    templates entirely and loaded from `web/locales/<locale>.json`. Adding a
    language is a new JSON file, not a code change.
  * **A request-scoped `Translator`** — a small object bound to one locale that
    looks strings up by key. It is injected into route handlers as a FastAPI
    dependency (`get_translator`) and passed into every template as `_`, so a
    template says `{{ _("home.title") }}` and never contains a literal string.

Keys are dotted (`"person.back"`, `"kin.uncle.female"`); a missing key falls
back to the default locale and then to the key itself, so a gap degrades to
something visible rather than crashing a page. `tests/test_i18n.py` asserts the
two catalogs share an identical key set, which is what actually guarantees no
key is missing at runtime.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from fastapi import Query

_LOCALES_DIR = Path(__file__).parent / "locales"


class Locale(StrEnum):
    DE = "de"
    EN = "en"


DEFAULT_LOCALE = Locale.DE


def negotiate(raw: str | None) -> Locale:
    """Resolve a requested locale code (the `lang` query param) to a supported
    Locale, falling back to the default for a missing or unknown value."""
    try:
        return Locale(raw) if raw else DEFAULT_LOCALE
    except ValueError:
        return DEFAULT_LOCALE


@lru_cache(maxsize=None)
def _catalog(locale: Locale) -> dict[str, str]:
    """Load and cache one locale's message catalog. Catalogs are static for a
    process, so a single read at first use is enough."""
    path = _LOCALES_DIR / f"{locale.value}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class Translator:
    """Looks up UI strings for one locale.

    `_("key")` returns the string; `_("key", name="X")` fills `{name}`
    placeholders; `_.plural("photos", n)` selects the `photos.one` / `photos.other`
    form by count. Missing keys fall back to the default locale, then to the key
    itself.
    """

    def __init__(self, locale: Locale) -> None:
        self._locale = locale
        self._catalog = _catalog(locale)
        self._fallback = _catalog(DEFAULT_LOCALE)

    @property
    def locale(self) -> Locale:
        return self._locale

    def gettext(self, key: str, /, **params: object) -> str:
        text = self._catalog.get(key) or self._fallback.get(key) or key
        return text.format(**params) if params else text

    # Templates call the translator directly: {{ _("home.title") }}.
    __call__ = gettext

    def plural(self, key: str, n: int, /, **params: object) -> str:
        """Select `<key>.one` for n == 1, else `<key>.other`; passes `n` through
        as a placeholder so the string can embed the count."""
        form = "one" if n == 1 else "other"
        return self.gettext(f"{key}.{form}", n=n, **params)


@lru_cache(maxsize=None)
def translator(locale: Locale) -> Translator:
    """The (cached) Translator for a locale — one instance is reused per locale."""
    return Translator(locale)


def get_translator(lang: str | None = Query(default=None)) -> Translator:
    """FastAPI dependency: the Translator for this request's `?lang=` param."""
    return translator(negotiate(lang))
