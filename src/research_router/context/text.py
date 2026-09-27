"""Lightweight, dependency-free text helpers shared by the context modules."""

from __future__ import annotations

import html
import re

STOPWORDS = frozenset(
    """
    a an the and or but of to in on at by for with from about into over under
    is are was were be been being do does did have has had it its this that these
    those them they their there here what which who whom whose when where why how
    i me my mine we our you your he she his her can could should would will shall
    may might must not no yes than then so as if also just only very more most
    find search show get give list tell look please some any all each other
    """.split()  # noqa: SIM905
)

# Words that describe the *kind* of result wanted rather than the topic.
GENERIC_TERMS = frozenset(
    """
    recent latest new newest current today week month year papers paper research
    researches study studies article articles result results approach approaches
    method methods technique techniques best top good information info about
    implementation implementations code github repo repos repository repositories
    source sources news
    """.split()  # noqa: SIM905
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_LEADING_DATE_RE = re.compile(
    r"^(?:\d{1,2}\s+\w{3,9}\s+\d{4}|\w{3,9}\s+\d{1,2},\s+\d{4}|\d+\s+(?:hours?|days?|weeks?)\s+ago)"
    r"\s*[—\-–·:]\s*",
    re.I,
)
_BOILERPLATE_RE = re.compile(
    r"\b(?:read more|click here|learn more|sign in|sign up|log in|subscribe( now)?|"
    r"accept (all )?cookies|cookie policy|privacy policy|all rights reserved|"
    r"skip to (main )?content|share this|advertisement)\b[.!:]*",
    re.I,
)


def clean_text(text: str | None) -> str:
    """Strip HTML, boilerplate phrases, ellipses and redundant whitespace."""
    if not text:
        return ""
    out = html.unescape(_TAG_RE.sub(" ", text))
    out = _BOILERPLATE_RE.sub(" ", out)
    out = out.replace("…", " ").replace("...", " ")
    out = _WS_RE.sub(" ", out).strip(" -—|·")
    return out.strip()


def strip_leading_date(text: str) -> str:
    """Remove a leading ``"Jan 5, 2024 — "`` style date prefix from a snippet."""
    return _LEADING_DATE_RE.sub("", text, count=1)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return [w.lower().strip(".-") for w in _WORD_RE.findall(text or "") if w.strip(".-")]


def salient_terms(text: str, *, drop_generic: bool = False) -> list[str]:
    """Content-bearing terms of *text* in order, without duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for w in tokenize(text):
        if len(w) < 2 or w in STOPWORDS or w in seen:
            continue
        if drop_generic and w in GENERIC_TERMS:
            continue
        seen.add(w)
        out.append(w)
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + " …"


def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    words = tokenize(text)
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
