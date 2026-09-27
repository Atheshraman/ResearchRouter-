"""Obsidian vault as persistent research memory.

Notes are plain Markdown under ``<vault>/<research_folder>/<Topic>/<Subtopic>.md``
so they are browsable, linkable and editable in Obsidian.  No Obsidian plugin
or running app is required — the vault is just a directory.

Retrieval scans only the research folder (bounded by file count and bytes
per file), scores notes against the query, and returns the few best
*passages* — never whole notes and never the whole vault.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from research_router.context.evidence import Evidence, EvidenceSource, SourceType
from research_router.context.text import salient_terms, tokenize, truncate_words
from research_router.memory.base import MemoryHit, MemoryPassage, MemoryStore, ResearchRecord
from research_router.utils.dates import now
from research_router.utils.logging import get_logger

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
_FIELD_RE = re.compile(r"^(created|updated):\s*(.+)$", re.M)
_URL_RE = re.compile(r"\((https?://[^)\s]+)\)|(https?://\S+)")
_DATE_IN_LINE_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})[^)]*\)")
_MD_LINK_RE = re.compile(r"\[\[[^\]]*\]\]|\[[^\]]*\]\([^)]*\)")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9 _\-]+")
_MIN_SCORE = 0.34


class ObsidianMemory(MemoryStore):
    name = "obsidian"

    def __init__(
        self,
        vault_path: str,
        research_folder: str = "Research",
        max_files: int = 2000,
        max_bytes_per_file: int = 64_000,
    ) -> None:
        self._vault = Path(vault_path).expanduser() if vault_path else None
        self._folder = _safe_segment(research_folder) or "Research"
        self._max_files = max_files
        self._max_bytes = max_bytes_per_file

    # ── status ────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._vault is not None and self._vault.is_dir()

    @property
    def research_dir(self) -> Path | None:
        return self._vault / self._folder if self._vault else None

    def status(self) -> dict[str, object]:
        reason = None
        if self._vault is None:
            reason = "vault not configured"
        elif not self._vault.is_dir():
            reason = "vault path does not exist"
        return {"backend": self.name, "available": self.available, "reason": reason}

    # ── search ────────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 3) -> list[MemoryHit]:
        if not self.available:
            return []
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[MemoryHit]:
        root = self.research_dir
        assert root is not None and self._vault is not None
        if not root.is_dir():
            return []
        terms = salient_terms(query, drop_generic=True) or salient_terms(query)
        if not terms:
            return []

        files = sorted(root.rglob("*.md"), key=_mtime, reverse=True)[: self._max_files]
        hits: list[MemoryHit] = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read(self._max_bytes)
            except OSError:
                continue
            hit = self._score_note(path, text, terms)
            if hit is not None:
                hits.append(hit)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _score_note(self, path: Path, text: str, terms: list[str]) -> MemoryHit | None:
        assert self._vault is not None
        rel = path.relative_to(self._vault).with_suffix("").as_posix()
        fm_match = _FRONTMATTER_RE.match(text)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = text[fm_match.end() :] if fm_match else text

        path_tokens = set(tokenize(rel.replace("/", " ")))
        head_tokens = set(tokenize(frontmatter + " " + _headings(body)))
        body_tokens = set(tokenize(body))
        term_set = set(terms)
        score = (
            0.45 * len(term_set & path_tokens)
            + 0.25 * len(term_set & head_tokens)
            + 0.30 * len(term_set & body_tokens)
        ) / len(term_set)
        if score < _MIN_SCORE:
            return None

        recorded_at = None
        for key, value in _FIELD_RE.findall(frontmatter):
            if key == "updated" or recorded_at is None:
                recorded_at = value.strip().strip('"')
        if recorded_at is None:
            recorded_at = datetime.fromtimestamp(_mtime(path), UTC).isoformat(timespec="seconds")

        return MemoryHit(
            note=rel,
            title=path.stem,
            score=round(min(score, 1.0), 3),
            recorded_at=recorded_at,
            passages=_best_passages(body, term_set),
        )

    # ── save ──────────────────────────────────────────────────────

    async def save(self, record: ResearchRecord) -> str | None:
        if not self.available or not record.findings:
            return None
        return await asyncio.to_thread(self._save_sync, record)

    def _save_sync(self, record: ResearchRecord) -> str:
        root = self.research_dir
        assert root is not None and self._vault is not None
        topic, subtopic = note_location(record.query)
        path = (root / topic / f"{subtopic}.md").resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Refusing to write outside the research folder")
        path.parent.mkdir(parents=True, exist_ok=True)

        stamp = now().isoformat(timespec="seconds")
        section = _render_session(record, stamp)
        tags = _tags(record)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            content = _update_frontmatter(existing, stamp, record.query, tags) + "\n" + section
        else:
            content = _render_new_note(topic, subtopic, record, stamp, tags) + section
        _atomic_write(path, content)
        rel = path.relative_to(self._vault.resolve()).with_suffix("").as_posix()
        logger.info("Saved research note", extra={"extra_data": {"note": rel}})
        return rel


def hits_to_evidence(hits: list[MemoryHit]) -> list[Evidence]:
    """Convert memory hits into source-attributed ``MEMORY`` evidence."""
    out: list[Evidence] = []
    for hit in hits:
        for p in hit.passages:
            out.append(
                Evidence(
                    claim=p.text,
                    source=EvidenceSource(title=hit.title, url=p.url, date=p.date),
                    relevance_score=hit.score,
                    source_type=SourceType.MEMORY,
                    task_id="memory",
                    recorded_at=hit.recorded_at,
                    metadata={"note": hit.note},
                )
            )
    return out


# ── note layout helpers ───────────────────────────────────────────────


def note_location(query: str) -> tuple[str, str]:
    """Derive ``(Topic, Subtopic)`` folder/file names from a query.

    ``"Find recent research papers about RAG hallucination"`` → ``("RAG", "Hallucination")``.
    """
    originals: dict[str, str] = {
        w.lower(): w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query)
    }
    terms = salient_terms(query, drop_generic=True) or salient_terms(query)

    def display(term: str) -> str:
        orig = originals.get(term, term)
        return orig if orig.isupper() and len(orig) > 1 else orig.capitalize()

    if not terms:
        return "General", "Untitled"
    # Prefer an acronym (RAG, LLM, GPU) as the topic folder.
    topic_term = next((t for t in terms if originals.get(t, "").isupper()), terms[0])
    rest = [t for t in terms if t != topic_term][:3]
    topic = _safe_segment(display(topic_term)) or "General"
    subtopic = _safe_segment(" ".join(display(t) for t in rest)) or "Overview"
    return topic, subtopic


def _safe_segment(name: str) -> str:
    cleaned = _UNSAFE_RE.sub(" ", name).strip()
    return re.sub(r"\s+", " ", cleaned)[:60].strip()


def _tags(record: ResearchRecord) -> list[str]:
    out: list[str] = []
    for t in ["research", record.domain, *record.tags]:
        slug = re.sub(r"[^a-z0-9\-]+", "-", t.lower()).strip("-")
        if slug and slug not in out:
            out.append(slug)
    return out[:12]


def _render_new_note(
    topic: str, subtopic: str, record: ResearchRecord, stamp: str, tags: list[str]
) -> str:
    title = f"{topic} — {subtopic}"
    return (
        "---\n"
        f'title: "{title}"\n'
        f"created: {stamp}\n"
        f"updated: {stamp}\n"
        f"tags: [{', '.join(tags)}]\n"
        "queries:\n"
        f'  - "{_yaml_escape(record.query)}"\n'
        "---\n\n"
        f"# {title}\n\n"
    )


def _update_frontmatter(existing: str, stamp: str, query: str, tags: list[str]) -> str:
    m = _FRONTMATTER_RE.match(existing)
    if not m:
        return existing.rstrip() + "\n"
    fm = m.group(1)
    fm = re.sub(r"^updated:.*$", lambda _: f"updated: {stamp}", fm, flags=re.M)
    tag_match = re.search(r"^tags:\s*\[(.*)\]\s*$", fm, flags=re.M)
    if tag_match:
        current = [t.strip() for t in tag_match.group(1).split(",") if t.strip()]
        merged = current + [t for t in tags if t not in current]
        fm = fm[: tag_match.start()] + f"tags: [{', '.join(merged)}]" + fm[tag_match.end() :]
    line = f'  - "{_yaml_escape(query)}"'
    if line not in fm:
        fm = re.sub(r"^queries:\s*$", lambda _: "queries:\n" + line, fm, count=1, flags=re.M)
    return f"---\n{fm}\n---\n" + existing[m.end() :].rstrip() + "\n"


def _render_session(record: ResearchRecord, stamp: str) -> str:
    findings = record.findings
    summary = " ".join(truncate_words(e.claim, 30) for e in findings[:3])
    lines = [
        f"## Session {stamp}",
        "",
        f"**Query:** {record.query}  ",
        f"**Intent:** {record.domain} · **Tools:** {', '.join(record.tools) or 'none'}",
        "",
        "### Summary",
        summary,
        "",
        "### Key findings",
    ]
    for e in findings:
        cite = ""
        if e.source.url:
            cite = f" — [{_md_escape(e.source.title or e.source.url)}]({e.source.url})"
        elif e.source.title:
            cite = f" — {_md_escape(e.source.title)}"
        date = f" ({e.source.date[:10]})" if e.source.date else ""
        lines.append(f"- {_md_escape(truncate_words(e.claim, 60))}{cite}{date}")
    if record.entities:
        lines += ["", "### Entities", ", ".join(record.entities[:15])]
    urls = [(e.source.title, e.source.url) for e in findings if e.source.url]
    if urls:
        lines += ["", "### Sources"]
        lines += [f"- [{_md_escape(t or u)}]({u})" for t, u in urls]
    if record.related_notes:
        lines += ["", "### Related"]
        lines += [f"- [[{n}]]" for n in record.related_notes]
    lines += ["", " ".join(f"#{t}" for t in _tags(record)), ""]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".rr-", suffix=".md.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _headings(body: str) -> str:
    return " ".join(line.lstrip("# ") for line in body.splitlines() if line.startswith("#"))


def _best_passages(body: str, terms: set[str], limit: int = 4) -> list[MemoryPassage]:
    """Pick the note passages that best match the query terms.

    The note as a whole already matched, so passages without term overlap
    are still eligible (in document order) — they are the note's findings.
    """
    candidates: list[tuple[int, int, str]] = []
    for pos, block in enumerate(re.split(r"\n\s*\n|\n(?=- )", body)):
        lines = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
        text = "\n".join(lines).strip()
        if not text or text.startswith("**Query") or text.startswith("**Intent"):
            continue
        if len(_MD_LINK_RE.sub("", text).strip("-* \n").split()) < 4:
            continue  # link lists (Sources/Related), tag lines, entity lists
        overlap = len(terms & set(tokenize(text)))
        candidates.append((overlap, pos, text))
    candidates.sort(key=lambda c: (-c[0], c[1]))

    out: list[MemoryPassage] = []
    for _, _, text in candidates[:limit]:
        url_m = _URL_RE.search(text)
        url = (url_m.group(1) or url_m.group(2)) if url_m else None
        date_m = _DATE_IN_LINE_RE.search(text)
        clean = _URL_RE.sub("", text).replace("[", "").replace("]", "").lstrip("- ").strip()
        out.append(
            MemoryPassage(
                text=truncate_words(clean, 60), url=url, date=date_m.group(1) if date_m else None
            )
        )
    return out


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _md_escape(s: str) -> str:
    return s.replace("[", "(").replace("]", ")").replace("\n", " ")
