"""Task decomposition and minimum-tool selection.

Turns a (context-resolved) query into an internal ``ExecutionPlan``::

    "Find recent RAG papers and their GitHub implementations"
      → t1 find_papers          academic_search (google_scholar)
      → t2 find_implementations github_search   (google, site:github.com)
      parallelizable: true

Decomposition is conservative: a query is only split when the parts need
*different* tools, when the second part has no subject of its own ("…and
their implementations"), or when the user explicitly chains requests.
Otherwise a single search handles "A and B" — fewer calls, same quality.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from research_router.context.manager import ResolvedContext
from research_router.context.text import GENERIC_TERMS, salient_terms
from research_router.models.intent import ResearchDepth, ResearchDomain
from research_router.models.plan import SearchPlan
from research_router.router.classifier import QueryClassifier
from research_router.router.intent import IntentAnalyser
from research_router.router.planner import ResearchPlanner

TOOL_NAMES: dict[str, str] = {
    "google": "web_search",
    "google_news": "news_search",
    "google_scholar": "academic_search",
    "google_jobs": "jobs_search",
    "google_shopping": "shopping_search",
    "google_maps": "places_search",
}
_TASK_NAMES: dict[ResearchDomain, str] = {
    ResearchDomain.GENERAL: "find_web_sources",
    ResearchDomain.NEWS: "find_news",
    ResearchDomain.ACADEMIC: "find_papers",
    ResearchDomain.JOBS: "find_jobs",
    ResearchDomain.SHOPPING: "find_products",
    ResearchDomain.PLACES: "find_places",
}

_SPLIT_RE = re.compile(
    r"\s*(?:;|\s+as well as\s+|\s+along with\s+|\s+plus\s+"
    r"|\s+and\s+(?:also\s+)?(?=(?:their|its|the|related|corresponding|any|some|recent|latest"
    r"|find|get|show|list|search|look\s+up)\b))\s*",
    re.I,
)
_EXPLICIT_CHAIN_RE = re.compile(r"^(?:find|get|show|list|search|look\s+up)\b", re.I)
_LEADING_REF_RE = re.compile(r"^(?:their|its|the|related|corresponding|any|some)\s+", re.I)
_CODE_STRONG_RE = re.compile(
    r"\b(github|gitlab|repo|repos|repository|repositories|source\s+code|open[- ]source\s+code)\b",
    re.I,
)
_CODE_WEAK_RE = re.compile(r"\b(implementations?|code|codebase|library|libraries)\b", re.I)
_CODE_WORDS = frozenset(
    "github gitlab repo repos repository repositories source code open-source site".split()  # noqa: SIM905
)


@dataclass
class PlannedTask:
    id: str
    name: str
    tool: str
    engine: str
    query: str
    plan: SearchPlan
    expansions: list[SearchPlan] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.name,
            "tool": self.tool,
            "engine": self.engine,
            "query": self.plan.query,
            "parameters": self.plan.parameters,
            "depends_on": self.depends_on,
            "adaptive_expansions": [p.query for p in self.expansions],
        }


@dataclass
class ExecutionPlan:
    intent: str
    user_intent: str
    tasks: list[PlannedTask]
    parallelizable: bool
    context_only: bool
    freshness_required: bool

    @property
    def primary(self) -> PlannedTask | None:
        return self.tasks[0] if self.tasks else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "user_intent": self.user_intent,
            "tasks": [t.as_dict() for t in self.tasks],
            "parallelizable": self.parallelizable,
            "context_only": self.context_only,
            "freshness_required": self.freshness_required,
        }


class TaskPlanner:
    def __init__(
        self,
        classifier: QueryClassifier,
        analyser: IntentAnalyser,
        planner: ResearchPlanner,
        max_tasks: int = 4,
    ) -> None:
        self._classifier = classifier
        self._analyser = analyser
        self._planner = planner
        self._max_tasks = max_tasks

    async def plan(
        self, resolved: ResolvedContext, depth: ResearchDepth, max_results: int
    ) -> ExecutionPlan:
        if not resolved.needs_search:
            return ExecutionPlan(
                intent="context",
                user_intent=resolved.user_intent,
                tasks=[],
                parallelizable=False,
                context_only=True,
                freshness_required=False,
            )

        parts = resolved.search_subjects or self.decompose(resolved.resolved_query)
        parts = parts[: self._max_tasks]
        split = len(parts) > 1
        built = await asyncio.gather(
            *(self._build_task(p, depth, max_results, split) for p in parts)
        )

        tasks: list[PlannedTask] = []
        seen: set[tuple[str, str]] = set()
        for task in built:
            key = (task.engine, " ".join(sorted(salient_terms(task.plan.query))))
            if key in seen:
                continue  # identical search — never call the same tool twice
            seen.add(key)
            task.id = f"t{len(tasks) + 1}"
            tasks.append(task)

        domains = {t.plan.domain.value for t in tasks}
        return ExecutionPlan(
            intent=domains.pop() if len(domains) == 1 else "multi",
            user_intent=resolved.user_intent,
            tasks=tasks,
            parallelizable=len(tasks) > 1 and all(not t.depends_on for t in tasks),
            context_only=False,
            freshness_required=any(
                t.plan.parameters.get("freshness") or t.plan.parameters.get("date_range")
                for t in tasks
            ),
        )

    # ── decomposition ─────────────────────────────────────────────

    def decompose(self, query: str) -> list[str]:
        """Split *query* into independent sub-requests (or return it whole)."""
        raw_parts = [p.strip(" ,.") for p in _SPLIT_RE.split(query) if p and p.strip(" ,.")]
        if len(raw_parts) < 2:
            return [query]

        first = raw_parts[0]
        subject = salient_terms(first, drop_generic=True)
        first_tool = self._tool_key(first, split_part=False)
        parts = [first]
        for part in raw_parts[1:]:
            explicit = bool(_EXPLICIT_CHAIN_RE.match(part))
            core = _LEADING_REF_RE.sub("", part)
            own_subject = [
                w for w in salient_terms(core, drop_generic=True) if w not in _CODE_WORDS
            ]
            if not own_subject and subject:
                # "their implementations" → "RAG implementations"
                core = f"{' '.join(subject)} {core}"
            different_tool = self._tool_key(core, split_part=True) != first_tool
            if explicit or different_tool or not own_subject:
                parts.append(core)
            else:
                parts[-1] = f"{parts[-1]} and {part}"  # same tool, own subject: one search
        return parts if len(parts) > 1 else [query]

    def _tool_key(self, text: str, *, split_part: bool) -> str:
        if _is_code_request(text, split_part):
            return "github"
        return self._classifier.classify(text).domain.value

    # ── task construction ─────────────────────────────────────────

    async def _build_task(
        self, text: str, depth: ResearchDepth, max_results: int, split_part: bool
    ) -> PlannedTask:
        if _is_code_request(text, split_part):
            core = [w for w in salient_terms(text) if w not in _CODE_WORDS]
            plan = SearchPlan(
                domain=ResearchDomain.GENERAL,
                engine="google",
                query=f"{' '.join(core) or text} site:github.com",
                max_results=max_results,
                research_depth=depth,
            )
            return PlannedTask(
                id="", name="find_implementations", tool="github_search",
                engine="google", query=text, plan=plan,
            )  # fmt: skip

        intent = await self._analyser.analyse(text)
        intent = intent.model_copy(update={"research_depth": depth})
        full = self._planner.plan(intent)
        primary = full.model_copy(update={"sub_queries": [], "max_results": max_results})
        expansions = [sq.model_copy(update={"max_results": max_results}) for sq in full.sub_queries]
        return PlannedTask(
            id="",
            name=_TASK_NAMES.get(primary.domain, "find_web_sources"),
            tool=TOOL_NAMES.get(primary.engine, primary.engine),
            engine=primary.engine,
            query=text,
            plan=primary,
            expansions=expansions,
        )


def _is_code_request(text: str, split_part: bool) -> bool:
    if _CODE_STRONG_RE.search(text):
        return True
    # "…and their implementations" — weak code words only count for split parts,
    # never for a standalone query such as "RAG implementation best practices".
    if split_part and _CODE_WEAK_RE.search(text):
        topic = [w for w in salient_terms(text) if w not in GENERIC_TERMS]
        return len(topic) <= 3
    return False
