from __future__ import annotations

import json
import operator
import os
import re
import time
from datetime import date, timedelta
from html import escape
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Any

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
DEFAULT_GROQ_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GROQ_FALLBACKS = [
    model.strip()
    for model in os.getenv(
        "GROQ_FALLBACK_MODELS",
        "llama-3.1-8b-instant",
    ).split(",")
    if model.strip()
]
MODEL_CANDIDATES = [DEFAULT_GROQ_MODEL, *[m for m in DEFAULT_GROQ_FALLBACKS if m != DEFAULT_GROQ_MODEL]]
MODEL_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
MODEL_RATE_LIMIT_RETRIES = int(os.getenv("GROQ_RATE_LIMIT_RETRIES", "2"))
MODEL_MAX_WAIT_SECONDS = float(os.getenv("GROQ_MAX_WAIT_SECONDS", "20"))
_MODEL_CACHE: dict[str, ChatGroq] = {}


def _get_model(model_name: str) -> ChatGroq:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = ChatGroq(model=model_name, temperature=MODEL_TEMPERATURE)
    return _MODEL_CACHE[model_name]


def _is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    text = str(exc).lower()
    return "ratelimit" in name.lower() or "rate limit" in text or "429" in text


def _is_retryable_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return _is_rate_limit_error(exc) or "decommissioned" in text or "no longer supported" in text


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    text = str(exc)
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _invoke_llm(messages: list[Any], schema: Any = None):
    last_rate_limit: Optional[Exception] = None

    for model_name in MODEL_CANDIDATES:
        client = _get_model(model_name)
        runnable = client.with_structured_output(schema) if schema is not None else client

        for attempt in range(MODEL_RATE_LIMIT_RETRIES + 1):
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                if not _is_retryable_model_error(exc):
                    raise

                last_rate_limit = exc
                wait_seconds = _retry_after_seconds(exc)
                is_temporary_rate_limit = _is_rate_limit_error(exc) and wait_seconds is not None

                if is_temporary_rate_limit and attempt < MODEL_RATE_LIMIT_RETRIES:
                    time.sleep(min(wait_seconds + 0.5, MODEL_MAX_WAIT_SECONDS))
                    continue

                break

    tried_models = ", ".join(MODEL_CANDIDATES)
    detail = str(last_rate_limit) if last_rate_limit else "No model could complete the request."
    raise RuntimeError(
        "All configured Groq models are currently rate-limited. "
        f"Tried: {tried_models}. "
        "Wait for the quota window to reset, set GROQ_PRIMARY_MODEL / GROQ_FALLBACK_MODELS to models with available quota, "
        "or upgrade the Groq project tier.\n\n"
        f"Last provider error: {detail}"
    ) from last_rate_limit

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Return output that matches the schema exactly.
- `needs_research` must be a JSON boolean: true or false, never a quoted string.
- `mode` must be exactly one of: "closed_book", "hybrid", "open_book".
- `reason` must be a short string.
- `queries` must be an array of strings.
- If `needs_research` is false, return `queries` as an empty array.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–5 high-signal, scoped queries (keep it concise).
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decision = _invoke_llm(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        schema=RouterDecision,
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        # ✅ FIX: Cap queries to 5 to reduce downstream token load
        "queries": decision.queries[:5],
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    """
    ✅ FIX: Reduced default max_results from 5 → 3 to keep token counts manageable.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            # ✅ FIX: Truncate snippet at source to 300 chars
            snippet = r.get("content") or r.get("snippet") or ""
            out.append(
                {
                    "title": (r.get("title") or "")[:120],
                    "url": r.get("url") or "",
                    "snippet": snippet[:300],
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets under 200 characters.
- Deduplicate by URL.
- Return at most 12 evidence items total.
"""

def _normalize_evidence_items(raw_items: List[dict]) -> List[EvidenceItem]:
    dedup: dict[str, EvidenceItem] = {}

    for item in raw_items:
        url = (item.get("url") or "").strip()
        if not url:
            continue

        title = (item.get("title") or "Untitled source").strip()[:120]
        snippet = (item.get("snippet") or "").strip()[:200] or None
        published_at = item.get("published_at")
        normalized_published_at = None
        if published_at:
            parsed = _iso_to_date(str(published_at))
            if parsed:
                normalized_published_at = parsed.isoformat()

        dedup[url] = EvidenceItem(
            title=title,
            url=url,
            published_at=normalized_published_at,
            snippet=snippet,
            source=(item.get("source") or None),
        )

    return list(dedup.values())[:12]


def research_node(state: State) -> dict:
    # ✅ FIX: Cap queries processed to 5 (was 10)
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        # ✅ FIX: max_results=3 per query (was 6) → max 15 raw results total
        raw.extend(_tavily_search(q, max_results=3))

    if not raw:
        return {"evidence": []}

    # ✅ FIX: Hard-cap raw results sent to LLM at 12 items
    raw = raw[:12]

    # ✅ FIX: Ensure all snippets are trimmed before normalization
    for r in raw:
        if r.get("snippet"):
            r["snippet"] = r["snippet"][:250]
        if r.get("title"):
            r["title"] = r["title"][:100]

    evidence = _normalize_evidence_items(raw)

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        recent_dated = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]
        if recent_dated:
            evidence = recent_dated
        else:
            # If sources don't expose publish dates, keep a small fallback set
            # so the Evidence tab and planner still have something useful to show.
            evidence = evidence[:5]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–7 tasks (keep it concise), each with goal + 3–5 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).

Return plain JSON only, with no markdown fences or extra commentary.
The JSON object must contain exactly these top-level keys:
- blog_title
- audience
- tone
- blog_kind
- constraints
- tasks

Each item in tasks must be an object with:
- id
- title
- goal
- bullets
- target_words
- tags
- requires_research
- requires_citations
- requires_code
"""


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _coerce_plan_dict(data: dict, topic: str, forced_kind: Optional[str] = None) -> Plan:
    allowed_blog_kinds = {"explainer", "tutorial", "news_roundup", "comparison", "system_design"}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Plan JSON is missing a valid tasks list.")

    cleaned_tasks = []
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        raw_id = task.get("id", idx)
        if isinstance(raw_id, int):
            task_id = raw_id
        else:
            match = re.search(r"\d+", str(raw_id))
            task_id = int(match.group()) if match else idx
        cleaned_tasks.append(
            {
                "id": task_id,
                "title": str(task.get("title", f"Section {idx}")).strip(),
                "goal": str(task.get("goal", f"Explain {topic}")).strip(),
                "bullets": [str(b).strip() for b in (task.get("bullets") or []) if str(b).strip()][:6] or [f"Key point {idx}", "Why it matters", "Practical takeaway"],
                "target_words": int(task.get("target_words", 180)),
                "tags": [str(t).strip() for t in (task.get("tags") or []) if str(t).strip()],
                "requires_research": bool(task.get("requires_research", forced_kind == "news_roundup")),
                "requires_citations": bool(task.get("requires_citations", forced_kind == "news_roundup")),
                "requires_code": bool(task.get("requires_code", False)),
            }
        )

    raw_blog_kind = str(data.get("blog_kind", "explainer")).strip()
    blog_kind = forced_kind or (raw_blog_kind if raw_blog_kind in allowed_blog_kinds else "explainer")

    plan_data = {
        "blog_title": str(data.get("blog_title", topic)).strip() or topic,
        "audience": str(data.get("audience", "developers and technical readers")).strip(),
        "tone": str(data.get("tone", "clear, practical, and well-structured")).strip(),
        "blog_kind": blog_kind,
        "constraints": [str(c).strip() for c in (data.get("constraints") or []) if str(c).strip()],
        "tasks": cleaned_tasks[:7],
    }
    return Plan(**plan_data)

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    # ✅ FIX: Only pass first 10 evidence items with trimmed snippets to orchestrator
    evidence_summary = [
        {
            "title": e.title,
            "url": e.url,
            "published_at": e.published_at,
            "snippet": (e.snippet or "")[:150],
        }
        for e in evidence[:10]
    ]

    plan_response = _invoke_llm(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{evidence_summary}"
                )
            ),
        ]
    )
    plan = _coerce_plan_dict(
        _extract_json_object(plan_response.content),
        topic=state["topic"],
        forced_kind=forced_kind,
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                # ✅ FIX: Pass only first 8 evidence items to each worker (titles+URLs only)
                "evidence": [
                    {
                        "title": e.title,
                        "url": e.url,
                        "published_at": e.published_at,
                        "snippet": (e.snippet or "")[:120],
                        "source": e.source,
                    }
                    for e in state.get("evidence", [])[:8]
                ],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".
- Do not include images, figures, placeholders, or captions.
- Use clean Markdown structure:
  - Start with exactly one `##` heading for the section.
  - Use short paragraphs, not one long block of text.
  - Use `###` subheadings when helpful.
  - Use bullet lists for examples, steps, or feature roundups.
  - Leave a blank line between headings, paragraphs, and lists.
  - Do not write the whole section as a single paragraph.

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def _normalize_markdown(md: str) -> str:
    text = (md or "").replace("\r\n", "\n").strip()

    # Some model outputs arrive with escaped newlines instead of real ones.
    if "\\n" in text and text.count("\n") <= 2:
        text = text.replace("\\n", "\n")

    # Keep image failure blocks on their own lines so they don't break bullets/paragraphs.
    text = re.sub(r"([^\n])\s+(> \*\*\[IMAGE GENERATION FAILED\]\*\*)", r"\1\n\n\2", text)

    # Ensure headings start on fresh lines.
    text = re.sub(r"(?<!\n)(#{1,6}\s)", r"\n\n\1", text)

    # Remove stray heading markers like standalone '#' lines.
    text = re.sub(r"(?m)^\s*#{1,6}\s*$\n?", "", text)

    # Ensure bullets start on fresh lines.
    text = re.sub(r"(?<!\n)([*-]\s+\*\*)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([*-]\s+)", r"\n\1", text)

    # Compress excessive blank lines while preserving readability.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def _format_section_markdown(section_md: str, section_title: str) -> str:
    text = _normalize_markdown(section_md)
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned: List[str] = []
    heading_seen = False
    skip_title_line = section_title.strip().lower()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        lower_line = line.lower()
        if lower_line == skip_title_line or lower_line == f"## {skip_title_line}":
            if heading_seen:
                continue

        if line.startswith("## "):
            if not heading_seen:
                heading_seen = True
                cleaned.append(line)
            continue

        if line.startswith("# "):
            if not heading_seen:
                cleaned.append(f"## {line[2:].strip() or section_title}")
                cleaned.append("")
                heading_seen = True
            else:
                cleaned.append("")
                cleaned.append(f"### {line[2:].strip()}")
                cleaned.append("")
            continue

        if not heading_seen:
            cleaned.append(f"## {section_title}")
            cleaned.append("")
            heading_seen = True

        if re.fullmatch(r"[A-Z][A-Za-z0-9 &/(),'-]{2,80}", line) and not line.endswith("."):
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            cleaned.append(f"### {line}")
            cleaned.append("")
            continue

        if (
            len(line) <= 90
            and not line.startswith(("* ", "- ", ">"))
            and ":" not in line
            and not line.endswith(".")
            and not line.endswith("!")
            and not line.endswith("?")
        ):
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            cleaned.append(f"### {line}")
            cleaned.append("")
            continue

        if cleaned and cleaned[-1].startswith(("### ", "* ", "- ")) and not line.startswith(("> ", "* ", "- ")):
            cleaned.append("")

        cleaned.append(line)

    if not heading_seen:
        cleaned.insert(0, "")
        cleaned.insert(0, f"## {section_title}")

    formatted = "\n".join(cleaned)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted).strip()
    return formatted + "\n"


def _finalize_blog_markdown(md: str, blog_title: str) -> str:
    text = _normalize_markdown(md)
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: List[str] = []
    title_written = False
    normalized_title = blog_title.strip().lower()

    for raw in lines:
        line = raw.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if re.fullmatch(r"#{1,6}", line):
            continue

        if line.lower() == normalized_title:
            if title_written:
                continue
            cleaned.append(f"# {blog_title}")
            title_written = True
            cleaned.append("")
            continue

        if line.startswith("# "):
            heading_text = line[2:].strip()
            if not title_written:
                cleaned.append(f"# {heading_text or blog_title}")
                title_written = True
            else:
                cleaned.append(f"## {heading_text}")
            cleaned.append("")
            continue

        if line.startswith("## ") or line.startswith("### "):
            cleaned.append(line)
            cleaned.append("")
            continue

        cleaned.append(line)

    if not title_written:
        cleaned.insert(0, "")
        cleaned.insert(0, f"# {blog_title}")

    output = "\n".join(cleaned)
    output = re.sub(r"\n{3,}", "\n\n", output).strip()
    return output + "\n"

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    # ✅ FIX: Evidence items here are already plain dicts (trimmed in fanout)
    evidence = payload.get("evidence", [])

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    # ✅ FIX: Only title + URL in evidence_text (no snippets), cap at 8 items
    evidence_text = "\n".join(
        f"- {e.get('title', '')} | {e.get('url', '')} | {e.get('published_at') or 'date:unknown'}"
        for e in evidence[:8]
    )

    section_md = _invoke_llm(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    section_md = _format_section_markdown(section_md, task.title)

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [_normalize_markdown(md) for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = _normalize_markdown(f"# {plan.blog_title}\n\n{body}\n")
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return plain JSON only, with no markdown fences or extra commentary.
The JSON object must contain:
- md_with_placeholders
- images

Each item in images must contain:
- placeholder
- filename
- alt
- caption
- prompt
- size
- quality
"""


def _coerce_image_plan_dict(data: dict, merged_md: str) -> GlobalImagePlan:
    raw_images = data.get("images") or []
    images: List[ImageSpec] = []

    for idx, image in enumerate(raw_images[:3], start=1):
        if not isinstance(image, dict):
            continue
        alt = str(image.get("alt", f"Diagram {idx}")).strip() or f"Diagram {idx}"
        caption = str(image.get("caption", alt)).strip() or alt
        filename = str(image.get("filename", f"diagram_{idx}.svg")).strip() or f"diagram_{idx}.svg"
        size = str(image.get("size", "1024x1024")).strip()
        if size not in {"1024x1024", "1024x1536", "1536x1024"}:
            size = "1024x1024"
        quality = str(image.get("quality", "medium")).strip()
        if quality not in {"low", "medium", "high"}:
            quality = "medium"

        images.append(
            ImageSpec(
                placeholder=f"[[IMAGE_{idx}]]",
                filename=filename,
                alt=alt,
                caption=caption,
                prompt=str(image.get("prompt", caption)).strip() or caption,
                size=size,  # type: ignore[arg-type]
                quality=quality,  # type: ignore[arg-type]
            )
        )

    md_with_placeholders = str(data.get("md_with_placeholders", merged_md)).strip() or merged_md
    return GlobalImagePlan(md_with_placeholders=md_with_placeholders, images=images)

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    return {
        "md_with_placeholders": merged_md,
        "image_specs": [],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _image_failure_note(spec: dict, error: Exception) -> str:
    message = str(error).lower()
    if "resource_exhausted" in message or "quota exceeded" in message or "429" in message:
        reason = "Image skipped because the image API quota is exhausted."
    else:
        reason = "Image could not be generated automatically."

    caption = spec.get("caption", "").strip()
    if caption:
        return f"*Image omitted: {caption}. {reason}*\n"
    return f"*Image omitted. {reason}*\n"


def _svg_dimensions(size: str) -> tuple[int, int]:
    if size == "1024x1536":
        return (960, 1280)
    if size == "1536x1024":
        return (1280, 960)
    return (1024, 1024)


def _wrap_svg_text(text: str, width: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _extract_diagram_labels(spec: dict) -> List[str]:
    source = " ".join(
        [
            str(spec.get("caption", "")),
            str(spec.get("alt", "")),
            str(spec.get("prompt", "")),
        ]
    )
    match = re.search(r"includes? (.+)", source, flags=re.IGNORECASE)
    if match:
        source = match.group(1)

    pieces = re.split(r",|;|\band\b|\bas well as\b|\n", source)
    labels = []
    for piece in pieces:
        cleaned = re.sub(r"[^A-Za-z0-9 /&()-]+", " ", piece).strip(" .:-")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if 3 <= len(cleaned) <= 40:
            labels.append(cleaned.title())

    deduped: List[str] = []
    for label in labels:
        if label.lower() not in {d.lower() for d in deduped}:
            deduped.append(label)
    return deduped[:5]


def _infer_diagram_kind(spec: dict) -> str:
    text = " ".join(
        [str(spec.get("filename", "")), str(spec.get("alt", "")), str(spec.get("caption", "")), str(spec.get("prompt", ""))]
    ).lower()
    if any(word in text for word in ["timeline", "roadmap", "evolution"]):
        return "timeline"
    if any(word in text for word in ["checklist", "steps", "template", "goals"]):
        return "checklist"
    if any(word in text for word in ["compare", "comparison", "versus", "vs"]):
        return "comparison"
    if any(word in text for word in ["flow", "process", "diagram", "workflow"]):
        return "flow"
    return "cards"


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">',
        f'<rect width="{width}" height="{height}" rx="28" fill="#F7F3EA"/>',
        f'<rect x="20" y="20" width="{width - 40}" height="{height - 40}" rx="24" fill="#FFFDFC" stroke="#D7C7B0" stroke-width="2"/>',
    ]


def _svg_title_block(spec: dict, width: int) -> List[str]:
    title = escape(str(spec.get("alt") or spec.get("caption") or "Diagram"))
    caption = escape(str(spec.get("caption") or ""))
    lines = _wrap_svg_text(title, 30)[:2]
    caption_lines = _wrap_svg_text(caption, 60)[:2]
    out = ['<g font-family="Segoe UI, Arial, sans-serif" fill="#24323D">']
    y = 84
    for line in lines:
        out.append(f'<text x="64" y="{y}" font-size="34" font-weight="700">{escape(line)}</text>')
        y += 38
    for line in caption_lines:
        out.append(f'<text x="64" y="{y}" font-size="18" fill="#5C6B75">{escape(line)}</text>')
        y += 24
    out.append("</g>")
    return out


def _render_flow_svg(spec: dict, width: int, height: int) -> str:
    labels = _extract_diagram_labels(spec) or ["Notice Pattern", "Pause", "Reframe Thought", "Act With Intent"]
    if len(labels) < 4:
        labels = (labels + ["Clarify", "Respond"])[:4]
    y = height // 2 - 70
    box_w = min(220, (width - 160) // 4)
    gap = 28
    x = 56
    parts = _svg_header(width, height) + _svg_title_block(spec, width)
    colors = ["#DCEBFF", "#FBE4D8", "#E3F4E8", "#FFF0B8"]
    for idx, label in enumerate(labels[:4]):
        bx = x + idx * (box_w + gap)
        parts.append(f'<rect x="{bx}" y="{y}" width="{box_w}" height="140" rx="24" fill="{colors[idx % len(colors)]}" stroke="#24323D" stroke-width="2"/>')
        wrapped = _wrap_svg_text(label, 18)[:3]
        ty = y + 54
        for line in wrapped:
            parts.append(f'<text x="{bx + box_w/2}" y="{ty}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="22" font-weight="600" fill="#24323D">{escape(line)}</text>')
            ty += 26
        if idx < 3:
            ax = bx + box_w
            parts.append(f'<path d="M {ax + 8} {y + 70} H {ax + gap - 18}" stroke="#24323D" stroke-width="4" stroke-linecap="round"/>')
            parts.append(f'<path d="M {ax + gap - 30} {y + 56} L {ax + gap - 18} {y + 70} L {ax + gap - 30} {y + 84}" stroke="#24323D" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _render_checklist_svg(spec: dict, width: int, height: int) -> str:
    labels = _extract_diagram_labels(spec) or ["Choose One Small Goal", "Break It Into Steps", "Track Progress", "Celebrate Wins"]
    parts = _svg_header(width, height) + _svg_title_block(spec, width)
    start_y = 230
    for idx, label in enumerate(labels[:5]):
        row_y = start_y + idx * 120
        parts.append(f'<rect x="72" y="{row_y}" width="{width - 144}" height="84" rx="20" fill="#FFFFFF" stroke="#D7C7B0" stroke-width="2"/>')
        parts.append(f'<rect x="98" y="{row_y + 22}" width="36" height="36" rx="10" fill="#DFF3E4" stroke="#2E6F40" stroke-width="2"/>')
        parts.append(f'<path d="M 108 {row_y + 42} L 116 {row_y + 50} L 128 {row_y + 34}" stroke="#2E6F40" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        wrapped = _wrap_svg_text(label, 42)[:2]
        ty = row_y + 40
        for line in wrapped:
            parts.append(f'<text x="160" y="{ty}" font-family="Segoe UI, Arial, sans-serif" font-size="24" font-weight="600" fill="#24323D">{escape(line)}</text>')
            ty += 24
    parts.append("</svg>")
    return "\n".join(parts)


def _render_timeline_svg(spec: dict, width: int, height: int) -> str:
    labels = _extract_diagram_labels(spec) or ["Start", "Build Skills", "Apply Consistently", "Review Progress"]
    points = labels[:4]
    parts = _svg_header(width, height) + _svg_title_block(spec, width)
    y = height // 2 + 50
    parts.append(f'<path d="M 90 {y} H {width - 90}" stroke="#24323D" stroke-width="6" stroke-linecap="round"/>')
    gap = (width - 180) / max(1, len(points) - 1)
    for idx, label in enumerate(points):
        cx = 90 + idx * gap
        parts.append(f'<circle cx="{cx}" cy="{y}" r="18" fill="#F19C79" stroke="#24323D" stroke-width="3"/>')
        parts.append(f'<line x1="{cx}" y1="{y - 18}" x2="{cx}" y2="{y - 88}" stroke="#24323D" stroke-width="3"/>')
        wrapped = _wrap_svg_text(label, 16)[:3]
        ty = y - 108
        for line in wrapped:
            parts.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="21" font-weight="600" fill="#24323D">{escape(line)}</text>')
            ty += 24
    parts.append("</svg>")
    return "\n".join(parts)


def _render_comparison_svg(spec: dict, width: int, height: int) -> str:
    labels = _extract_diagram_labels(spec) or ["Current Approach", "Improved Approach", "Benefit One", "Benefit Two"]
    left_title = labels[0]
    right_title = labels[1] if len(labels) > 1 else "Alternative"
    left_points = labels[2:4] or ["Baseline", "Trade-offs"]
    right_points = labels[4:6] or ["Faster Progress", "Clearer Outcomes"]
    parts = _svg_header(width, height) + _svg_title_block(spec, width)
    col_w = (width - 192) / 2
    top = 240
    for idx, (title, bullets, fill) in enumerate([(left_title, left_points, "#E8F0FE"), (right_title, right_points, "#FCE8D8")]):
        x = 64 + idx * (col_w + 64)
        parts.append(f'<rect x="{x}" y="{top}" width="{col_w}" height="{height - top - 80}" rx="24" fill="{fill}" stroke="#24323D" stroke-width="2"/>')
        parts.append(f'<text x="{x + col_w/2}" y="{top + 52}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#24323D">{escape(title[:26])}</text>')
        for bullet_idx, bullet in enumerate(bullets[:3]):
            by = top + 110 + bullet_idx * 82
            parts.append(f'<circle cx="{x + 34}" cy="{by - 10}" r="7" fill="#24323D"/>')
            wrapped = _wrap_svg_text(bullet, 24)[:2]
            ty = by
            for line in wrapped:
                parts.append(f'<text x="{x + 56}" y="{ty}" font-family="Segoe UI, Arial, sans-serif" font-size="22" fill="#24323D">{escape(line)}</text>')
                ty += 24
    parts.append("</svg>")
    return "\n".join(parts)


def _render_cards_svg(spec: dict, width: int, height: int) -> str:
    labels = _extract_diagram_labels(spec) or ["Key Idea", "Why It Matters", "How To Apply", "Next Step"]
    parts = _svg_header(width, height) + _svg_title_block(spec, width)
    card_w = (width - 192) / 2
    card_h = (height - 340) / 2
    fills = ["#E8F0FE", "#FCE8D8", "#E3F4E8", "#FFF0B8"]
    positions = [(64, 240), (64 + card_w + 64, 240), (64, 240 + card_h + 40), (64 + card_w + 64, 240 + card_h + 40)]
    for idx, label in enumerate(labels[:4]):
        x, y = positions[idx]
        parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="24" fill="{fills[idx % len(fills)]}" stroke="#24323D" stroke-width="2"/>')
        wrapped = _wrap_svg_text(label, 18)[:3]
        ty = y + 56
        for line in wrapped:
            parts.append(f'<text x="{x + 28}" y="{ty}" font-family="Segoe UI, Arial, sans-serif" font-size="24" font-weight="600" fill="#24323D">{escape(line)}</text>')
            ty += 28
    parts.append("</svg>")
    return "\n".join(parts)


def _generate_local_svg(spec: dict) -> str:
    width, height = _svg_dimensions(str(spec.get("size") or "1024x1024"))
    kind = _infer_diagram_kind(spec)
    if kind == "timeline":
        return _render_timeline_svg(spec, width, height)
    if kind == "checklist":
        return _render_checklist_svg(spec, width, height)
    if kind == "comparison":
        return _render_comparison_svg(spec, width, height)
    if kind == "flow":
        return _render_flow_svg(spec, width, height)
    return _render_cards_svg(spec, width, height)


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = _finalize_blog_markdown(
        state.get("md_with_placeholders") or state["merged_md"],
        plan.blog_title,
    )
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = str(spec["filename"])
        svg_filename = f"{Path(filename).stem}.svg"
        out_path = images_dir / filename
        svg_path = images_dir / svg_filename

        if not svg_path.exists():
            try:
                svg_path.write_text(_generate_local_svg(spec), encoding="utf-8")
            except Exception as e:
                md = md.replace(placeholder, _image_failure_note(spec, e))
                continue

        img_md = f"![{spec['alt']}](images/{svg_filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    md = _normalize_markdown(md)
    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
