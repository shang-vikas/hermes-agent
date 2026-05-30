"""Task classifier — estimates task requirement vector from goal/context/toolsets.

Produces [coding, reasoning, knowledge] weights (0-1) that correspond to
BenchLM's category dimensions so we can do a direct cosine-like match.

ARCHITECTURE: 3-layer hybrid

  Layer 1 — Expanded SDLC keyword classifier (comprehensive sets)
  Layer 2 — ministral-3:14b LLM booster (low-confidence only)
  Layer 3 — Toolset + file-ref overlay (always on)

Flow:
  1. Run expanded keyword classifier → [c, r, k] + confidence score
  2. If confidence >= MIN_CONFIDENCE, return keyword result (fast path ~70%)
  3. If confidence < MIN_CONFIDENCE, invoke LLM booster for the task
  4. Always overlay toolset + file-ref signals
  5. Normalize to [0, 1] preserving relative strengths

Usage:
    from agent.task_classifier import estimate_task_vector, match_model
    vec = estimate_task_vector("Fix test failures", "path: test_x.py", ["terminal"])
    model_id = match_model(vec, available_models)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# LAYER 1: EXPANDED SDLC KEYWORD SETS
# =====================================================================
# Each category spans the full Software Development Life Cycle:
# Requirement → Design → Implementation → Testing → Deployment →
# Maintenance → Documentation → Research

CODING_KEYS = {
    # ── Implementation ──
    "implement", "build", "write", "code", "program", "develop",
    "create", "construct", "produce", "author", "craft",

    # ── Bug fixing ──
    "fix", "bug", "issue", "error", "crash", "broken", "failing",
    "failure", "fault", "defect", "regression", "hotfix", "rollback",
    "patch", "workaround", "hack", "debug", "troubleshoot",

    # ── Testing ──
    "test", "pytest", "unittest", "integration test", "unit test",
    "e2e", "spec", "coverage", "assert", "mock", "stub", "fixture",
    "snapshot", "regression test", "tester", "qa",

    # ── CI / CD ──
    "ci", "cd", "pipeline", "build", "compile", "deploy",
    "rollout", "release", "canary", "blue-green", "containerize",
    "docker", "dockerfile", "k8s", "kubernetes", "helm",

    # ── Refactoring / Optimization ──
    "refactor", "clean", "deduplicate", "simplify", "optimize",
    "performance", "latency", "throughput", "bottleneck", "inline",
    "extract", "consolidate", "migrate", "upgrade", "downgrade",
    "deprecate", "remove", "delete",

    # ── API / Service ──
    "api", "endpoint", "route", "handler", "middleware", "service",
    "microservice", "rest", "grpc", "graphql", "webhook",
    "callback", "request", "response", "serialize", "deserialize",
    "schema", "validation", "auth", "authentication", "authorization",
    "oauth", "jwt", "session", "cookie", "token",

    # ── Database / Storage ──
    "query", "sql", "database", "migration", "orm", "schema",
    "index", "transaction", "store", "cache", "redis", "postgres",
    "mongodb", "elasticsearch",

    # ── Frontend ──
    "ui", "ux", "component", "view", "template", "render",
    "css", "html", "javascript", "typescript", "react", "vue",
    "angular", "form", "input", "button", "modal", "dropdown",
    "responsive", "mobile", "ios", "android",

    # ── Config / Infra ──
    "config", "configuration", "yaml", "toml", "json", "env",
    "environment", "infrastructure", "terraform", "ansible",
    "docker-compose", "dockerfile",

    # ── Code review / PR ──
    "pr", "pull request", "merge", "review", "lgtm", "approve",
    "squash", "rebase", "cherry-pick", "commit", "git",
    "branch", "conflict", "revert",

    # ── Linting / Types ──
    "lint", "prettier", "eslint", "ruff", "mypy", "typed",
    "type-ignore", "cast", "annotation", "pyright",

    # ── Error signals ──
    "traceback", "stack trace", "exception", "null pointer", "npe",
    "segfault", "timeout", "rate limit", "429", "500", "import error",
    "module not found", "importerror", "syntax error",
}


REASONING_KEYS = {
    # ── Architecture / Design ──
    "architecture", "design", "pattern", "trade-off", "decision",
    "spike", "proposal", "adr", "rfc", "rfd", "tech spec",
    "technical specification", "design doc", "design review",
    "schema design", "data model", "domain model", "entity",
    "relationship", "abstraction", "component", "module",

    # ── Analysis / Evaluation ──
    "analyze", "evaluate", "assess", "compare", "contrast",
    "choose", "select", "decide", "feasibility", "viability",
    "pros and cons", "advantage", "disadvantage", "trade-offs",
    "consider", "weigh", "strategy", "approach",

    # ── Planning ──
    "plan", "estimate", "timeline", "roadmap", "milestone",
    "sprint", "backlog", "grooming", "retrospective",
    "capacity", "work breakdown", "decomposition",

    # ── Algorithm / Math ──
    "algorithm", "complexity", "big o", "recurrence", "proof",
    "theorem", "lemma", "hypothesis", "deduce", "induction",
    "logic", "reasoning", "rationale", "soundness", "correctness",

    # ── Optimization / Performance Analysis ──
    "optimization", "analyze performance", "bottleneck analysis",
    "root cause", "causality", "correlation", "dependency",
    "graph", "topology", "hierarchy",

    # ── Scalability / Reliability ──
    "scalability", "reliability", "availability", "consistency",
    "partition", "cap theorem", "eventual consistency",
    "strong consistency", "quorum", "slo", "sli", "budget",
    "error budget", "chaos engineering",

    # ── Technical investigations ──
    "investigate", "probe", "diagnose", "debug analysis",
    "postmortem", "incident analysis",

    # ── Research (reasoning-oriented) ──
    "survey", "literature", "paper", "blog post",
    "technique", "methodology", "framework comparison",
    # ── Deep architectural analysis ──
    "architecture review", "design review of existing",
    "evaluate existing", "assess current",
}


KNOWLEDGE_KEYS = {
    # ── Information Retrieval ──
    "search", "find", "lookup", "retrieve", "fetch",
    "query", "what is", "who is", "where is", "when",
    "find out", "tell me", "i need to know",

    # ── Summarization ──
    "summarize", "summarise", "brief", "digest", "overview",
    "tl;dr", "abstract", "summary",

    # ── Reporting ──
    "report", "status", "update", "news", "current",
    "latest", "recent", "today", "yesterday", "week",
    "month", "quarter", "annual", "trend",

    # ── Documentation ──
    "documentation", "docs", "readme", "read the docs",
    "wiki", "confluence", "notion", "guide", "tutorial",
    "how to", "walkthrough", "example", "reference",
    "manual", "handbook", "specification",

    # ── Learning / Education ──
    "learn", "study", "understand", "comprehend", "grasp",
    "explain", "describe", "elaborate", "clarify",
    "difference between", "comparison", "versus", "vs",
    "explore", "exploration",

    # ── Definition / Terminology ──
    "define", "meaning", "definition", "what does",
    "term", "vocabulary", "glossary", "dictionary",

    # ── List / Enumeration ──
    "list", "enumerate", "catalog", "index", "directory",
    "show me", "give me", "top 10", "best", "rankings",
    "popular", "trending",

    # ── Monitoring / Observability (read-only) ──
    "check", "verify", "inspect", "monitor",
    "dashboard", "metric", "alert", "log", "grafana",
    "prometheus", "datadog", "splunk", "new relic",

    # ── Prices / Data ──
    "price", "cost", "pricing", "pricing table",
    "quota", "limit", "usage", "statistics",
}

# =====================================================================
# File extension patterns (unchanged)
# =====================================================================

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".h",
    ".rb", ".sh", ".yaml", ".yml", ".json", ".toml", ".md",
    ".cue", ".sql", ".css", ".scss", ".html", ".tsx", ".jsx",
    ".kt", ".swift", ".c", ".hpp", ".hxx", ".zig", ".r",
    ".lua", ".ex", ".exs", ".elm", ".clj", ".cljs",
    ".gradle", ".tf", ".hcl", ".nix", ".bzl",
}


# ── Helpers ──


def _count_keyword_matches(text: str, keywords: set) -> int:
    """Count how many unique keywords match in the text."""
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count


def _has_code_file_refs(text: str) -> int:
    """Count file paths with code extensions in text."""
    if not text:
        return 0
    pattern = r'(?:[\w./\\-]+(?:' + '|'.join(re.escape(e) for e in CODE_EXTENSIONS) + r'))'
    matches = re.findall(pattern, text)
    count = 0
    for m in matches:
        parts = m.split("/")
        filename = parts[-1] if parts else m
        has_path_sep = "/" in m or "\\" in m
        has_ext = any(m.endswith(ext) for ext in CODE_EXTENSIONS)
        if has_path_sep or has_ext:
            count += 1
    return count


# =====================================================================
# LAYER 1: Keyword classifier with confidence
# =====================================================================

# Minimum keyword matches to consider a dimension "active"
_MIN_MATCHES_PER_DIM = 2
# Base per-keyword weight
_KW_WEIGHT = 0.10
# Toolset signal weights
_TOOL_SIGNALS = {
    # Exact set matches
    (("terminal", "file"),): ("coding", 0.25),
    (("terminal",),): ("coding", 0.15),
    # Strong knowledge signals
    (("web_search",),): ("knowledge", 0.35),
    (("browser", "web_search"),): ("knowledge", 0.40),
    (("browser",),): ("knowledge", 0.20),
    # Reasoning signals
    (("delegation",),): ("reasoning", 0.25),
    (("todo", "kanban"),): ("reasoning", 0.15),
    # Mixed context
    (("terminal", "web_search"),): ("coding", 0.10),
    (("terminal", "web_search"),): ("knowledge", 0.15),
}


def _estimate_confidence(vec: Dict[str, float]) -> float:
    """Estimate how confident we are in the classification.

    Uses relative spread between dimensions (not absolute values).
    Returns 0.0 to 1.0. High = clear signal, Low = ambiguous.
    """
    values = list(vec.values())
    max_val = max(values)
    if max_val <= 0.01:
        return 0.0

    # Primary dimension's share of total signal
    total = sum(values)
    primary_share = max_val / total if total > 0 else 0

    # How many dimensions have significant signal (≥20% of max)
    threshold = max_val * 0.20
    significant = sum(1 for v in values if v >= threshold)
    clarity_penalty = 1.0 - (significant - 1) * 0.25
    clarity_penalty = max(clarity_penalty, 0.25)

    # How much does #1 beat #2?
    sorted_vals = sorted(values, reverse=True)
    if len(sorted_vals) > 1 and sorted_vals[1] > 0:
        margin = (sorted_vals[0] - sorted_vals[1]) / sorted_vals[0]
    else:
        margin = 1.0

    # Combine: primary_share * clarity_penalty + margin bonus
    confidence = primary_share * clarity_penalty + margin * 0.2
    confidence = min(confidence, 1.0)

    return round(confidence, 2)


MIN_CONFIDENCE = 0.55  # Below this, invoke LLM booster


def _is_ambiguous(vec: Dict[str, float]) -> bool:
    """Check if the keyword result is ambiguous enough to warrant LLM boost.

    Returns True when the #2 dimension has ≥30% signal strength of #1,
    indicating possible confusion between categories.
    """
    values = sorted(vec.values(), reverse=True)
    if len(values) < 2 or values[0] <= 0.01:
        return False
    return values[1] >= values[0] * 0.30


def _keyword_classify(
    goal: str,
    context: str,
    toolsets: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float]:
    """Layer 1: Expanded keyword classification with confidence.

    Returns (vector, confidence_score).
    """
    vec = {"coding": 0.0, "reasoning": 0.0, "knowledge": 0.0}
    goal_lower = (goal or "").lower()
    context_lower = (context or "").lower()
    combined = goal_lower + " " + context_lower
    tools = set(t.lower() for t in (toolsets or []))

    # ── Keyword scoring ──
    coding_matches = _count_keyword_matches(combined, CODING_KEYS)
    reasoning_matches = _count_keyword_matches(combined, REASONING_KEYS)
    knowledge_matches = _count_keyword_matches(combined, KNOWLEDGE_KEYS)

    vec["coding"] += coding_matches * _KW_WEIGHT
    vec["reasoning"] += reasoning_matches * _KW_WEIGHT
    vec["knowledge"] += knowledge_matches * _KW_WEIGHT

    # ── Toolset signals ──
    tools_str = "_".join(sorted(tools))
    # terminal alone → coding
    if tools == {"terminal"} or tools == {"terminal", "file"}:
        vec["coding"] += 0.2
    # web_search without terminal → strong knowledge
    if "web_search" in tools and "terminal" not in tools:
        vec["knowledge"] += 0.3
    # browser without terminal or web → reading = knowledge
    if "browser" in tools and "terminal" not in tools and "web_search" not in tools:
        vec["knowledge"] += 0.2
    # delegation → orchestration = reasoning
    if "delegation" in tools:
        vec["reasoning"] += 0.25

    # ── File refs → coding boost ──
    file_count = _has_code_file_refs(context)
    if file_count > 0:
        vec["coding"] += min(file_count * 0.05, 0.3)

    confidence = _estimate_confidence(vec)

    return vec, confidence


# =====================================================================
# LAYER 2: LLM booster (ministral-3:14b)
# =====================================================================

# LLM booster is only called when keyword confidence < MIN_CONFIDENCE
# Uses a cheap model: ministral-3:14b
# Expected cost: ~20 tokens/ticket
# Only invoked for ~30% of tickets

_LLM_CLASSIFICATION_PROMPT = """Respond with exactly ONE word: coding, reasoning, or knowledge.

coding = implement, fix bugs, write tests, refactor, deploy, optimize, write code
reasoning = design architecture, analyze trade-offs, plan, evaluate, make decisions
knowledge = research, find information, learn, document, summarize, check status

Goal: {goal}

ONE WORD ONLY: coding, reasoning, or knowledge"""

# Provider + model for the LLM booster
_LLM_BOOSTER_PROVIDER = "ollama-cloud"
_LLM_BOOSTER_MODEL = "ministral-3:14b-cloud"


def _llm_classify(
    goal: str,
    context: str,
    toolsets: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Layer 2: LLM-based classification for low-confidence cases.

    Calls ministral-3:14b via auxiliary_client.call_llm.
    Returns a 3-dim vector with the LLM's classification.
    Falls back to heuristic pattern matching if LLM call fails.
    """
    try:
        from agent.auxiliary_client import call_llm

        prompt = _LLM_CLASSIFICATION_PROMPT.format(
            goal=goal or "(empty)",
        )
        response = call_llm(
            provider=_LLM_BOOSTER_PROVIDER,
            model=_LLM_BOOSTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )
        label = response.choices[0].message.content.strip().lower()
        # Strip markdown formatting and extract first valid label
        import re
        m = re.search(r"\b(coding|reasoning|knowledge)\b", label)
        label = m.group(1) if m else label.strip("*_`.").strip()

        vec = {"coding": 0.0, "reasoning": 0.0, "knowledge": 0.0}
        if label == "coding":
            vec["coding"] = 1.0
        elif label == "reasoning":
            vec["reasoning"] = 1.0
        elif label == "knowledge":
            vec["knowledge"] = 1.0
        else:
            # Unexpected response — fall through to heuristic
            logger.warning("LLM booster returned unexpected label: %s", label)
        return vec

    except Exception as exc:
        logger.debug("LLM booster call failed: %s — using heuristic fallback", exc)

    # ── Heuristic fallback (no LLM available) ──
    vec = {"coding": 0.0, "reasoning": 0.0, "knowledge": 0.0}
    goal_lower = (goal or "").lower()
    context_lower = (context or "").lower()
    combined = goal_lower + " " + context_lower

    # Questions → knowledge
    if "?" in goal or "?" in context:
        vec["knowledge"] += 0.3

    # Imperative verbs mapping
    imperative_coding = {"fix", "implement", "build", "add", "remove", "update", "install"}
    imperative_knowledge = {"find", "search", "check", "verify", "read", "learn", "study"}
    imperative_reasoning = {"design", "plan", "compare", "evaluate", "investigate", "explore"}

    words = set(combined.split())

    for w in words & imperative_coding:
        vec["coding"] += 0.12
    for w in words & imperative_knowledge:
        vec["knowledge"] += 0.12
    for w in words & imperative_reasoning:
        vec["reasoning"] += 0.12

    # ── Bigram patterns (fallback heuristic) ──
    bigram_coding = {"pull request", "merge request", "code review", "unit test",
                     "integration test", "memory leak", "compile error",
                     "type hints", "test failures", "test coverage",
                     "fix test", "pipeline", "data in test",
                     "classifier scoring", "model mapping"}
    bigram_reasoning = {"should we", "how should", "what if", "what approach",
                        "best way", "trade off", "decision record",
                        "compare approaches", "evaluate options",
                        "design proposal", "architecture review",
                        "technical design", "design spike",
                        "design doc", "feasibility"}
    bigram_knowledge = {"how does it work", "what is a",
                        "where can i find", "documentation for",
                        "tell me about", "read about", "look up",
                        "understand how", "check if it exists"}

    for cat, vec_key, bk in [("coding", "coding", bigram_coding),
                              ("reasoning", "reasoning", bigram_reasoning),
                              ("knowledge", "knowledge", bigram_knowledge)]:
        for bg in bk:
            if bg in combined:
                vec[vec_key] += 0.15

    return vec


# =====================================================================
# PUBLIC API
# =====================================================================


def estimate_task_vector(
    goal: str = "",
    context: str = "",
    toolsets: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Estimate task requirement weights from goal text, context, and toolsets.

    Returns dict with keys: coding, reasoning, knowledge
    Each value is 0-1 (higher = stronger signal for that dimension).

    Uses 3-layer hybrid:
      Layer 1 — Expanded keyword classifier (fast path, ~85%)
      Layer 2 — LLM booster (only when confidence < MIN_CONFIDENCE)
      Layer 3 — Toolset + file-ref overlay (always on)
    """
    # Layer 1 + 3 combined
    vec, confidence = _keyword_classify(goal, context, toolsets)
    logger.debug(
        "Keyword: coding=%.2f reasoning=%.2f knowledge=%.2f conf=%.2f",
        vec["coding"], vec["reasoning"], vec["knowledge"], confidence,
    )

    # Layer 2: Boost with LLM only when keyword signals are genuinely low
    need_boost = confidence < MIN_CONFIDENCE and max(vec.values()) > 0.01
    if need_boost:
        llm_vec = _llm_classify(goal, context, toolsets)
        logger.debug(
            "LLM booster: coding=%.2f reasoning=%.2f knowledge=%.2f",
            llm_vec["coding"], llm_vec["reasoning"], llm_vec["knowledge"],
        )
        # LLM is a tiebreaker: boost keyword-agreeing cases, gently nudge disagreeing
        keyword_primary = max(vec.items(), key=lambda x: x[1])[0]
        if llm_vec.get(keyword_primary, 0) > 0:
            # LLM agrees with keyword result — reinforce (+0.1)
            vec[keyword_primary] += 0.10
        else:
            # LLM disagrees — nudge if keyword signal is weak (gap < 50%)
            values = sorted(vec.values(), reverse=True)
            if len(values) >= 2 and values[0] > 0 and (values[0] - values[1]) / max(values[0], 0.01) < 0.50:
                llm_primary = max(llm_vec.items(), key=lambda x: x[1])[0]
                if llm_vec[llm_primary] > 0:
                    vec[llm_primary] += 0.08  # Meaningful nudge towards LLM choice

    # ── Normalize to [0, 1] maintaining relative strengths ──
    max_val = max(vec.values()) or 1.0
    if max_val > 0:
        for k in vec:
            vec[k] = round(vec[k] / max_val, 2)

    return vec


def match_model(
    task_vector: Dict[str, float],
    available_models: List[Dict[str, Any]],
    cost_weight: float = 0.3,
) -> Optional[str]:
    """Select the best model for a task using weighted cosine + cost bias.

    Args:
        task_vector: {coding: 0.8, reasoning: 0.3, knowledge: 0.1}
        available_models: list of dicts with keys:
            - id: str (model identifier)
            - capability_vector: dict {coding, reasoning, knowledge, agentic} (0-1)
            - input_price: float ($ per 1M tokens)
        cost_weight: how much to favor cheaper models (0-1, default 0.3)

    Returns:
        model_id: str or None if no models available
    """
    if not available_models:
        return None

    # Normalize task vector to unit vector
    task_mag = sum(v ** 2 for v in task_vector.values()) ** 0.5
    task_unit = {k: v / task_mag for k, v in task_vector.items()} if task_mag > 0 else task_vector.copy()

    # If all task values are 0, default to cheapest model
    if all(v == 0 for v in task_vector.values()):
        return min(available_models, key=lambda m: m.get("input_price", 0) or 0)["id"]

    best_id = None
    best_combined = -float('inf')

    for model in available_models:
        cap = model.get("capability_vector", {})

        # Weighted similarity: task_weight * capability_score for each dimension
        similarity = sum(
            task_unit.get(k, 0) * cap.get(k, 0)
            for k in task_unit
        )

        # Cost factor: cheaper is better, normalize
        price = model.get("input_price", 0) or 0
        cost_factor = 1.0 / (price + 0.1)  # Avoid division by zero
        cost_factor = min(cost_factor, 10.0) / 10.0  # Normalize to [0, 1]

        # Combined: capability match + cost efficiency
        cap_weight = 1.0 - cost_weight
        combined = cap_weight * similarity + cost_weight * cost_factor

        logger.debug(
            "Model %s: sim=%.3f cost=%.3f combined=%.3f",
            model["id"], similarity, cost_factor, combined,
        )

        if combined > best_combined:
            best_combined = combined
            best_id = model["id"]

    return best_id


if __name__ == "__main__":
    # Self-test
    print("Testing expanded task classifier...\n")

    test_cases = [
        # Pure coding
        ("Fix test failures in test_delegate.py", "", ["terminal"],
         {"coding_min": 0.5}),
        ("Merge pull request #42: fix null pointer in handler", "", ["terminal", "file"],
         {"coding_min": 0.5}),
        # Pure reasoning
        ("Design architecture for orchestrator module", "", [],
         {"reasoning_min": 0.5}),
        ("Evaluate trade-offs between Raft and Paxos for leader election", "", [],
         {"reasoning_min": 0.5, "coding_max": 0.4}),
        # Pure knowledge
        ("Find current USD to JPY exchange rate", "", ["web_search"],
         {"knowledge_min": 0.5}),
        ("What is the difference between GraphQL and REST?", "", [],
         {"knowledge_min": 0.5}),
        ("Summarize the latest AWS Lambda pricing update", "", ["web_search"],
         {"knowledge_min": 0.5}),
        # Cross-category: SDLC tickets
        ("Research Docker networking for microservices", "", [],
         {"knowledge_min": 0.3}),
        ("Write documentation for v2 API endpoints", "", ["terminal", "file"],
         {"coding_min": 0.2, "knowledge_min": 0.1}),
        # Empty
        ("", "", None,
         {"coding_max": 0.1, "reasoning_max": 0.1, "knowledge_max": 0.1}),
    ]

    all_passed = True
    for goal, ctx, tls, expected in test_cases:
        vec = estimate_task_vector(goal=goal or "", context=ctx or "", toolsets=tls)
        label = "CODING" if max(vec.items(), key=lambda x: x[1])[0] == "coding" else (
            "REASONING" if max(vec.items(), key=lambda x: x[1])[0] == "reasoning" else "KNOWLEDGE")
        print(f"  [{label:>10}] c={vec['coding']:.2f} r={vec['reasoning']:.2f} k={vec['knowledge']:.2f}  |  {(goal or '(empty)'):60s}")

        # Verify bounds
        for k, v in expected.items():
            dim = k.split("_")[0]
            check_type = k.split("_")[1]
            if check_type == "min":
                ok = vec[dim] >= v
            elif check_type == "max":
                ok = vec[dim] <= v
            else:
                continue
            if not ok:
                all_passed = False
                print(f"    ❌ FAILED: {dim} {check_type} {v} (got {vec[dim]})")

    print()
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests FAILED!")
        import sys
        sys.exit(1)
