"""Tests for model mention routing — extract_model_mention()."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.model_mention import extract_model_mention


# ── Null / Empty ─────────────────────────────────────────────────────

def test_empty_goal_and_context():
    """No tags → None."""
    assert extract_model_mention("", "") is None


def test_no_tag_in_goal():
    """Plain text goal without {model:...} → None."""
    assert extract_model_mention("Fix test failures in test_delegate.py", "") is None


def test_no_tag_in_context():
    """Plain text context without {model:...} → None."""
    assert extract_model_mention("Investigate root cause", "path: test_x.py tools: terminal") is None


# ── Exact Model ID ───────────────────────────────────────────────────

def test_exact_model_id_in_goal():
    """Exact model ID in goal tag → that model."""
    result = extract_model_mention("{model:deepseek-v4-pro:cloud} Design the architecture", "")
    assert result == "deepseek-v4-pro:cloud"


def test_exact_model_id_in_context():
    """Exact model ID in context tag → that model."""
    result = extract_model_mention("", "The task uses {model:deepseek-v4-flash:cloud} for speed")
    assert result == "deepseek-v4-flash:cloud"


def test_exact_model_id_prefers_goal():
    """Goal tag takes priority over context tag."""
    result = extract_model_mention(
        goal="{model:deepseek-v4-pro:cloud} Do the thing",
        context="{model:gemma4:31b-cloud} In background",
    )
    assert result == "deepseek-v4-pro:cloud"


# ── Aliases ──────────────────────────────────────────────────────────

def test_alias_deepseek_pro():
    """{model:deepseek-pro} → deepseek-v4-pro:cloud."""
    assert extract_model_mention("{model:deepseek-pro} Run analysis", "") == "deepseek-v4-pro:cloud"


def test_alias_flash():
    """{model:flash} → deepseek-v4-flash:cloud."""
    assert extract_model_mention("{model:flash} Fix bug", "") == "deepseek-v4-flash:cloud"


def test_alias_ministral():
    """{model:ministral} → ministral-3:14b-cloud."""
    assert extract_model_mention("{model:ministral} Look up info", "") == "ministral-3:14b-cloud"


def test_alias_kimi():
    """{model:kimi} → kimi-k2.6:cloud."""
    assert extract_model_mention("{model:kimi} Deep research", "") == "kimi-k2.6:cloud"


def test_alias_gemma4():
    """{model:gemma4} → gemma4:31b-cloud."""
    assert extract_model_mention("{model:gemma4} Balanced task", "") == "gemma4:31b-cloud"


def test_alias_devstral():
    """{model:devstral} → devstral-small-2:24b-cloud."""
    assert extract_model_mention("{model:devstral} Cheap batch", "") == "devstral-small-2:24b-cloud"


def test_alias_case_insensitive():
    """Tag matching is case-insensitive."""
    assert extract_model_mention("{model:DEEPSEEK-PRO} Analysis", "") == "deepseek-v4-pro:cloud"
    assert extract_model_mention("{Model:Flash} Coding", "") == "deepseek-v4-flash:cloud"
    assert extract_model_mention("{MODEL:tier-HIGH} Planning", "") == "deepseek-v4-pro:cloud"


# ── Tiers ────────────────────────────────────────────────────────────

def test_tier_high():
    """{model:tier-high} → deepseek-v4-pro:cloud."""
    assert extract_model_mention("{model:tier-high} Design", "") == "deepseek-v4-pro:cloud"


def test_tier_mid():
    """{model:tier-mid} → deepseek-v4-flash:cloud."""
    assert extract_model_mention("{model:tier-mid} Coding", "") == "deepseek-v4-flash:cloud"


def test_tier_cheap():
    """{model:tier-cheap} → devstral-small-2:24b-cloud."""
    assert extract_model_mention("{model:tier-cheap} Lookup", "") == "devstral-small-2:24b-cloud"


def test_tier_code():
    """{model:tier-code} → deepseek-v4-flash:cloud (best coding model)."""
    assert extract_model_mention("{model:tier-code} Implement API", "") == "deepseek-v4-flash:cloud"


def test_tier_reason():
    """{model:tier-reason} → deepseek-v4-pro:cloud (best reasoning model)."""
    assert extract_model_mention("{model:tier-reason} Analyze trade-offs", "") == "deepseek-v4-pro:cloud"


def test_tier_budget():
    """{model:budget} → devstral-small-2:24b-cloud (human-friendly tier name)."""
    assert extract_model_mention("{model:budget} Simple query", "") == "devstral-small-2:24b-cloud"


def test_tier_high_reasoning():
    """{model:high reasoning} → deepseek-v4-pro:cloud (multi-word tier)."""
    assert extract_model_mention("{model:high reasoning} Deep analysis", "") == "deepseek-v4-pro:cloud"


# ── Edge Cases ───────────────────────────────────────────────────────

def test_unknown_model_name():
    """Unrecognized model name → None (no failure)."""
    assert extract_model_mention("{model:o4-mini} Some task", "") is None
    assert extract_model_mention("{model:gpt-5} Some task", "") is None


def test_malformed_bracket_no_colon():
    """{something:else} not model → ignored (no match)."""
    assert extract_model_mention("{priority:high} Design task", "") is None


def test_tag_embedded_in_text():
    """Tag can appear anywhere in goal string."""
    result = extract_model_mention(
        "Task 3.1: {model:tier-high} ConfidenceScorer Algorithm — CLASS B",
        "",
    )
    assert result == "deepseek-v4-pro:cloud"


def test_tag_with_extra_spaces():
    """Extra whitespace inside {model:...} is stripped."""
    result = extract_model_mention("{model:  deepseek-pro  } Analysis", "")
    assert result == "deepseek-v4-pro:cloud"


def test_first_tag_wins():
    """If multiple tags, the first one wins (regex find earliest)."""
    result = extract_model_mention(
        "{model:flash} {model:deepseek-pro} Two models mentioned",
        "",
    )
    assert result == "deepseek-v4-flash:cloud"  # flash comes first
