"""
Model Mention Routing — extract explicit model/tier overrides from task text.

Convention: {model:<name_or_tier>} in goal or context text.

Priority chain:
  1. delegate_task(model="...")  — explicit tool arg (strongest)
  2. {model:<name_or_tier>}      — tag in goal or context  (← this module)
  3. config.yaml delegation.model
  4. Automatic match_model()

Usage:
    from agent.model_mention import extract_model_mention
    model_id = extract_model_mention(goal, context)
    if model_id:
        # Short-circuit: use model_id directly, skip task matcher
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Complete list of known model IDs ──────────────────────────────────
KNOWN_MODEL_IDS = {
    # T4 — Frontier (high reasoning / best coding)
    "deepseek-v4-pro:cloud",
    "deepseek-v4-flash:cloud",
    "kimi-k2.6:cloud",
    "glm-5.1:cloud",
    "deepseek-v3.2:cloud",
    "gemini-3-flash-preview:cloud",
    # T2 — Balanced / Mid
    "gemma4:31b-cloud",
    # T1 — Cheap / Fast
    "ministral-3:14b-cloud",
    "devstral-small-2:24b-cloud",
    "nemotron-3-nano:30b-cloud",
    "gemma3:27b-cloud",
}

# ── Alias → exact model ID mapping ───────────────────────────────────
ALIASES = {
    "deepseek-pro": "deepseek-v4-pro:cloud",
    "deepseek_v4_pro": "deepseek-v4-pro:cloud",
    "deepseekv4pro": "deepseek-v4-pro:cloud",
    "ds-pro": "deepseek-v4-pro:cloud",
    "dspro": "deepseek-v4-pro:cloud",
    "deepseek flash": "deepseek-v4-flash:cloud",
    "deepseek_flash": "deepseek-v4-flash:cloud",
    "deepseekv4flash": "deepseek-v4-flash:cloud",
    "ds-flash": "deepseek-v4-flash:cloud",
    "dsflash": "deepseek-v4-flash:cloud",
    "flash": "deepseek-v4-flash:cloud",
    "kimi": "kimi-k2.6:cloud",
    "kimi-k2.6": "kimi-k2.6:cloud",
    "kimi_k2_6": "kimi-k2.6:cloud",
    "kimi_k26": "kimi-k2.6:cloud",
    "glm": "glm-5.1:cloud",
    "glm-5.1": "glm-5.1:cloud",
    "glm_5_1": "glm-5.1:cloud",
    "deepseek-v3.2": "deepseek-v3.2:cloud",
    "deepseek_v3_2": "deepseek-v3.2:cloud",
    "ds-v3.2": "deepseek-v3.2:cloud",
    "dsv32": "deepseek-v3.2:cloud",
    "gemini flash": "gemini-3-flash-preview:cloud",
    "gemini_flash": "gemini-3-flash-preview:cloud",
    "gemma4": "gemma4:31b-cloud",
    "gemma-4": "gemma4:31b-cloud",
    "gemma_4": "gemma4:31b-cloud",
    "ministral": "ministral-3:14b-cloud",
    "ministral-3": "ministral-3:14b-cloud",
    "ministral_3": "ministral-3:14b-cloud",
    "devstral": "devstral-small-2:24b-cloud",
    "devstral-small": "devstral-small-2:24b-cloud",
    "devstral_small": "devstral-small-2:24b-cloud",
    "devstral_small_2": "devstral-small-2:24b-cloud",
    "nemotron": "nemotron-3-nano:30b-cloud",
    "nemotron-3": "nemotron-3-nano:30b-cloud",
    "nemotron_3": "nemotron-3-nano:30b-cloud",
    "gemma3": "gemma3:27b-cloud",
    "gemma-3": "gemma3:27b-cloud",
}

# ── Tier → model resolution ──────────────────────────────────────────
TIERS = {
    # Tier names
    "tier-high": "deepseek-v4-pro:cloud",
    "tier-frontier": "deepseek-v4-pro:cloud",
    "tier-top": "deepseek-v4-pro:cloud",
    "tier-best": "deepseek-v4-pro:cloud",
    "tier-reason": "deepseek-v4-pro:cloud",        # best reasoning model
    "tier-deep": "deepseek-v4-pro:cloud",
    "tier-mid": "deepseek-v4-flash:cloud",
    "tier-balanced": "deepseek-v4-flash:cloud",
    "tier-medium": "deepseek-v4-flash:cloud",
    "tier-code": "deepseek-v4-flash:cloud",         # best coding model
    "tier-cheap": "devstral-small-2:24b-cloud",
    "tier-budget": "devstral-small-2:24b-cloud",
    "tier-fast": "devstral-small-2:24b-cloud",
    "tier-light": "devstral-small-2:24b-cloud",
    "tier-knowledge": "ministral-3:14b-cloud",      # best knowledge model
    # Human-friendly tier names (no prefix)
    "high reasoning": "deepseek-v4-pro:cloud",
    "high_reasoning": "deepseek-v4-pro:cloud",
    "frontier model": "deepseek-v4-pro:cloud",
    "frontier_model": "deepseek-v4-pro:cloud",
    "best model": "deepseek-v4-pro:cloud",
    "best_model": "deepseek-v4-pro:cloud",
    "reasoning model": "deepseek-v4-pro:cloud",
    "reasoning_model": "deepseek-v4-pro:cloud",
    "balanced model": "deepseek-v4-flash:cloud",
    "balanced_model": "deepseek-v4-flash:cloud",
    "cheapest": "devstral-small-2:24b-cloud",
    "budget": "devstral-small-2:24b-cloud",
    "fastest": "devstral-small-2:24b-cloud",
    "coding model": "deepseek-v4-flash:cloud",
    "coding_model": "deepseek-v4-flash:cloud",
}

# Pre-compute set for quick exact-model-ID checking
_EXACT_MODEL_LOWER = {m.lower() for m in KNOWN_MODEL_IDS}

# Pre-compile regex
_TAG_PATTERN = re.compile(r'\{model:([^}]+)\}', re.IGNORECASE)


def extract_model_mention(
    goal: str = "",
    context: str = "",
) -> Optional[str]:
    """Scan goal and context for {model:<name_or_tier>} tags.

    Returns the resolved model ID (e.g. 'deepseek-v4-pro:cloud') or None
    if no valid mention is found. Scans goal first, then context.

    Resolution order:
      1. Exact match - tag value matches a known model ID
      2. Alias match - tag value matches an alias (e.g. 'flash' → deepseek-v4-flash:cloud)
      3. Tier match - tag value matches a tier key (e.g. 'tier-high' → deepseek-v4-pro:cloud)
    """
    text = (goal or "") + " " + (context or "")

    # Step 1: Find {model:...} tags
    match = _TAG_PATTERN.search(text)
    if not match:
        return None

    tag_value = match.group(1).strip().lower()

    # Step 2: Exact model ID match
    if tag_value in _EXACT_MODEL_LOWER:
        logger.debug("Model mention: exact match '%s'", tag_value)
        return tag_value

    # Step 3: Alias match (full key, then partial key)
    if tag_value in ALIASES:
        logger.debug("Model mention: alias '%s' → '%s'", tag_value, ALIASES[tag_value])
        return ALIASES[tag_value]

    # Partial alias match: tag_value contains one of our alias keys
    # e.g. "use deepseek pro" vs "deepseek-pro"
    for alias_key, model_id in ALIASES.items():
        if alias_key in tag_value or tag_value in alias_key:
            logger.debug("Model mention: partial alias '%s' → '%s'", tag_value, model_id)
            return model_id

    # Step 4: Tier match
    if tag_value in TIERS:
        logger.debug("Model mention: tier '%s' → '%s'", tag_value, TIERS[tag_value])
        return TIERS[tag_value]

    # Partial tier match
    for tier_key, model_id in TIERS.items():
        if tier_key in tag_value or tag_value in tier_key:
            logger.debug("Model mention: partial tier '%s' → '%s'", tag_value, model_id)
            return model_id

    logger.debug("Model mention: unrecognized '%s' — ignoring", tag_value)
    return None
