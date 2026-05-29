"""Fallback capability estimation for unlisted models.

When a model is not in the published benchmark registry, estimate its capability
score by:
1. Model size interpolation (if parameters known)
2. Peer matching (find closest model by name/family)
3. Tier default (fallback to reasoning_capability mapping)
"""

import re
from typing import Optional, Tuple
from agent.benchmark_registry import (
    BENCHMARK_REGISTRY,
    MODEL_ALIASES,
    calculate_capability_score,
    estimate_latency_tier,
)


# Model family patterns for peer matching
MODEL_FAMILIES = {
    "claude": ["claude-3-5-sonnet", "claude-3-5-haiku"],
    "gpt": ["gpt-4o", "gpt-4o-mini"],
    "llama": ["llama-3-1-405b", "llama-3-1-70b", "llama-3-1-8b"],
    "mistral": ["mistral-large-2", "mistral-7b"],
    "gemma": ["gemma-7b"],
    "deepseek": ["deepseek-v3"],
    "kimi": ["kimi-k2-6"],
    "qwen": ["qwen-2-5-72b"],
    "glm": ["glm-5-1"],
}

# Model size tier boundaries (parameters in billions)
SIZE_TIERS = [
    (0, 8, 0.55),           # <8B: lightweight (0.55 capability)
    (8, 30, 0.70),          # 8-30B: mid-tier (0.70)
    (30, 100, 0.80),        # 30-100B: advanced (0.80)
    (100, float('inf'), 0.85),  # 100B+: frontier (0.85)
]


def extract_parameters(model_id: str) -> Optional[float]:
    """Extract model size in billions from model ID.
    
    Examples:
        "llama-3-1-70b" → 70.0
        "mistral-7b" → 7.0
        "qwen3.5:397b" → 397.0
    
    Args:
        model_id: Model identifier string
    
    Returns:
        Parameters in billions, or None if not found
    """
    # Match patterns like "70b", "7b", "397b"
    match = re.search(r'(\d+(?:\.\d+)?)\s*b(?:illions?)?', model_id, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def find_closest_peer(model_id: str) -> Optional[str]:
    """Find closest model in registry by family/name similarity.
    
    Args:
        model_id: Model identifier to find peer for
    
    Returns:
        Canonical model ID of closest peer, or None
    """
    model_lower = model_id.lower()
    
    # Check each family
    for family_name, family_models in MODEL_FAMILIES.items():
        if family_name in model_lower:
            # Prefer most capable model in family
            candidates = [
                mid for mid in family_models
                if mid in BENCHMARK_REGISTRY
            ]
            if candidates:
                # Sort by capability_score descending
                candidates.sort(
                    key=lambda mid: calculate_capability_score(
                        BENCHMARK_REGISTRY[mid]
                    ),
                    reverse=True
                )
                return candidates[0]
    
    # No family match; return highest-capability frontier model
    frontier = [
        mid for mid, entry in BENCHMARK_REGISTRY.items()
        if calculate_capability_score(entry) >= 0.85
    ]
    if frontier:
        return frontier[0]  # arbitrary frontier model
    
    return None


def estimate_from_size(parameters_billions: float) -> float:
    """Estimate capability score from model size alone.
    
    Args:
        parameters_billions: Model size
    
    Returns:
        Estimated capability score (0-1.0)
    """
    for min_b, max_b, score in SIZE_TIERS:
        if min_b <= parameters_billions < max_b:
            return score
    return 0.55  # fallback


def estimate_capability_score(
    model_id: str,
    reasoning_capability: Optional[str] = None,
) -> Tuple[float, str, str]:
    """Estimate capability score for unlisted model.
    
    Priority:
    1. Exact match in registry (already checked by caller)
    2. Size-based estimation (if parameters extractable)
    3. Peer matching (similar family)
    4. Reasoning capability mapping (fallback)
    
    Args:
        model_id: Model identifier
        reasoning_capability: Optional reasoning tier ("low", "medium", "high")
    
    Returns:
        Tuple: (capability_score, source, note)
    """
    
    # Try size extraction
    params_b = extract_parameters(model_id)
    if params_b is not None:
        score = estimate_from_size(params_b)
        return score, "size-tier", f"Estimated from {params_b}B parameters"
    
    # Try peer matching
    peer = find_closest_peer(model_id)
    if peer and peer in BENCHMARK_REGISTRY:
        entry = BENCHMARK_REGISTRY[peer]
        score = calculate_capability_score(entry)
        return score, "peer-match", f"Estimated via peer {peer}"
    
    # Try reasoning capability mapping
    if reasoning_capability:
        effort_map = {"low": 0.55, "medium": 0.75, "high": 0.85}
        score = effort_map.get(reasoning_capability, 0.70)
        return score, "reasoning-tier", f"Estimated from reasoning={reasoning_capability}"
    
    # Ultimate fallback: average frontier
    return 0.75, "default", "Using default mid-capability estimate"


if __name__ == "__main__":
    # Self-test
    print("Testing Model Fallback Estimator...\n")
    
    test_cases = [
        ("llama-3-70b", "high"),
        ("mistral-8b", "medium"),
        ("unknown-model-5b", "low"),
        ("claude-4", None),
        ("gpt-5-large", "high"),
    ]
    
    for model_id, reasoning in test_cases:
        score, source, note = estimate_capability_score(model_id, reasoning)
        print(f"✅ {model_id:25s} | score: {score:.2f} ({source:15s}) | {note}")
    
    print("\n--- Parameter Extraction ---")
    extract_tests = [
        "llama-3-1-70b",
        "mistral-7b",
        "qwen3.5:397b",
        "claude-3-5-sonnet",
        "unknown-model",
    ]
    
    for model_id in extract_tests:
        params = extract_parameters(model_id)
        if params:
            tier = estimate_from_size(params)
            print(f"✅ {model_id:25s} | {params:7.1f}B → capability {tier:.2f}")
        else:
            print(f"❌ {model_id:25s} | No parameters found")
