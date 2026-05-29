"""Model registry with caching and orchestration.

Provides unified interface for discovering and caching models across all providers.
Handles fallback strategies, TTL-based invalidation, and persistence to disk.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from agent.model_discovery import (
    ModelDescriptor,
    ModelDiscoveryProvider,
    create_discovery_provider,
)
from agent.benchmark_registry import (
    get_benchmark_entry,
    calculate_capability_score,
    score_to_reasoning_effort,
    estimate_latency_tier,
)
from agent.model_fallback_estimator import estimate_capability_score as estimate_capability

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cached models for a provider with TTL tracking."""
    
    def __init__(self, models: List[ModelDescriptor], ttl_hours: int = 24):
        self.models = models
        self.timestamp = datetime.utcnow()
        self.ttl_hours = ttl_hours
    
    def is_stale(self) -> bool:
        """Check if cache entry has expired."""
        expiry = self.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.utcnow() > expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for disk persistence."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ttl_hours": self.ttl_hours,
            "models": [m.to_dict() for m in self.models],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Deserialize from disk."""
        entry = cls([], ttl_hours=data.get("ttl_hours", 24))
        entry.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
        entry.models = [ModelDescriptor(**m) for m in data.get("models", [])]
        return entry


class ModelRegistry:
    """Unified model discovery registry with caching.
    
    - Discovers models for each provider (API or config-backed)
    - Caches results to disk with TTL
    - Provides unified interface for querying available models
    - Handles fallback to config when API is unavailable
    - Augments models with published benchmark scores
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize registry.
        
        Args:
            cache_dir: Path to cache directory (default: ~/.hermes/model_cache)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".hermes" / "model_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache: provider -> CacheEntry
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        logger.info(f"ModelRegistry initialized with cache dir: {self.cache_dir}")
    
    def _cache_file(self, provider_name: str) -> Path:
        """Get cache file path for a provider."""
        return self.cache_dir / f"models_{provider_name}.json"
    
    def _augment_with_benchmarks(self, model: ModelDescriptor) -> ModelDescriptor:
        """Augment model descriptor with benchmark data.
        
        Priority:
        1. Published benchmark (exact match)
        2. Fallback estimation (size-tier → peer-match → reasoning-tier)
        
        Args:
            model: ModelDescriptor from discovery
        
        Returns:
            Same descriptor with benchmark fields populated
        """
        entry = get_benchmark_entry(model.id)
        
        if entry:
            # Published benchmarks found
            model.capability_score = calculate_capability_score(entry)
            model.mmlu_pct = entry.mmlu_pct
            model.humaneval_pct = entry.humaneval_pct
            model.math_pct = entry.math_pct
            model.gpqa_pct = entry.gpqa_pct
            model.latency_tier = estimate_latency_tier(entry.parameters_billions)
            model.parameters_billions = entry.parameters_billions
            model.benchmark_source = "published"
            model.benchmark_date = entry.published_date
            
            logger.debug(f"✅ {model.id}: augmented with published benchmarks (score={model.capability_score})")
        else:
            # No published benchmarks; use fallback estimation
            score, source, note = estimate_capability(model.id, model.reasoning_capability)
            model.capability_score = score
            model.benchmark_source = source
            
            # Estimate latency tier from capability score
            if score >= 0.85:
                model.latency_tier = "slow"
            elif score >= 0.75:
                model.latency_tier = "balanced"
            else:
                model.latency_tier = "fast"
            
            logger.debug(f"⚠️  {model.id}: fallback estimated score {score:.2f} ({source}) — {note}")
        
        return model
    
    async def discover_provider(
        self,
        provider_name: str,
        config: Dict[str, Any],
        auth: Dict[str, Any],
    ) -> List[ModelDescriptor]:
        """Auto-discover models for a provider.
        
        Args:
            provider_name: Name of provider
            config: Provider config dict
            auth: Auth credentials dict
        
        Returns:
            List of discovered models (with benchmarks augmented)
        """
        logger.info(f"Discovering models for provider: {provider_name}")
        
        try:
            # Create appropriate discovery provider
            discoverer = create_discovery_provider(provider_name, config, auth)
            
            # Attempt discovery
            models = await discoverer.list_models()
            
            # Augment each model with benchmarks
            models = [self._augment_with_benchmarks(m) for m in models]
            
            # Cache results
            discovery_type = config.get("discovery", {}).get("type", "none")
            ttl = config.get("discovery", {}).get("ttl_hours", 24)
            
            cache_entry = CacheEntry(models, ttl_hours=ttl)
            self.memory_cache[provider_name] = cache_entry
            self._persist_cache(provider_name, cache_entry)
            
            logger.info(f"{provider_name}: discovered {len(models)} models (cached, benchmarks augmented)")
            return models
        
        except Exception as e:
            logger.warning(f"{provider_name}: discovery failed: {e}")
            # Fallback to cached or config models
            return await self.get_models(provider_name)
    
    async def get_models(
        self,
        provider_name: str,
        config: Optional[Dict[str, Any]] = None,
        auth: Optional[Dict[str, Any]] = None,
    ) -> List[ModelDescriptor]:
        """Get models for a provider (from cache or discovery).
        
        Args:
            provider_name: Provider name
            config: Optional provider config (for fallback discovery)
            auth: Optional auth credentials (for fallback discovery)
        
        Returns:
            List of models (empty if not found)
        """
        # Check memory cache first
        if provider_name in self.memory_cache:
            cache_entry = self.memory_cache[provider_name]
            if not cache_entry.is_stale():
                logger.debug(f"{provider_name}: using memory cache ({len(cache_entry.models)} models)")
                return cache_entry.models
        
        # Check disk cache
        cached = self._load_cache(provider_name)
        if cached:
            self.memory_cache[provider_name] = cached
            logger.debug(f"{provider_name}: loaded from disk cache ({len(cached.models)} models)")
            return cached.models
        
        # If config/auth provided, attempt discovery
        if config and auth:
            logger.debug(f"{provider_name}: no cache found, attempting discovery")
            return await self.discover_provider(provider_name, config, auth)
        
        logger.warning(f"{provider_name}: no models found (not cached, no config provided)")
        return []
    
    def _persist_cache(self, provider_name: str, cache_entry: CacheEntry):
        """Save cache to disk."""
        try:
            cache_file = self._cache_file(provider_name)
            with open(cache_file, "w") as f:
                json.dump(cache_entry.to_dict(), f, indent=2)
            logger.debug(f"{provider_name}: cache persisted to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to persist cache for {provider_name}: {e}")
    
    def _load_cache(self, provider_name: str) -> Optional[CacheEntry]:
        """Load cache from disk."""
        try:
            cache_file = self._cache_file(provider_name)
            if not cache_file.exists():
                return None
            
            with open(cache_file) as f:
                data = json.load(f)
            
            cache_entry = CacheEntry.from_dict(data)
            
            # Check staleness
            if cache_entry.is_stale():
                logger.debug(f"{provider_name}: cache is stale (ttl={cache_entry.ttl_hours}h)")
                cache_file.unlink()  # Delete stale cache
                return None
            
            return cache_entry
        except Exception as e:
            logger.warning(f"Failed to load cache for {provider_name}: {e}")
            return None
    
    def invalidate_provider(self, provider_name: str) -> bool:
        """Manually invalidate cache for a provider.
        
        Args:
            provider_name: Provider name
        
        Returns:
            True if cache was invalidated, False if not found
        """
        # Remove from memory
        removed_memory = provider_name in self.memory_cache
        self.memory_cache.pop(provider_name, None)
        
        # Remove from disk
        cache_file = self._cache_file(provider_name)
        removed_disk = False
        if cache_file.exists():
            try:
                cache_file.unlink()
                removed_disk = True
                logger.info(f"{provider_name}: cache invalidated")
            except Exception as e:
                logger.error(f"Failed to delete cache file for {provider_name}: {e}")
        
        return removed_memory or removed_disk
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about current cache state.
        
        Returns:
            Dict with cache stats
        """
        stats = {
            "memory_cache_entries": len(self.memory_cache),
            "providers": {},
        }
        
        for provider_name, cache_entry in self.memory_cache.items():
            stats["providers"][provider_name] = {
                "models_count": len(cache_entry.models),
                "cached_at": cache_entry.timestamp.isoformat(),
                "ttl_hours": cache_entry.ttl_hours,
                "is_stale": cache_entry.is_stale(),
            }
        
        return stats


async def discover_all_providers(
    registry: ModelRegistry,
    full_config: Dict[str, Any],
) -> Dict[str, List[ModelDescriptor]]:
    """Discover models for all configured providers (parallel).
    
    Args:
        registry: ModelRegistry instance
        full_config: Full config dict (from load_config())
    
    Returns:
        Dict mapping provider_name -> list of models
    """
    providers_config = full_config.get("providers", {})
    
    # Prepare discovery tasks
    tasks = {}
    for provider_name, provider_cfg in providers_config.items():
        if not isinstance(provider_cfg, dict):
            logger.warning(f"Skipping {provider_name}: config is not dict")
            continue
        
        # Extract auth from environment or config
        auth = {
            "api_key": provider_cfg.get("api_key", ""),
        }
        
        tasks[provider_name] = registry.discover_provider(provider_name, provider_cfg, auth)
    
    # Run all discoveries concurrently
    results = {}
    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for provider_name, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.error(f"{provider_name}: discovery error: {result}")
                results[provider_name] = []
            else:
                results[provider_name] = result
    
    return results


if __name__ == "__main__":
    # Example: test registry
    print("Testing ModelRegistry...\n")
    
    registry = ModelRegistry()
    
    config = {
        "models": [
            {
                "id": "gpt-4o",
                "display_name": "GPT-4o",
                "context_window": 128000,
            },
            {
                "id": "gemma4:31b",
                "display_name": "Gemma 7B",
                "context_window": 8192,
            },
        ],
    }
    
    async def test():
        # Discover models
        models = await registry.discover_provider("test-provider", config, {})
        print(f"✅ Discovered {len(models)} models")
        for m in models:
            print(f"   {m.id}: score={m.capability_score}, source={m.benchmark_source}")
        
        # Get stats
        stats = registry.get_cache_stats()
        print(f"\n✅ Cache stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test())
