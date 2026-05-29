"""Unified model discovery interface and concrete implementations.

Provides abstract base class ModelDiscoveryProvider and concrete implementations
for each provider type (OpenAI-compatible, AWS SDK, static config, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ModelDescriptor:
    """Metadata for a discovered model."""
    id: str                              # Model identifier (e.g., "gpt-4", "deepseek-v4-flash:cloud")
    display_name: str                    # Human-readable name
    context_window: int                  # Max context length in tokens
    reasoning_capability: Optional[str]  # "low", "medium", "high", or None
    provider: str                        # Provider name (e.g., "ollama-cloud")
    
    # NEW (Phase 3): Benchmark-derived capability scores
    capability_score: Optional[float] = None       # 0-1.0 (from weighted benchmarks)
    latency_tier: Optional[str] = None             # "fast", "balanced", "slow"
    humaneval_pct: Optional[float] = None          # HumanEval % (0-100)
    mmlu_pct: Optional[float] = None               # MMLU % (0-100)
    math_pct: Optional[float] = None               # MATH % (0-100)
    gpqa_pct: Optional[float] = None               # GPQA % (0-100)
    benchmark_source: str = "discovered"           # "published", "estimated", "discovered"
    benchmark_date: Optional[str] = None           # YYYY-MM when benchmarks were published
    parameters_billions: Optional[float] = None    # Model size in billions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


class ModelDiscoveryProvider(ABC):
    """Abstract base for provider model discovery.
    
    Subclasses implement discovery strategies for specific providers:
    - OpenAI-compatible endpoints (/v1/models)
    - AWS SDK operations (ListFoundationModels)
    - OAuth-based discovery (future)
    - Static config-backed lists
    """
    
    def __init__(self, provider_name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        """Initialize discovery provider.
        
        Args:
            provider_name: Name of the provider (e.g., "ollama-cloud")
            config: Provider configuration dict (from config.yaml)
            auth: Authentication credentials dict (api_key, aws credentials, etc.)
        """
        self.provider_name = provider_name
        self.config = config
        self.auth = auth
    
    @abstractmethod
    async def list_models(self) -> List[ModelDescriptor]:
        """Discover and return available models for this provider.
        
        Returns:
            List of ModelDescriptor objects
        
        Raises:
            Exception: If discovery fails
        """
        pass
    
    @abstractmethod
    def supports_dynamic_discovery(self) -> bool:
        """Check if provider supports live model enumeration.
        
        Returns:
            True if provider has a discovery API/endpoint
            False if discovery is static/config-only
        """
        pass


class OpenAICompatibleDiscovery(ModelDiscoveryProvider):
    """Discovery for OpenAI-compatible /v1/models endpoints.
    
    Handles providers that implement the OpenAI API standard:
    - Ollama Cloud
    - OpenRouter
    - DeepSeek
    - Kimi
    - MiniMax
    - Anthropic
    - etc.
    """
    
    async def list_models(self) -> List[ModelDescriptor]:
        """Call /v1/models endpoint and parse response."""
        import httpx
        
        base_url = self.config.get("base_url", "").strip()
        if not base_url:
            logger.warning(f"{self.provider_name}: no base_url configured")
            return []
        
        discovery_cfg = self.config.get("discovery", {})
        endpoint = discovery_cfg.get("endpoint", "/v1/models")
        auth_header = discovery_cfg.get("auth_header", "Authorization")
        auth_scheme = discovery_cfg.get("auth_scheme", "Bearer")
        
        # Build full URL
        url = base_url.rstrip("/") + endpoint
        
        # Build auth header
        auth_value = self.auth.get("api_key", "")
        if not auth_value:
            logger.warning(f"{self.provider_name}: no api_key in auth")
            return []
        
        headers = {}
        if auth_scheme == "Bearer":
            headers[auth_header] = f"Bearer {auth_value}"
        else:
            headers[auth_header] = auth_value
        
        # Make request
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            
            # Parse OpenAI-compatible response
            models = []
            for entry in data.get("data", []):
                model_id = entry.get("id") or entry.get("model")
                if not model_id:
                    continue
                
                models.append(ModelDescriptor(
                    id=model_id,
                    display_name=entry.get("name", entry.get("owned_by", model_id)),
                    context_window=entry.get("context_length") or \
                                   entry.get("context_window") or \
                                   self._infer_context_window(model_id),
                    reasoning_capability=None,  # Will be scored in Phase 3
                    provider=self.provider_name,
                ))
            
            logger.info(f"{self.provider_name}: discovered {len(models)} models")
            return models
        
        except Exception as e:
            logger.error(f"{self.provider_name}: discovery failed: {e}")
            raise
    
    def supports_dynamic_discovery(self) -> bool:
        return True
    
    def _infer_context_window(self, model_id: str) -> int:
        """Infer context window from model name or fallback to default."""
        # Known models mapping (can be expanded)
        context_map = {
            "gemma4:31b": 8192,
            "gemma-2-9b": 8192,
            "deepseek-v4-flash": 8000,
            "deepseek-v4-pro": 8000,
            "kimi-k2.6": 200000,
            "kimi-k2": 128000,
            "qwen3.5": 32000,
            "qwen3.5:397b": 32000,
            "glm-5.1": 8000,
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
        }
        
        # Try exact match
        if model_id in context_map:
            return context_map[model_id]
        
        # Try substring match
        for key, val in context_map.items():
            if key.lower() in model_id.lower():
                return val
        
        # Default fallback
        return 4096


class AWSBedrockDiscovery(ModelDiscoveryProvider):
    """Discovery for AWS Bedrock via ListFoundationModels API.
    
    Uses boto3 to call ListFoundationModels and DescribeFoundationModel.
    """
    
    async def list_models(self) -> List[ModelDescriptor]:
        """Call AWS Bedrock ListFoundationModels."""
        try:
            import boto3
        except ImportError:
            logger.error("boto3 not installed; cannot discover Bedrock models")
            return []
        
        try:
            region = self.config.get("aws_region", "us-east-1")
            client = boto3.client("bedrock", region_name=region)
            
            models = []
            paginator = client.get_paginator("list_foundation_models")
            
            for page in paginator.paginate():
                for model_summary in page.get("modelSummaries", []):
                    model_id = model_summary.get("modelId")
                    if not model_id:
                        continue
                    
                    # Optionally get additional details via DescribeFoundationModel
                    context_window = 4096  # Fallback
                    try:
                        details = client.describe_foundation_model(modelIdentifier=model_id)
                        model_details = details.get("modelDetails", {})
                        context_window = model_details.get("modelArn")
                        # Parse context from model specs if available
                        if "outputTokenLimit" in model_details:
                            context_window = model_details["outputTokenLimit"]
                    except Exception:
                        pass
                    
                    models.append(ModelDescriptor(
                        id=model_id,
                        display_name=model_summary.get("modelName", model_id),
                        context_window=context_window,
                        reasoning_capability=None,
                        provider=self.provider_name,
                    ))
            
            logger.info(f"{self.provider_name}: discovered {len(models)} models via Bedrock")
            return models
        
        except Exception as e:
            logger.error(f"{self.provider_name}: Bedrock discovery failed: {e}")
            raise
    
    def supports_dynamic_discovery(self) -> bool:
        return True


class StaticConfigDiscovery(ModelDiscoveryProvider):
    """Fallback discovery: load models from config.yaml.
    
    Used when a provider has no API or when API discovery fails.
    Models are manually maintained in the config.
    """
    
    async def list_models(self) -> List[ModelDescriptor]:
        """Load models from config.yaml models list."""
        models = []
        
        for model_cfg in self.config.get("models", []):
            if not isinstance(model_cfg, dict):
                logger.warning(f"{self.provider_name}: invalid model config: {model_cfg}")
                continue
            
            model_id = model_cfg.get("id")
            if not model_id:
                logger.warning(f"{self.provider_name}: model missing id")
                continue
            
            models.append(ModelDescriptor(
                id=model_id,
                display_name=model_cfg.get("display_name", model_id),
                context_window=model_cfg.get("context_window", 4096),
                reasoning_capability=model_cfg.get("reasoning_capability"),
                provider=self.provider_name,
            ))
        
        logger.info(f"{self.provider_name}: loaded {len(models)} models from config")
        return models
    
    def supports_dynamic_discovery(self) -> bool:
        return False


def create_discovery_provider(
    provider_name: str,
    config: Dict[str, Any],
    auth: Dict[str, Any],
) -> ModelDiscoveryProvider:
    """Factory function to create appropriate discovery provider.
    
    Args:
        provider_name: Provider name (e.g., "ollama-cloud")
        config: Provider config dict
        auth: Auth credentials dict
    
    Returns:
        ModelDiscoveryProvider instance (appropriate subclass)
    
    Raises:
        ValueError: If discovery type is not supported
    """
    discovery_cfg = config.get("discovery", {})
    discovery_type = discovery_cfg.get("type", "none")
    
    if discovery_type == "openai_compatible":
        return OpenAICompatibleDiscovery(provider_name, config, auth)
    elif discovery_type == "aws_sdk":
        return AWSBedrockDiscovery(provider_name, config, auth)
    elif discovery_type == "none" or discovery_type is None:
        return StaticConfigDiscovery(provider_name, config, auth)
    else:
        raise ValueError(f"Unknown discovery type: {discovery_type}")


if __name__ == "__main__":
    # Example: test StaticConfigDiscovery
    print("Testing ModelDiscoveryProvider implementations...\n")
    
    config = {
        "models": [
            {
                "id": "deepseek-v4-flash:cloud",
                "display_name": "DeepSeek v4 Flash",
                "context_window": 8000,
                "reasoning_capability": "medium",
            },
            {
                "id": "kimi-k2.6:cloud",
                "display_name": "Kimi K2.6",
                "context_window": 200000,
                "reasoning_capability": "high",
            },
        ],
    }
    
    # Test factory and static discovery
    provider = create_discovery_provider("test-provider", config, {})
    
    async def test():
        models = await provider.list_models()
        print(f"✅ Discovered {len(models)} models:")
        for m in models:
            print(f"   - {m.id} ({m.context_window} tokens, reasoning: {m.reasoning_capability})")
    
    asyncio.run(test())
