"""Configuration management for Cloud FinOps environment.

Provides structured configuration with validation, type safety, and
support for environment variables, config files, and programmatic setup.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator

from server.exceptions import ConfigurationError


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentConfig(BaseModel):
    """Configuration for environment simulation.
    
    Controls simulation parameters, episode settings, and resource initialization.
    """
    
    max_steps: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum steps per episode"
    )
    
    random_seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility (None for non-deterministic)"
    )
    
    cost_variance: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Variance in resource costs (0.0-0.5)"
    )
    
    enable_task_hints: bool = Field(
        default=False,
        description="Whether to include hints in observations"
    )
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Immutable after creation


class APIConfig(BaseModel):
    """Configuration for API endpoints and networking.
    
    Controls FastAPI server behavior, CORS, and security settings.
    """
    
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the environment API"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port"
    )
    
    allow_origins: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    
    max_request_size: int = Field(
        default=1_000_000,  # 1MB
        ge=100_000,
        le=100_000_000,
        description="Maximum request body size in bytes"
    )
    
    enable_docs: bool = Field(
        default=True,
        description="Enable interactive API documentation"
    )
    
    class Config:
        frozen = True


class LLMConfig(BaseModel):
    """Configuration for LLM inference.
    
    Controls LLM API connectivity, model selection, and inference parameters.
    """
    
    model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat",
        description="Model identifier (HuggingFace model ID)"
    )
    
    hf_token: str = Field(
        default="",
        description="HuggingFace API token for authentication"
    )
    
    api_endpoint: str = Field(
        default="https://api-inference.huggingface.co/openai/v1",
        description="LLM API endpoint (OpenAI-compatible)"
    )
    
    max_tokens: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="Maximum tokens in LLM response"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (higher = more creative)"
    )
    
    timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="LLM request timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for LLM calls"
    )
    
    retry_backoff: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Exponential backoff multiplier for retries"
    )
    
    @validator("hf_token", always=True)
    @classmethod
    def validate_token(cls, v):
        """Validate HF token is not empty in production."""
        if not v or not v.strip():
            raise ValueError("HuggingFace token is required")
        return v
    
    class Config:
        frozen = True


class ApplicationConfig(BaseModel):
    """Complete application configuration combining all subsystems.
    
    This is the root configuration object that integrates
    environment, API, and LLM configurations.
    """
    
    app_name: str = Field(
        default="Cloud FinOps & Security Auditor",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application-wide logging level"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Environment simulation configuration"
    )
    
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API and networking configuration"
    )
    
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM inference configuration"
    )
    
    class Config:
        frozen = True
    
    @classmethod
    def from_environment(cls) -> "ApplicationConfig":
        """Load configuration from environment variables.
        
        Reads all configuration from environment with sensible defaults.
        Validates all values before returning.
        
        Returns:
            Validated ApplicationConfig instance
            
        Raises:
            ConfigurationError: If required variables are missing or invalid
        """
        try:
            # Load required LLM config (will fail if not present)
            llm_config = LLMConfig(
                model_name=os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat"),
                hf_token=os.getenv("HF_TOKEN", ""),
                api_endpoint=os.getenv("LLM_ENDPOINT", "https://api-inference.huggingface.co/openai/v1"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                timeout=int(os.getenv("LLM_TIMEOUT", "30")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
                retry_backoff=float(os.getenv("LLM_RETRY_BACKOFF", "1.0")),
            )
            
            # Load optional API config
            api_config = APIConfig(
                api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                request_timeout=int(os.getenv("API_TIMEOUT", "30")),
                enable_docs=os.getenv("API_ENABLE_DOCS", "true").lower() == "true",
            )
            
            # Load optional environment config
            env_config = EnvironmentConfig(
                max_steps=int(os.getenv("ENV_MAX_STEPS", "100")),
                random_seed=int(os.getenv("ENV_SEED", "-1")) if os.getenv("ENV_SEED") and os.getenv("ENV_SEED") != "-1" else None,
                cost_variance=float(os.getenv("ENV_COST_VARIANCE", "0.0")),
                enable_task_hints=os.getenv("ENV_TASK_HINTS", "false").lower() == "true",
            )
            
            # Combine into full config
            log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
            
            config = cls(
                log_level=LogLevel[log_level_str] if log_level_str in LogLevel.__members__ else LogLevel.INFO,
                debug=os.getenv("DEBUG", "false").lower() == "true",
                environment=env_config,
                api=api_config,
                llm=llm_config,
            )
            
            return config
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}")
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets).
        
        Returns:
            Dictionary representation with sensitive values masked
        """
        d = self.dict()
        # Mask sensitive values
        if d.get("llm", {}).get("hf_token"):
            d["llm"]["hf_token"] = "***MASKED***"
        return d
