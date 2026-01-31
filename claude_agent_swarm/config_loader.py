"""
Configuration Loader for Claude Agent Swarm Framework

Provides YAML configuration parsing with Pydantic validation,
config merging, inheritance, and environment variable substitution.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(str, Enum):
    """Model provider enumeration."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"


class AgentCapability(str, Enum):
    """Agent capability enumeration."""
    CODE = "code"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    WRITING = "writing"
    REVIEW = "review"
    TESTING = "testing"
    PLANNING = "planning"
    COMMUNICATION = "communication"


class RetryConfig(BaseModel):
    """Retry configuration."""
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0)
    max_delay: float = Field(default=60.0, ge=0)
    exponential_base: float = Field(default=2.0, ge=1.0)
    
    @model_validator(mode="after")
    def max_delay_greater_than_base(self):
        """Ensure max_delay is greater than base_delay."""
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be greater than base_delay")
        return self


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=100000, ge=1)
    concurrent_requests: int = Field(default=5, ge=1)
    

class TokenBudgetConfig(BaseModel):
    """Token budget configuration."""
    max_input_tokens: int = Field(default=100000, ge=1)
    max_output_tokens: int = Field(default=4096, ge=1)
    total_budget: Optional[int] = Field(default=None, ge=1)
    warning_threshold: float = Field(default=0.8, ge=0, le=1.0)


class LLMConfig(BaseModel):
    """LLM configuration."""
    model: str = Field(default="claude-3-sonnet-20240229")
    provider: ModelProvider = Field(default=ModelProvider.ANTHROPIC)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    timeout: float = Field(default=60.0, ge=1)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    
    @field_validator("temperature")
    @classmethod
    def valid_temperature(cls, v: float) -> float:
        """Validate temperature is reasonable."""
        if v > 1.5:
            return 1.5
        return v


class ToolConfig(BaseModel):
    """Tool configuration."""
    name: str
    enabled: bool = True
    timeout: float = Field(default=30.0, ge=1)
    retry: Optional[RetryConfig] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolRegistryConfig(BaseModel):
    """Tool registry configuration."""
    tools: List[ToolConfig] = Field(default_factory=list)
    auto_discover: bool = True
    tool_timeout: float = Field(default=30.0, ge=1)


class AgentTemplate(BaseModel):
    """Agent template configuration."""
    name: str
    description: str = ""
    system_prompt: str = ""
    capabilities: List[AgentCapability] = Field(default_factory=list)
    llm: Optional[LLMConfig] = None
    tools: List[str] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1)
    token_budget: Optional[TokenBudgetConfig] = None
    rate_limits: Optional[RateLimitConfig] = None
    parent_template: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def merge_with_parent(self, parent: AgentTemplate) -> AgentTemplate:
        """Merge this template with a parent template."""
        merged_data = parent.dict()
        child_data = self.dict(exclude_unset=True)
        
        # Deep merge for certain fields
        merged_data["capabilities"] = list(set(
            parent.capabilities + self.capabilities
        ))
        merged_data["tools"] = list(set(
            parent.tools + self.tools
        ))
        
        # Override with child values
        for key, value in child_data.items():
            if key not in ("capabilities", "tools") and value is not None:
                merged_data[key] = value
        
        return AgentTemplate(**merged_data)


class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    name: str
    agent_template: str
    description: str = ""
    dependencies: List[str] = Field(default_factory=list)
    condition: Optional[str] = None  # Conditional execution
    timeout: Optional[float] = None
    retries: int = Field(default=0, ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_key: Optional[str] = None  # Store output in context


class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    name: str
    description: str = ""
    steps: List[WorkflowStep] = Field(default_factory=list)
    max_parallel: int = Field(default=5, ge=1)
    timeout: Optional[float] = None
    on_failure: str = Field(default="stop", pattern="^(stop|continue|retry)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def validate_step_dependencies(self):
        """Validate that step dependencies exist."""
        step_names = {step.name for step in self.steps}

        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(default="json", pattern="^(json|text)$")
    file: Optional[str] = None
    max_size_mb: int = Field(default=10, ge=1)
    backup_count: int = Field(default=5, ge=0)
    console: bool = True


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""
    enabled: bool = True
    export_format: str = Field(default="json", pattern="^(json|prometheus)$")
    export_path: Optional[str] = None
    max_events: int = Field(default=10000, ge=100)
    metrics_interval: float = Field(default=60.0, ge=1)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    enabled: bool = True
    refresh_rate: float = Field(default=0.5, ge=0.1)
    show_token_usage: bool = True
    show_tool_calls: bool = True
    show_timestamps: bool = True


class SwarmConfig(BaseModel):
    """Swarm configuration."""
    model_config = {"extra": "allow"}

    name: str = "Claude Agent Swarm"
    description: str = ""
    max_agents: int = Field(default=10, ge=1)
    default_llm: Optional[LLMConfig] = None
    agent_templates: Dict[str, AgentTemplate] = Field(default_factory=dict)
    workflows: Dict[str, WorkflowConfig] = Field(default_factory=dict)
    tools: ToolRegistryConfig = Field(default_factory=ToolRegistryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    global_rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)
    environment: Dict[str, str] = Field(default_factory=dict)


class ConfigLoader:
    """
    Configuration loader for Claude Agent Swarm.
    
    Features:
    - YAML configuration parsing
    - Pydantic validation
    - Config inheritance and merging
    - Environment variable substitution
    - Agent template resolution
    - Workflow validation
    
    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config.yaml")
        >>> agent_template = loader.get_agent_template("code_reviewer")
        >>> workflow = loader.get_workflow("main")
    """
    
    # Environment variable pattern: ${VAR_NAME} or ${VAR_NAME:-default}
    ENV_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self):
        """Initialize the config loader."""
        self._config: Optional[SwarmConfig] = None
        self._loaded_files: Set[str] = set()
        self._merged_configs: List[Dict[str, Any]] = []
    
    def load(
        self,
        config_path: Union[str, Path],
        base_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        env_substitution: bool = True
    ) -> SwarmConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            base_config: Optional base config to merge with
            env_substitution: Enable environment variable substitution
            
        Returns:
            Validated SwarmConfig
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load base config if provided
        merged_config: Dict[str, Any] = {}
        
        if base_config:
            if isinstance(base_config, (str, Path)):
                base_data = self._load_yaml_file(Path(base_config))
                merged_config = self._deep_merge(merged_config, base_data)
            elif isinstance(base_config, dict):
                merged_config = self._deep_merge(merged_config, base_config)
        
        # Load main config
        config_data = self._load_yaml_file(config_path)
        merged_config = self._deep_merge(merged_config, config_data)
        
        # Track loaded file
        self._loaded_files.add(str(config_path))
        
        # Environment variable substitution
        if env_substitution:
            merged_config = self._substitute_env_vars(merged_config)
        
        # Resolve agent template inheritance
        merged_config = self._resolve_template_inheritance(merged_config)
        
        # Validate and create config
        self._config = SwarmConfig(**merged_config)
        
        return self._config
    
    def load_multiple(
        self,
        config_paths: List[Union[str, Path]],
        env_substitution: bool = True
    ) -> SwarmConfig:
        """
        Load and merge multiple configuration files.
        
        Args:
            config_paths: List of config file paths
            env_substitution: Enable environment variable substitution
            
        Returns:
            Merged and validated SwarmConfig
        """
        merged_config: Dict[str, Any] = {}
        
        for path in config_paths:
            config_data = self._load_yaml_file(Path(path))
            merged_config = self._deep_merge(merged_config, config_data)
            self._loaded_files.add(str(path))
        
        if env_substitution:
            merged_config = self._substitute_env_vars(merged_config)
        
        merged_config = self._resolve_template_inheritance(merged_config)
        
        self._config = SwarmConfig(**merged_config)
        return self._config
    
    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load YAML file and return data."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data or {}
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # For lists, we extend (but could be configurable)
                result[key] = result[key] + value
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Substitute environment variables in data."""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_env_string(data)
        else:
            return data
    
    def _substitute_env_string(self, value: str) -> str:
        """Substitute environment variables in a string."""
        def replace_var(match):
            var_expr = match.group(1)
            
            # Check for default value syntax: VAR:-default
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name, default)
            
            # Check for required syntax: VAR:?error
            if ':?' in var_expr:
                var_name, error_msg = var_expr.split(':?', 1)
                if var_name not in os.environ:
                    raise ValueError(f"Required environment variable {var_name}: {error_msg}")
                return os.environ[var_name]
            
            return os.environ.get(var_expr, match.group(0))
        
        return self.ENV_PATTERN.sub(replace_var, value)
    
    def _resolve_template_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve agent template inheritance."""
        templates = config.get("agent_templates", {})
        
        if not templates:
            return config
        
        resolved = {}
        
        for name, template_data in templates.items():
            parent_name = template_data.get("parent_template")
            
            if parent_name and parent_name in templates:
                parent_data = templates[parent_name]
                # Merge parent into child
                merged = self._deep_merge(parent_data.copy(), template_data)
                resolved[name] = merged
            else:
                resolved[name] = template_data
        
        config["agent_templates"] = resolved
        return config
    
    def validate(self, config: Optional[SwarmConfig] = None) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Config to validate (uses loaded config if None)
            
        Returns:
            List of validation issues (empty if valid)
        """
        config = config or self._config
        issues = []
        
        if not config:
            return ["No configuration loaded"]
        
        # Validate agent templates reference valid LLM configs
        for name, template in config.agent_templates.items():
            if template.llm and template.llm.provider == ModelProvider.ANTHROPIC:
                if "ANTHROPIC_API_KEY" not in os.environ:
                    issues.append(
                        f"Agent template '{name}' uses Anthropic but ANTHROPIC_API_KEY not set"
                    )
        
        # Validate workflow steps reference valid agent templates
        for wf_name, workflow in config.workflows.items():
            for step in workflow.steps:
                if step.agent_template not in config.agent_templates:
                    issues.append(
                        f"Workflow '{wf_name}' step '{step.name}' references "
                        f"unknown agent template '{step.agent_template}'"
                    )
        
        # Validate tool references
        tool_names = {t.name for t in config.tools.tools}
        for name, template in config.agent_templates.items():
            for tool_name in template.tools:
                if tool_name not in tool_names:
                    issues.append(
                        f"Agent template '{name}' references unknown tool '{tool_name}'"
                    )
        
        return issues
    
    def get_agent_template(self, name: str) -> Optional[AgentTemplate]:
        """
        Get an agent template by name.
        
        Args:
            name: Template name
            
        Returns:
            Agent template or None if not found
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.agent_templates.get(name)
    
    def get_workflow(self, name: str) -> Optional[WorkflowConfig]:
        """
        Get a workflow by name.
        
        Args:
            name: Workflow name
            
        Returns:
            Workflow config or None if not found
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.workflows.get(name)
    
    def get_all_agent_templates(self) -> Dict[str, AgentTemplate]:
        """
        Get all agent templates.
        
        Returns:
            Dictionary of agent templates
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.agent_templates.copy()
    
    def get_all_workflows(self) -> Dict[str, WorkflowConfig]:
        """
        Get all workflows.
        
        Returns:
            Dictionary of workflows
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.workflows.copy()
    
    def get_default_llm_config(self) -> Optional[LLMConfig]:
        """
        Get default LLM configuration.
        
        Returns:
            Default LLM config or None
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.default_llm
    
    def get_logging_config(self) -> LoggingConfig:
        """
        Get logging configuration.
        
        Returns:
            Logging config
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.logging
    
    def get_telemetry_config(self) -> TelemetryConfig:
        """
        Get telemetry configuration.
        
        Returns:
            Telemetry config
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.telemetry
    
    def get_dashboard_config(self) -> DashboardConfig:
        """
        Get dashboard configuration.
        
        Returns:
            Dashboard config
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config.dashboard
    
    def get_config(self) -> SwarmConfig:
        """
        Get the loaded configuration.
        
        Returns:
            Loaded SwarmConfig
        """
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        return self._config
    
    def get_loaded_files(self) -> Set[str]:
        """
        Get set of loaded configuration files.
        
        Returns:
            Set of file paths
        """
        return self._loaded_files.copy()
    
    def create_agent_from_template(
        self,
        template_name: str,
        agent_id: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> AgentTemplate:
        """
        Create an agent configuration from a template with optional overrides.
        
        Args:
            template_name: Name of the template
            agent_id: Unique agent identifier
            overrides: Optional configuration overrides
            
        Returns:
            Agent template with overrides applied
        """
        template = self.get_agent_template(template_name)
        if not template:
            raise ValueError(f"Agent template '{template_name}' not found")
        
        if overrides:
            template_data = template.dict()
            merged = self._deep_merge(template_data, overrides)
            return AgentTemplate(**merged)
        
        return template
    
    def export_to_yaml(
        self,
        filepath: Union[str, Path],
        config: Optional[SwarmConfig] = None
    ) -> Path:
        """
        Export configuration to YAML file.
        
        Args:
            filepath: Output file path
            config: Config to export (uses loaded config if None)
            
        Returns:
            Path to exported file
        """
        config = config or self._config
        
        if not config:
            raise RuntimeError("No configuration to export")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and remove None values
        data = self._remove_none_values(config.dict())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return filepath
    
    def _remove_none_values(self, data: Any) -> Any:
        """Remove None values from data structure."""
        if isinstance(data, dict):
            return {k: self._remove_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data


def load_config(
    config_path: Union[str, Path],
    base_config: Optional[Union[str, Path]] = None
) -> SwarmConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file
        base_config: Optional base config
        
    Returns:
        Loaded SwarmConfig
    """
    loader = ConfigLoader()
    return loader.load(config_path, base_config)


def create_default_config() -> SwarmConfig:
    """
    Create a default configuration.
    
    Returns:
        Default SwarmConfig
    """
    return SwarmConfig(
        name="Claude Agent Swarm",
        description="Default swarm configuration",
        default_llm=LLMConfig(),
        agent_templates={
            "default": AgentTemplate(
                name="default",
                description="Default agent template",
                system_prompt="You are a helpful AI assistant.",
                capabilities=[AgentCapability.COMMUNICATION]
            )
        },
        workflows={
            "default": WorkflowConfig(
                name="default",
                description="Default workflow",
                steps=[]
            )
        }
    )
