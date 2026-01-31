"""
Claude Agent Swarm Framework

A production-ready agent swarm framework for Claude Code with dynamic agent
spawning, parallel execution, and multiple orchestration patterns.
"""

from .models import (
    ClaudeModel,
    TokenUsage,
    SwarmPattern,
    AgentConfig,
    SwarmConfig,
    TaskResult,
    SwarmStatus,
    ExecutionPlan,
    ComplexityScore,
    Subtask,
    LoadBalancePlan,
    Task,
)

from .agent import (
    ClaudeAgent,
    Agent,
    AgentStatus,
    AgentMessage,
)

from .orchestrator import SwarmOrchestrator

from .swarm_manager import SwarmManager

from .task_decomposer import TaskDecomposer

from .exceptions import (
    SwarmError,
    AgentError,
    AgentInitializationError,
    AgentExecutionError,
    AgentTimeoutError,
    OrchestratorError,
    SwarmCreationError,
    TaskDistributionError,
    ConfigurationError,
    SwarmManagerError,
    ResourceExhaustedError,
    ScalingError,
    TaskDecomposerError,
    ComplexityAnalysisError,
    DependencyError,
    MCPError,
    MCPConnectionError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
)

from .config_loader import (
    ConfigLoader,
    load_config,
    create_default_config,
)

from .tools import (
    BaseTool,
    ToolResult,
    ToolSchema,
    ToolRegistry,
    ToolExecutor,
    get_global_registry,
    register_tool,
    get_tool,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",

    # Models
    "ClaudeModel",
    "TokenUsage",
    "SwarmPattern",
    "AgentConfig",
    "SwarmConfig",
    "TaskResult",
    "SwarmStatus",
    "ExecutionPlan",
    "ComplexityScore",
    "Subtask",
    "LoadBalancePlan",
    "Task",

    # Agent
    "ClaudeAgent",
    "Agent",
    "AgentStatus",
    "AgentMessage",

    # Orchestrator
    "SwarmOrchestrator",

    # Swarm Manager
    "SwarmManager",

    # Task Decomposer
    "TaskDecomposer",

    # Configuration
    "ConfigLoader",
    "load_config",
    "create_default_config",

    # Tools
    "BaseTool",
    "ToolResult",
    "ToolSchema",
    "ToolRegistry",
    "ToolExecutor",
    "get_global_registry",
    "register_tool",
    "get_tool",

    # Exceptions
    "SwarmError",
    "AgentError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "OrchestratorError",
    "SwarmCreationError",
    "TaskDistributionError",
    "ConfigurationError",
    "SwarmManagerError",
    "ResourceExhaustedError",
    "ScalingError",
    "TaskDecomposerError",
    "ComplexityAnalysisError",
    "DependencyError",
    "MCPError",
    "MCPConnectionError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
]
