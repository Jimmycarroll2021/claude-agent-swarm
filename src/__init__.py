"""
Claude Agent Swarm Framework - Observability Components

Provides telemetry, logging, dashboard, task board, and configuration
management for the Claude Agent Swarm framework.
"""

from .telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    TokenUsage,
    ExecutionMetrics,
    EventType,
    get_telemetry,
    set_telemetry
)

from .logging_config import (
    configure_logging,
    get_logger,
    LogContext,
    AgentLogger,
    bind_context,
    unbind_context,
    clear_context,
    log_agent_event,
    log_tool_call,
    log_llm_request,
    log_llm_response,
    init_default_logging,
    get_default_logger
)

from .config_loader import (
    ConfigLoader,
    SwarmConfig,
    AgentTemplate,
    WorkflowConfig,
    LLMConfig,
    LoggingConfig,
    TelemetryConfig,
    DashboardConfig,
    load_config,
    create_default_config
)

from .ui import (
    SwarmDashboard,
    AsyncSwarmDashboard,
    TaskBoard,
    Task,
    TaskResult,
    TaskStatus
)

__version__ = "0.1.0"

__all__ = [
    # Telemetry
    "TelemetryCollector",
    "TelemetryEvent",
    "TokenUsage",
    "ExecutionMetrics",
    "EventType",
    "get_telemetry",
    "set_telemetry",
    
    # Logging
    "configure_logging",
    "get_logger",
    "LogContext",
    "AgentLogger",
    "bind_context",
    "unbind_context",
    "clear_context",
    "log_agent_event",
    "log_tool_call",
    "log_llm_request",
    "log_llm_response",
    "init_default_logging",
    "get_default_logger",
    
    # Config
    "ConfigLoader",
    "SwarmConfig",
    "AgentTemplate",
    "WorkflowConfig",
    "LLMConfig",
    "LoggingConfig",
    "TelemetryConfig",
    "DashboardConfig",
    "load_config",
    "create_default_config",
    
    # UI
    "SwarmDashboard",
    "AsyncSwarmDashboard",
    "TaskBoard",
    "Task",
    "TaskResult",
    "TaskStatus"
]
