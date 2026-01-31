"""Custom exceptions for the Claude Agent Swarm framework."""


class SwarmError(Exception):
    """Base exception for all swarm-related errors."""
    
    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class AgentError(SwarmError):
    """Exception raised for errors in agent operations."""
    pass


class AgentInitializationError(AgentError):
    """Exception raised when agent initialization fails."""
    pass


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    pass


class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out."""
    pass


class OrchestratorError(SwarmError):
    """Exception raised for errors in orchestrator operations."""
    pass


class SwarmCreationError(OrchestratorError):
    """Exception raised when swarm creation fails."""
    pass


class TaskDistributionError(OrchestratorError):
    """Exception raised when task distribution fails."""
    pass


class ConfigurationError(OrchestratorError):
    """Exception raised when configuration is invalid."""
    pass


class SwarmManagerError(SwarmError):
    """Exception raised for errors in swarm manager operations."""
    pass


class ResourceExhaustedError(SwarmManagerError):
    """Exception raised when swarm resources are exhausted."""
    pass


class ScalingError(SwarmManagerError):
    """Exception raised when scaling operations fail."""
    pass


class TaskDecomposerError(SwarmError):
    """Exception raised for errors in task decomposition."""
    pass


class ComplexityAnalysisError(TaskDecomposerError):
    """Exception raised when complexity analysis fails."""
    pass


class DependencyError(TaskDecomposerError):
    """Exception raised when dependency detection fails."""
    pass


class MCPError(SwarmError):
    """Exception raised for MCP server connection errors."""
    pass


class MCPConnectionError(MCPError):
    """Exception raised when MCP server connection fails."""
    pass


class ToolError(SwarmError):
    """Exception raised for tool execution errors."""
    pass


class ToolNotFoundError(ToolError):
    """Exception raised when a requested tool is not found."""
    pass


class ToolExecutionError(ToolError):
    """Exception raised when tool execution fails."""
    pass
