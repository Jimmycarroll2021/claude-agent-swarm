"""Shared data models for Claude Agent Swarm framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal

# Type alias for Claude models
ClaudeModel = Literal[
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]


class SwarmPattern(Enum):
    """Enumeration of swarm execution patterns."""

    AUTO = auto()       # Automatically select best pattern
    LEADER = auto()     # Leader-follower pattern
    SWARM = auto()      # Decentralized swarm
    PIPELINE = auto()   # Sequential pipeline
    COUNCIL = auto()    # Consensus-based council


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens

    def add(self, other: "TokenUsage") -> None:
        """Add another TokenUsage to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AgentConfig:
    """Configuration for an agent in the swarm."""

    name: str = "agent"
    model: ClaudeModel = "claude-3-7-sonnet-20250219"
    system_prompt: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: list[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 3
    specialization: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Configuration for a swarm."""

    name: str
    pattern: SwarmPattern = SwarmPattern.AUTO
    max_agents: int = 10
    agent_configs: list[AgentConfig] = field(default_factory=list)
    shared_system_prompt: str | None = None
    enable_mcp: bool = False
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    agent_id: str | None
    status: Literal["success", "failure", "timeout", "cancelled"]
    result: Any = None
    error: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmStatus:
    """Status information for a swarm."""

    swarm_id: str
    config: SwarmConfig
    active_agents: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    is_running: bool = False
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan for a task."""

    plan_id: str
    pattern: SwarmPattern
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    estimated_tokens: int = 0
    estimated_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexityScore:
    """Complexity score for a task."""

    overall: float = 0.5  # 0.0 to 1.0
    cognitive: float = 0.5  # Cognitive complexity
    domain: float = 0.5  # Domain knowledge required
    steps: float = 0.5  # Number of steps required
    dependencies: float = 0.5  # Dependency complexity
    data_volume: float = 0.5  # Amount of data to process

    @property
    def is_complex(self) -> bool:
        """Determine if task is complex enough to benefit from swarm."""
        return self.overall > 0.6

    @property
    def complexity_level(self) -> str:
        """Get human-readable complexity level."""
        if self.overall < 0.3:
            return "low"
        elif self.overall < 0.7:
            return "moderate"
        else:
            return "high"

    @property
    def recommended_agents(self) -> int:
        """Recommend number of agents based on complexity."""
        if self.overall < 0.3:
            return 1
        elif self.overall < 0.5:
            return 2
        elif self.overall < 0.7:
            return 3
        elif self.overall < 0.85:
            return 5
        else:
            return min(10, int(self.overall * 10))


@dataclass
class Subtask:
    """Represents a decomposed subtask."""

    subtask_id: str
    description: str
    estimated_complexity: ComplexityScore
    dependencies: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    estimated_time: float = 0.0
    required_specialization: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancePlan:
    """Plan for distributing tasks across agents."""

    batches: list[list[str]]  # Task IDs grouped for parallel execution
    agent_assignments: dict[str, str]  # task_id -> agent_id
    estimated_total_time: float
    estimated_total_tokens: int
    parallelization_factor: float  # Ratio of parallel to sequential work


@dataclass
class Task:
    """Represents a task to be executed by agents."""

    task_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    assigned_agent: str | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
