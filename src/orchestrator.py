"""Main Orchestrator for the Claude Agent Swarm framework."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, TypeVar
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

import yaml

from agent import ClaudeAgent, ClaudeModel, TokenUsage
from exceptions import (
    ConfigurationError,
    OrchestratorError,
    SwarmCreationError,
    TaskDistributionError,
)
from swarm_manager import SwarmManager
from task_decomposer import TaskDecomposer

# Type variables
T = TypeVar("T")

logger = logging.getLogger(__name__)


class SwarmPattern(Enum):
    """Enumeration of swarm execution patterns."""
    
    AUTO = auto()       # Automatically select best pattern
    LEADER = auto()     # Leader-follower pattern
    SWARM = auto()      # Decentralized swarm
    PIPELINE = auto()   # Sequential pipeline
    COUNCIL = auto()    # Consensus-based council


@dataclass
class AgentConfig:
    """Configuration for an agent in the swarm."""
    
    name: str
    model: ClaudeModel = "claude-3-7-sonnet-20250219"
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: list[str] = field(default_factory=list)
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


class SwarmOrchestrator:
    """
    Main orchestrator for the Claude Agent Swarm framework.
    
    The orchestrator manages the creation, configuration, and execution
    of agent swarms. It supports multiple execution patterns and provides
    dynamic agent spawning with up to 100 concurrent agents.
    
    Attributes:
        orchestrator_id: Unique identifier for this orchestrator
        config_path: Path to the configuration file
        max_concurrent_agents: Maximum number of concurrent agents
    
    Example:
        >>> orchestrator = await SwarmOrchestrator.create(
        ...     config_path="config.yaml"
        ... )
        >>> swarm_id = await orchestrator.create_swarm("my_swarm")
        >>> results = await orchestrator.execute_task(swarm_id, "Analyze this data")
    """
    
    # Maximum agents allowed per swarm
    MAX_AGENTS = 100
    
    def __init__(
        self,
        orchestrator_id: str,
        config_path: Path | str | None = None,
        max_concurrent_agents: int = 100,
        api_key: str | None = None,
        default_model: ClaudeModel = "claude-3-7-sonnet-20250219",
    ) -> None:
        """
        Initialize the orchestrator. Use `create()` for async initialization.
        
        Args:
            orchestrator_id: Unique identifier for this orchestrator
            config_path: Path to the YAML configuration file
            max_concurrent_agents: Maximum concurrent agents allowed
            api_key: Anthropic API key
            default_model: Default model for agents
        """
        self._orchestrator_id = orchestrator_id
        self._config_path = Path(config_path) if config_path else None
        self._max_concurrent_agents = min(max_concurrent_agents, self.MAX_AGENTS)
        self._api_key = api_key
        self._default_model = default_model
        
        # State
        self._initialized = False
        self._swarms: dict[str, SwarmManager] = {}
        self._swarm_configs: dict[str, SwarmConfig] = {}
        self._swarm_status: dict[str, SwarmStatus] = {}
        self._execution_plans: dict[str, ExecutionPlan] = {}
        self._global_token_usage = TokenUsage()
        
        # Components
        self._task_decomposer: TaskDecomposer | None = None
        
        # Progress tracking
        self._progress_callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        
        # Locks
        self._lock = asyncio.Lock()
        
        logger.debug(f"Orchestrator {orchestrator_id} initialized")
    
    @classmethod
    async def create(
        cls,
        config_path: Path | str | None = None,
        max_concurrent_agents: int = 100,
        api_key: str | None = None,
        default_model: ClaudeModel = "claude-3-7-sonnet-20250219",
        load_config: bool = True,
    ) -> SwarmOrchestrator:
        """
        Async factory method to create and initialize a SwarmOrchestrator.
        
        Args:
            config_path: Path to the YAML configuration file
            max_concurrent_agents: Maximum concurrent agents allowed
            api_key: Anthropic API key
            default_model: Default model for agents
            load_config: Whether to load configuration from file
        
        Returns:
            An initialized SwarmOrchestrator instance
        
        Raises:
            OrchestratorError: If initialization fails
        """
        orchestrator_id = str(uuid4())
        
        try:
            instance = cls(
                orchestrator_id=orchestrator_id,
                config_path=config_path,
                max_concurrent_agents=max_concurrent_agents,
                api_key=api_key,
                default_model=default_model,
            )
            
            # Initialize task decomposer
            instance._task_decomposer = await TaskDecomposer.create(
                api_key=api_key,
                model=default_model,
            )
            
            # Load configuration if provided
            if load_config and config_path:
                await instance._load_config()
            
            instance._initialized = True
            logger.info(f"Orchestrator {orchestrator_id} created successfully")
            return instance
            
        except Exception as e:
            raise OrchestratorError(
                f"Failed to initialize orchestrator: {e}",
                error_code="INIT_ERROR"
            ) from e
    
    @property
    def orchestrator_id(self) -> str:
        """Get the orchestrator's unique identifier."""
        return self._orchestrator_id
    
    @property
    def swarm_count(self) -> int:
        """Get the number of active swarms."""
        return len(self._swarms)
    
    @property
    def total_token_usage(self) -> TokenUsage:
        """Get the total token usage across all swarms."""
        return self._global_token_usage
    
    def register_progress_callback(
        self,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """
        Register a callback for progress updates.
        
        Args:
            callback: Function called with (event_type, data) on progress updates
        """
        self._progress_callbacks.append(callback)
    
    def unregister_progress_callback(
        self,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Unregister a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _notify_progress(self, event_type: str, data: dict[str, Any]) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def _load_config(self) -> None:
        """
        Load configuration from YAML file.
        
        Raises:
            ConfigurationError: If configuration loading fails
        """
        if not self._config_path or not self._config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self._config_path}",
                error_code="CONFIG_NOT_FOUND"
            )
        
        try:
            with open(self._config_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ConfigurationError(
                    "Configuration file is empty",
                    error_code="EMPTY_CONFIG"
                )
            
            # Parse swarm configurations
            swarms_config = config_data.get("swarms", [])
            for swarm_data in swarms_config:
                swarm_config = self._parse_swarm_config(swarm_data)
                self._swarm_configs[swarm_config.name] = swarm_config
            
            logger.info(f"Loaded configuration with {len(self._swarm_configs)} swarm definitions")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}",
                error_code="YAML_ERROR"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                error_code="CONFIG_LOAD_ERROR"
            ) from e
    
    def _parse_swarm_config(self, data: dict[str, Any]) -> SwarmConfig:
        """Parse swarm configuration from dictionary."""
        pattern_str = data.get("pattern", "auto").upper()
        try:
            pattern = SwarmPattern[pattern_str]
        except KeyError:
            pattern = SwarmPattern.AUTO
        
        agent_configs = []
        for agent_data in data.get("agents", []):
            agent_configs.append(AgentConfig(
                name=agent_data["name"],
                model=agent_data.get("model", self._default_model),
                system_prompt=agent_data.get(
                    "system_prompt",
                    "You are a helpful AI assistant."
                ),
                max_tokens=agent_data.get("max_tokens", 4096),
                temperature=agent_data.get("temperature", 0.7),
                tools=agent_data.get("tools", []),
                specialization=agent_data.get("specialization"),
                metadata=agent_data.get("metadata", {}),
            ))
        
        return SwarmConfig(
            name=data["name"],
            pattern=pattern,
            max_agents=data.get("max_agents", 10),
            agent_configs=agent_configs,
            shared_system_prompt=data.get("shared_system_prompt"),
            enable_mcp=data.get("enable_mcp", False),
            mcp_servers=data.get("mcp_servers", []),
            metadata=data.get("metadata", {}),
        )
    
    async def create_swarm(
        self,
        name: str | None = None,
        config: SwarmConfig | None = None,
        pattern: SwarmPattern | str | None = None,
        num_agents: int | None = None,
    ) -> str:
        """
        Create a new agent swarm.
        
        Args:
            name: Swarm name (optional, uses config name or generates one)
            config: Swarm configuration (optional, loads from file if not provided)
            pattern: Execution pattern (optional, overrides config)
            num_agents: Number of agents to create (optional, uses config)
        
        Returns:
            Swarm ID for the created swarm
        
        Raises:
            SwarmCreationError: If swarm creation fails
        """
        async with self._lock:
            try:
                # Determine configuration
                if config:
                    swarm_config = config
                elif name and name in self._swarm_configs:
                    swarm_config = self._swarm_configs[name]
                else:
                    # Create default configuration
                    swarm_config = SwarmConfig(
                        name=name or f"swarm_{len(self._swarms)}",
                        pattern=SwarmPattern.AUTO,
                        max_agents=num_agents or 5,
                    )
                
                # Override pattern if specified
                if pattern:
                    if isinstance(pattern, str):
                        swarm_config.pattern = SwarmPattern[pattern.upper()]
                    else:
                        swarm_config.pattern = pattern
                
                # Validate agent count
                if swarm_config.max_agents > self._max_concurrent_agents:
                    raise SwarmCreationError(
                        f"Requested {swarm_config.max_agents} agents exceeds "
                        f"maximum of {self._max_concurrent_agents}",
                        error_code="AGENT_LIMIT_EXCEEDED"
                    )
                
                # Create swarm manager
                swarm_id = str(uuid4())
                swarm_manager = await SwarmManager.create(
                    swarm_id=swarm_id,
                    config=swarm_config,
                    api_key=self._api_key,
                )
                
                self._swarms[swarm_id] = swarm_manager
                self._swarm_status[swarm_id] = SwarmStatus(
                    swarm_id=swarm_id,
                    config=swarm_config,
                )
                
                self._notify_progress("swarm_created", {
                    "swarm_id": swarm_id,
                    "config": swarm_config,
                })
                
                logger.info(f"Swarm {swarm_id} created with pattern {swarm_config.pattern.name}")
                return swarm_id
                
            except Exception as e:
                raise SwarmCreationError(
                    f"Failed to create swarm: {e}",
                    error_code="SWARM_CREATE_ERROR"
                ) from e
    
    async def execute_task(
        self,
        swarm_id: str,
        task: str,
        pattern: SwarmPattern | str | None = None,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task using a swarm.
        
        Args:
            swarm_id: ID of the swarm to use
            task: The task description or prompt
            pattern: Override the execution pattern
            context: Additional context for the task
            timeout: Execution timeout
        
        Returns:
            Dictionary containing results and metadata
        
        Raises:
            TaskDistributionError: If task distribution fails
        """
        if swarm_id not in self._swarms:
            raise TaskDistributionError(
                f"Swarm {swarm_id} not found",
                error_code="SWARM_NOT_FOUND"
            )
        
        swarm_manager = self._swarms[swarm_id]
        status = self._swarm_status[swarm_id]
        
        try:
            status.is_running = True
            status.start_time = datetime.now()
            status.total_tasks += 1
            
            self._notify_progress("task_started", {
                "swarm_id": swarm_id,
                "task": task,
            })
            
            # Analyze task and determine pattern
            effective_pattern = self._determine_pattern(
                task, pattern, status.config.pattern
            )
            
            # Decompose task if needed
            execution_plan = await self._plan_execution(
                swarm_id, task, effective_pattern, context
            )
            
            # Execute based on pattern
            if effective_pattern == SwarmPattern.LEADER:
                results = await self._execute_leader_pattern(
                    swarm_manager, execution_plan, timeout
                )
            elif effective_pattern == SwarmPattern.PIPELINE:
                results = await self._execute_pipeline_pattern(
                    swarm_manager, execution_plan, timeout
                )
            elif effective_pattern == SwarmPattern.COUNCIL:
                results = await self._execute_council_pattern(
                    swarm_manager, execution_plan, timeout
                )
            else:  # SWARM or AUTO
                results = await self._execute_swarm_pattern(
                    swarm_manager, execution_plan, timeout
                )
            
            # Update status
            status.completed_tasks += 1
            status.total_token_usage.add(
                TokenUsage(
                    input_tokens=results.get("token_usage", {}).get("input_tokens", 0),
                    output_tokens=results.get("token_usage", {}).get("output_tokens", 0),
                )
            )
            
            self._global_token_usage.add(status.total_token_usage)
            
            self._notify_progress("task_completed", {
                "swarm_id": swarm_id,
                "results": results,
            })
            
            return results
            
        except Exception as e:
            status.failed_tasks += 1
            raise TaskDistributionError(
                f"Task execution failed: {e}",
                error_code="EXECUTION_ERROR"
            ) from e
        finally:
            status.is_running = False
            status.end_time = datetime.now()
    
    def _determine_pattern(
        self,
        task: str,
        override_pattern: SwarmPattern | str | None,
        config_pattern: SwarmPattern,
    ) -> SwarmPattern:
        """Determine the execution pattern to use."""
        if override_pattern:
            if isinstance(override_pattern, str):
                return SwarmPattern[override_pattern.upper()]
            return override_pattern
        
        if config_pattern != SwarmPattern.AUTO:
            return config_pattern
        
        # Auto-select based on task characteristics
        # This is a simplified heuristic - can be enhanced
        if "analyze" in task.lower() or "review" in task.lower():
            return SwarmPattern.COUNCIL
        elif "process" in task.lower() or "transform" in task.lower():
            return SwarmPattern.PIPELINE
        elif "coordinate" in task.lower() or "manage" in task.lower():
            return SwarmPattern.LEADER
        
        return SwarmPattern.SWARM
    
    async def _plan_execution(
        self,
        swarm_id: str,
        task: str,
        pattern: SwarmPattern,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Create an execution plan for the task."""
        if not self._task_decomposer:
            raise OrchestratorError("Task decomposer not initialized")
        
        # Analyze complexity
        complexity = await self._task_decomposer.analyze_complexity(task, context)
        
        # Decompose task
        subtasks = await self._task_decomposer.decompose_task(
            task, complexity, context
        )
        
        # Get execution plan
        plan = await self._task_decomposer.get_execution_plan(
            subtasks, pattern.name.lower()
        )
        
        execution_plan = ExecutionPlan(
            plan_id=str(uuid4()),
            pattern=pattern,
            subtasks=subtasks,
            dependencies=plan.get("dependencies", {}),
            estimated_tokens=plan.get("estimated_tokens", 0),
            estimated_time=plan.get("estimated_time", 0.0),
        )
        
        self._execution_plans[execution_plan.plan_id] = execution_plan
        
        return execution_plan
    
    async def _execute_leader_pattern(
        self,
        swarm_manager: SwarmManager,
        plan: ExecutionPlan,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Execute using leader-follower pattern."""
        # First agent is the leader
        agents = await swarm_manager.spawn_agents(1)
        leader = agents[0]
        
        # Leader coordinates the task
        task_description = "\n".join(
            f"Subtask {i+1}: {subtask['description']}"
            for i, subtask in enumerate(plan.subtasks)
        )
        
        result = await leader.execute(
            f"As the leader, coordinate and complete the following subtasks:\n{task_description}",
            timeout=timeout,
        )
        
        return {
            "pattern": "leader",
            "leader_result": result,
            "token_usage": result.get("usage", {}),
        }
    
    async def _execute_pipeline_pattern(
        self,
        swarm_manager: SwarmManager,
        plan: ExecutionPlan,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Execute using pipeline pattern."""
        num_stages = len(plan.subtasks)
        agents = await swarm_manager.spawn_agents(num_stages)
        
        results = []
        pipeline_data = ""
        total_usage = TokenUsage()
        
        for i, (agent, subtask) in enumerate(zip(agents, plan.subtasks)):
            prompt = f"Stage {i+1}: {subtask['description']}\n\nInput: {pipeline_data}"
            
            result = await agent.execute(prompt, timeout=timeout)
            results.append({
                "stage": i + 1,
                "result": result,
            })
            
            pipeline_data = result.get("content", "")
            
            usage = result.get("usage", {})
            total_usage.input_tokens += usage.get("input_tokens", 0)
            total_usage.output_tokens += usage.get("output_tokens", 0)
        
        return {
            "pattern": "pipeline",
            "stages": results,
            "final_output": pipeline_data,
            "token_usage": {
                "input_tokens": total_usage.input_tokens,
                "output_tokens": total_usage.output_tokens,
                "total_tokens": total_usage.total_tokens,
            },
        }
    
    async def _execute_council_pattern(
        self,
        swarm_manager: SwarmManager,
        plan: ExecutionPlan,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Execute using council/consensus pattern."""
        num_members = min(len(plan.subtasks), 5)  # Max 5 council members
        agents = await swarm_manager.spawn_agents(num_members)
        
        # All agents analyze the same task
        task = plan.subtasks[0]["description"] if plan.subtasks else "Analyze the input"
        
        # Execute in parallel
        agent_tasks = [
            agent.execute(
                f"As a council member, provide your analysis:\n{task}",
                timeout=timeout,
            )
            for agent in agents
        ]
        
        responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Collect results
        analyses = []
        total_usage = TokenUsage()
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                analyses.append({
                    "member": i + 1,
                    "error": str(response),
                })
            else:
                analyses.append({
                    "member": i + 1,
                    "analysis": response.get("content", ""),
                })
                usage = response.get("usage", {})
                total_usage.input_tokens += usage.get("input_tokens", 0)
                total_usage.output_tokens += usage.get("output_tokens", 0)
        
        return {
            "pattern": "council",
            "analyses": analyses,
            "token_usage": {
                "input_tokens": total_usage.input_tokens,
                "output_tokens": total_usage.output_tokens,
                "total_tokens": total_usage.total_tokens,
            },
        }
    
    async def _execute_swarm_pattern(
        self,
        swarm_manager: SwarmManager,
        plan: ExecutionPlan,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Execute using decentralized swarm pattern."""
        # Spawn agents for each subtask
        num_agents = min(len(plan.subtasks), swarm_manager.config.max_agents)
        agents = await swarm_manager.spawn_agents(num_agents)
        
        # Create task-agent pairs
        task_agent_pairs = []
        for i, subtask in enumerate(plan.subtasks):
            agent = agents[i % len(agents)]
            task_agent_pairs.append((agent, subtask))
        
        # Execute in parallel
        results = await swarm_manager.execute_parallel(
            task_agent_pairs,
            timeout=timeout,
        )
        
        # Aggregate results
        total_usage = TokenUsage()
        for result in results:
            if "usage" in result:
                usage = result["usage"]
                total_usage.input_tokens += usage.get("input_tokens", 0)
                total_usage.output_tokens += usage.get("output_tokens", 0)
        
        return {
            "pattern": "swarm",
            "results": results,
            "token_usage": {
                "input_tokens": total_usage.input_tokens,
                "output_tokens": total_usage.output_tokens,
                "total_tokens": total_usage.total_tokens,
            },
        }
    
    async def get_status(self, swarm_id: str | None = None) -> dict[str, Any]:
        """
        Get status information.
        
        Args:
            swarm_id: Specific swarm ID, or None for all swarms
        
        Returns:
            Status dictionary
        """
        if swarm_id:
            if swarm_id not in self._swarm_status:
                raise OrchestratorError(f"Swarm {swarm_id} not found")
            
            status = self._swarm_status[swarm_id]
            return {
                "swarm_id": status.swarm_id,
                "config": {
                    "name": status.config.name,
                    "pattern": status.config.pattern.name,
                    "max_agents": status.config.max_agents,
                },
                "active_agents": status.active_agents,
                "total_tasks": status.total_tasks,
                "completed_tasks": status.completed_tasks,
                "failed_tasks": status.failed_tasks,
                "is_running": status.is_running,
                "token_usage": {
                    "input_tokens": status.total_token_usage.input_tokens,
                    "output_tokens": status.total_token_usage.output_tokens,
                    "total_tokens": status.total_token_usage.total_tokens,
                },
            }
        
        # Return status for all swarms
        return {
            "orchestrator_id": self._orchestrator_id,
            "total_swarms": len(self._swarms),
            "swarms": [
                {
                    "swarm_id": sid,
                    "name": self._swarm_status[sid].config.name,
                    "is_running": self._swarm_status[sid].is_running,
                }
                for sid in self._swarms
            ],
            "global_token_usage": {
                "input_tokens": self._global_token_usage.input_tokens,
                "output_tokens": self._global_token_usage.output_tokens,
                "total_tokens": self._global_token_usage.total_tokens,
            },
        }
    
    async def terminate_all(self, graceful: bool = True) -> None:
        """
        Terminate all swarms and clean up resources.
        
        Args:
            graceful: Whether to wait for ongoing tasks to complete
        """
        async with self._lock:
            self._notify_progress("terminating_all", {
                "swarm_count": len(self._swarms),
                "graceful": graceful,
            })
            
            # Terminate each swarm
            termination_tasks = []
            for swarm_id, swarm_manager in self._swarms.items():
                task = asyncio.create_task(swarm_manager.cleanup(graceful))
                termination_tasks.append(task)
            
            if termination_tasks:
                await asyncio.gather(*termination_tasks, return_exceptions=True)
            
            # Clear state
            self._swarms.clear()
            self._swarm_status.clear()
            self._execution_plans.clear()
            
            # Clean up task decomposer
            if self._task_decomposer:
                await self._task_decomposer.close()
                self._task_decomposer = None
            
            self._initialized = False
            
            logger.info(f"Orchestrator {self._orchestrator_id} terminated all swarms")
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.terminate_all(graceful=True)
    
    async def __aenter__(self) -> SwarmOrchestrator:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
