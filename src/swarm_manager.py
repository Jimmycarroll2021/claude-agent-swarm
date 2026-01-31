"""Swarm Manager for the Claude Agent Swarm framework."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar
from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from agent import ClaudeAgent, ClaudeModel, TokenUsage
from exceptions import (
    AgentError,
    ResourceExhaustedError,
    ScalingError,
    SwarmManagerError,
)
from orchestrator import AgentConfig, SwarmConfig

# Type variables
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class AgentInstance:
    """Represents a managed agent instance."""
    
    instance_id: str
    agent: ClaudeAgent
    config: AgentConfig
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    task_count: int = 0
    total_execution_time: float = 0.0
    is_active: bool = True
    is_busy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """Metrics for swarm resource usage."""
    
    active_agents: int = 0
    idle_agents: int = 0
    busy_agents: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_execution_time: float = 0.0
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of a parallel execution."""
    
    task_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    timestamp: datetime = field(default_factory=datetime.now)


class SwarmManager:
    """
    Manages the lifecycle and execution of agent swarms.
    
    The SwarmManager handles agent creation, reuse, termination, and
    parallel execution coordination. It provides load balancing and
    resource monitoring capabilities.
    
    Attributes:
        swarm_id: Unique identifier for this swarm
        config: Swarm configuration
        max_concurrent: Maximum concurrent executions allowed
    
    Example:
        >>> manager = await SwarmManager.create(swarm_id="swarm_1", config=swarm_config)
        >>> agents = await manager.spawn_agents(5)
        >>> results = await manager.execute_parallel(tasks)
    """
    
    def __init__(
        self,
        swarm_id: str,
        config: SwarmConfig,
        api_key: str | None = None,
        max_concurrent: int = 50,
        agent_timeout: float = 300.0,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the swarm manager. Use `create()` for async initialization.
        
        Args:
            swarm_id: Unique identifier for this swarm
            config: Swarm configuration
            api_key: Anthropic API key
            max_concurrent: Maximum concurrent executions
            agent_timeout: Timeout for agent operations
            enable_metrics: Whether to collect metrics
        """
        self._swarm_id = swarm_id
        self._config = config
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._agent_timeout = agent_timeout
        self._enable_metrics = enable_metrics
        
        # State
        self._initialized = False
        self._agents: dict[str, AgentInstance] = {}
        self._agent_pool: asyncio.Queue[AgentInstance] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = ResourceMetrics()
        self._metrics_history: list[ResourceMetrics] = []
        
        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        
        logger.debug(f"SwarmManager {swarm_id} initialized")
    
    @classmethod
    async def create(
        cls,
        swarm_id: str,
        config: SwarmConfig,
        api_key: str | None = None,
        max_concurrent: int = 50,
        agent_timeout: float = 300.0,
        enable_metrics: bool = True,
    ) -> SwarmManager:
        """
        Async factory method to create and initialize a SwarmManager.
        
        Args:
            swarm_id: Unique identifier for this swarm
            config: Swarm configuration
            api_key: Anthropic API key
            max_concurrent: Maximum concurrent executions
            agent_timeout: Timeout for agent operations
            enable_metrics: Whether to collect metrics
        
        Returns:
            An initialized SwarmManager instance
        
        Raises:
            SwarmManagerError: If initialization fails
        """
        try:
            instance = cls(
                swarm_id=swarm_id,
                config=config,
                api_key=api_key,
                max_concurrent=max_concurrent,
                agent_timeout=agent_timeout,
                enable_metrics=enable_metrics,
            )
            
            # Start background tasks
            instance._cleanup_task = asyncio.create_task(
                instance._cleanup_loop()
            )
            
            if enable_metrics:
                instance._metrics_task = asyncio.create_task(
                    instance._metrics_loop()
                )
            
            instance._initialized = True
            logger.info(f"SwarmManager {swarm_id} created successfully")
            return instance
            
        except Exception as e:
            raise SwarmManagerError(
                f"Failed to initialize swarm manager: {e}",
                error_code="INIT_ERROR"
            ) from e
    
    @property
    def swarm_id(self) -> str:
        """Get the swarm's unique identifier."""
        return self._swarm_id
    
    @property
    def config(self) -> SwarmConfig:
        """Get the swarm configuration."""
        return self._config
    
    @property
    def agent_count(self) -> int:
        """Get the current number of agents."""
        return len(self._agents)
    
    @property
    def metrics(self) -> ResourceMetrics:
        """Get the current resource metrics."""
        return self._metrics
    
    async def spawn_agents(
        self,
        count: int,
        configs: list[AgentConfig] | None = None,
        reuse_existing: bool = True,
    ) -> list[ClaudeAgent]:
        """
        Spawn new agents for the swarm.
        
        Args:
            count: Number of agents to spawn
            configs: Optional list of agent configurations
            reuse_existing: Whether to reuse idle agents from the pool
        
        Returns:
            List of spawned agent instances
        
        Raises:
            ResourceExhaustedError: If agent limit is reached
        """
        async with self._lock:
            # Check if we can reuse existing agents
            if reuse_existing:
                available = [
                    inst for inst in self._agents.values()
                    if inst.is_active and not inst.is_busy
                ]
                
                if len(available) >= count:
                    # Reuse existing agents
                    selected = available[:count]
                    for inst in selected:
                        inst.is_busy = True
                        inst.last_used = datetime.now()
                    
                    logger.debug(f"Reused {count} existing agents")
                    return [inst.agent for inst in selected]
            
            # Calculate how many new agents to create
            current_count = len(self._agents)
            needed = count - (len(available) if reuse_existing else 0)
            
            if current_count + needed > self._config.max_agents:
                raise ResourceExhaustedError(
                    f"Cannot spawn {needed} more agents. "
                    f"Current: {current_count}, Max: {self._config.max_agents}",
                    error_code="AGENT_LIMIT_REACHED"
                )
            
            # Create new agents
            new_agents: list[AgentInstance] = []
            agent_configs = configs or []
            
            for i in range(needed):
                config = agent_configs[i] if i < len(agent_configs) else AgentConfig(
                    name=f"agent_{current_count + i}",
                    model=self._config.agent_configs[0].model
                    if self._config.agent_configs
                    else "claude-3-7-sonnet-20250219",
                    system_prompt=self._config.shared_system_prompt
                    or "You are a helpful AI assistant.",
                )
                
                agent = await ClaudeAgent.create(
                    model=config.model,
                    system_prompt=config.system_prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    api_key=self._api_key,
                )
                
                # Register tools
                for tool_name in config.tools:
                    # Tool registration would happen here
                    pass
                
                instance = AgentInstance(
                    instance_id=str(uuid4()),
                    agent=agent,
                    config=config,
                    is_busy=True,  # Mark as busy since they're being used
                )
                
                self._agents[instance.instance_id] = instance
                new_agents.append(instance)
            
            # Combine reused and new agents
            result_agents = []
            
            if reuse_existing and available:
                reused = available[:count]
                for inst in reused:
                    inst.is_busy = True
                    inst.last_used = datetime.now()
                result_agents.extend([inst.agent for inst in reused])
            
            result_agents.extend([inst.agent for inst in new_agents])
            
            self._update_metrics()
            
            logger.info(f"Spawned {len(new_agents)} new agents, total: {len(self._agents)}")
            return result_agents[:count]
    
    async def execute_parallel(
        self,
        tasks: list[tuple[ClaudeAgent, dict[str, Any]]],
        timeout: float | None = None,
        max_concurrent: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute tasks in parallel across agents.
        
        Args:
            tasks: List of (agent, task_spec) tuples
            timeout: Timeout for each task
            max_concurrent: Override max concurrent executions
        
        Returns:
            List of execution results
        """
        if not tasks:
            return []
        
        semaphore = asyncio.Semaphore(
            max_concurrent or self._max_concurrent
        )
        
        async def execute_single(
            agent: ClaudeAgent,
            task_spec: dict[str, Any],
            task_id: str,
        ) -> dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                
                try:
                    # Find agent instance
                    instance = self._get_agent_instance(agent.agent_id)
                    if instance:
                        instance.is_busy = True
                        instance.last_used = datetime.now()
                    
                    # Execute task
                    prompt = task_spec.get("prompt", "")
                    tools = task_spec.get("tools")
                    
                    result = await asyncio.wait_for(
                        agent.execute(prompt, tools=tools),
                        timeout=timeout or self._agent_timeout,
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Update instance metrics
                    if instance:
                        instance.task_count += 1
                        instance.total_execution_time += execution_time
                        instance.is_busy = False
                    
                    # Update global metrics
                    self._metrics.total_tasks_completed += 1
                    
                    return {
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "success": True,
                        "result": result,
                        "execution_time": execution_time,
                        "token_usage": result.get("usage", {}),
                    }
                    
                except asyncio.TimeoutError as e:
                    if instance:
                        instance.is_busy = False
                    self._metrics.total_tasks_failed += 1
                    
                    return {
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "success": False,
                        "error": f"Timeout: {e}",
                        "execution_time": time.time() - start_time,
                    }
                    
                except Exception as e:
                    if instance:
                        instance.is_busy = False
                    self._metrics.total_tasks_failed += 1
                    
                    return {
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "success": False,
                        "error": str(e),
                        "execution_time": time.time() - start_time,
                    }
        
        # Create execution tasks
        execution_tasks = [
            execute_single(agent, task_spec, f"task_{i}")
            for i, (agent, task_spec) in enumerate(tasks)
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        self._update_metrics()
        
        return processed_results
    
    async def scale_swarm(
        self,
        target_count: int,
        strategy: str = "gradual",
    ) -> int:
        """
        Scale the swarm to a target number of agents.
        
        Args:
            target_count: Target number of agents
            strategy: Scaling strategy ("gradual", "immediate", "lazy")
        
        Returns:
            New agent count
        
        Raises:
            ScalingError: If scaling fails
        """
        async with self._lock:
            current_count = len(self._agents)
            
            if target_count == current_count:
                return current_count
            
            if target_count > self._config.max_agents:
                raise ScalingError(
                    f"Target count {target_count} exceeds max agents {self._config.max_agents}",
                    error_code="SCALE_LIMIT_EXCEEDED"
                )
            
            try:
                if target_count > current_count:
                    # Scale up
                    to_add = target_count - current_count
                    
                    if strategy == "gradual":
                        # Add agents gradually
                        for _ in range(to_add):
                            await self.spawn_agents(1, reuse_existing=False)
                            await asyncio.sleep(0.1)  # Small delay between spawns
                    else:
                        # Add all at once
                        await self.spawn_agents(to_add, reuse_existing=False)
                    
                    logger.info(f"Scaled up swarm to {target_count} agents")
                    
                else:
                    # Scale down
                    to_remove = current_count - target_count
                    
                    # Find idle agents to remove
                    idle_agents = [
                        inst for inst in self._agents.values()
                        if not inst.is_busy
                    ]
                    
                    if len(idle_agents) < to_remove:
                        raise ScalingError(
                            f"Cannot scale down: only {len(idle_agents)} idle agents available",
                            error_code="SCALE_DOWN_ERROR"
                        )
                    
                    # Remove agents
                    for i, instance in enumerate(idle_agents[:to_remove]):
                        await self._terminate_agent(instance.instance_id)
                    
                    logger.info(f"Scaled down swarm to {target_count} agents")
                
                self._update_metrics()
                return len(self._agents)
                
            except Exception as e:
                raise ScalingError(
                    f"Scaling failed: {e}",
                    error_code="SCALE_ERROR"
                ) from e
    
    async def get_agent_metrics(self, agent_id: str | None = None) -> dict[str, Any]:
        """
        Get metrics for a specific agent or all agents.
        
        Args:
            agent_id: Specific agent ID, or None for all agents
        
        Returns:
            Metrics dictionary
        """
        if agent_id:
            instance = self._get_agent_instance(agent_id)
            if not instance:
                raise SwarmManagerError(f"Agent {agent_id} not found")
            
            return {
                "instance_id": instance.instance_id,
                "agent_id": agent_id,
                "task_count": instance.task_count,
                "total_execution_time": instance.total_execution_time,
                "average_execution_time": (
                    instance.total_execution_time / instance.task_count
                    if instance.task_count > 0
                    else 0
                ),
                "created_at": instance.created_at.isoformat(),
                "last_used": instance.last_used.isoformat(),
                "is_active": instance.is_active,
                "is_busy": instance.is_busy,
            }
        
        # Return metrics for all agents
        return {
            "swarm_id": self._swarm_id,
            "total_agents": len(self._agents),
            "active_agents": sum(1 for inst in self._agents.values() if inst.is_active),
            "busy_agents": sum(1 for inst in self._agents.values() if inst.is_busy),
            "idle_agents": sum(
                1 for inst in self._agents.values()
                if inst.is_active and not inst.is_busy
            ),
            "agents": [
                {
                    "agent_id": inst.agent.agent_id,
                    "task_count": inst.task_count,
                    "is_busy": inst.is_busy,
                }
                for inst in self._agents.values()
            ],
        }
    
    async def cleanup(self, graceful: bool = True) -> None:
        """
        Clean up all agents and resources.
        
        Args:
            graceful: Whether to wait for ongoing tasks
        """
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            if graceful:
                # Wait for busy agents to finish
                busy_agents = [
                    inst for inst in self._agents.values()
                    if inst.is_busy
                ]
                
                if busy_agents:
                    logger.info(f"Waiting for {len(busy_agents)} agents to finish...")
                    await asyncio.sleep(2)  # Give agents time to finish
            
            # Terminate all agents
            termination_tasks = []
            for instance_id in list(self._agents.keys()):
                task = asyncio.create_task(
                    self._terminate_agent(instance_id)
                )
                termination_tasks.append(task)
            
            if termination_tasks:
                await asyncio.gather(*termination_tasks, return_exceptions=True)
            
            self._agents.clear()
            
            logger.info(f"SwarmManager {self._swarm_id} cleaned up")
    
    async def _terminate_agent(self, instance_id: str) -> None:
        """Terminate a single agent instance."""
        if instance_id not in self._agents:
            return
        
        instance = self._agents[instance_id]
        instance.is_active = False
        
        try:
            await instance.agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent {instance_id}: {e}")
        
        del self._agents[instance_id]
    
    def _get_agent_instance(self, agent_id: str) -> AgentInstance | None:
        """Find agent instance by agent ID."""
        for instance in self._agents.values():
            if instance.agent.agent_id == agent_id:
                return instance
        return None
    
    def _update_metrics(self) -> None:
        """Update current metrics."""
        active = sum(1 for inst in self._agents.values() if inst.is_active)
        busy = sum(1 for inst in self._agents.values() if inst.is_busy)
        
        self._metrics = ResourceMetrics(
            active_agents=active,
            idle_agents=active - busy,
            busy_agents=busy,
            total_tasks_completed=self._metrics.total_tasks_completed,
            total_tasks_failed=self._metrics.total_tasks_failed,
        )
    
    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(60)  # Run every minute
                
                if self._shutdown_event.is_set():
                    break
                
                await self._perform_cleanup()
                
        except asyncio.CancelledError:
            logger.debug("Cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform cleanup of idle agents."""
        async with self._lock:
            now = datetime.now()
            to_remove: list[str] = []
            
            for instance_id, instance in self._agents.items():
                # Remove agents idle for more than 10 minutes
                if (
                    not instance.is_busy
                    and (now - instance.last_used).seconds > 600
                    and len(self._agents) > 2  # Keep at least 2 agents
                ):
                    to_remove.append(instance_id)
            
            for instance_id in to_remove[:5]:  # Remove max 5 per cleanup
                await self._terminate_agent(instance_id)
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove[:5])} idle agents")
    
    async def _metrics_loop(self) -> None:
        """Background task for collecting metrics."""
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                if self._shutdown_event.is_set():
                    break
                
                self._update_metrics()
                self._metrics_history.append(self._metrics)
                
                # Keep only last 100 metrics
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-100:]
                
        except asyncio.CancelledError:
            logger.debug("Metrics loop cancelled")
        except Exception as e:
            logger.error(f"Metrics loop error: {e}")
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.cleanup(graceful=True)
    
    async def __aenter__(self) -> SwarmManager:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
