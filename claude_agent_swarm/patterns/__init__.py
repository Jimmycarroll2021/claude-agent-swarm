"""
Orchestration Patterns for Claude Agent Swarm Framework.

This module provides abstract base classes and common interfaces for
implementing distributed orchestration patterns in multi-agent systems.

Patterns:
    - LeaderPattern: Central orchestrator with dynamic sub-agent delegation
    - SwarmPattern: Parallel processing with work distribution
    - PipelinePattern: Sequential multi-stage workflows
    - CouncilPattern: Multi-perspective analysis and consensus building

Example:
    >>> from patterns import LeaderPattern, SwarmPattern
    >>> leader = LeaderPattern(max_concurrent=5)
    >>> result = await leader.delegate(task, agent_pool)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)
import asyncio
import time
from contextlib import asynccontextmanager


# Type variables for generic patterns
T = TypeVar("T")
R = TypeVar("R")
AgentType = TypeVar("AgentType")


class PatternStatus(Enum):
    """Status enumeration for pattern execution states."""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()  # Completed with some failures
    TIMEOUT = auto()


class AgentCapability(Protocol):
    """Protocol defining agent capabilities for pattern matching."""
    
    @property
    def agent_id(self) -> str:
        """Unique identifier for the agent."""
        ...
    
    @property
    def capabilities(self) -> Set[str]:
        """Set of capability strings this agent possesses."""
        ...
    
    async def execute(self, task: Any) -> Any:
        """Execute a task and return results."""
        ...


@dataclass
class TaskResult(Generic[T]):
    """Container for task execution results with metadata."""
    
    task_id: str
    status: PatternStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    agent_id: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == PatternStatus.COMPLETED and self.error is None
    
    @property
    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status in (PatternStatus.FAILED, PatternStatus.TIMEOUT) or self.error is not None


@dataclass
class PatternMetrics:
    """Metrics collection for pattern execution."""
    
    pattern_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timed_out: int = 0
    total_execution_time: float = 0.0
    
    def record_completion(self, execution_time: float) -> None:
        """Record a task completion."""
        self.tasks_completed += 1
        self.total_execution_time += execution_time
    
    def record_failure(self) -> None:
        """Record a task failure."""
        self.tasks_failed += 1
    
    def record_timeout(self) -> None:
        """Record a task timeout."""
        self.tasks_timed_out += 1
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Total duration of pattern execution."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.tasks_completed + self.tasks_failed + self.tasks_timed_out
        if total == 0:
            return 0.0
        return (self.tasks_completed / total) * 100


class OrchestrationPattern(ABC, Generic[AgentType]):
    """
    Abstract base class for all orchestration patterns.
    
    Provides common interface and utilities for implementing distributed
    orchestration patterns with proper resource management, timeout handling,
    and telemetry integration.
    
    Type Parameters:
        AgentType: The type of agents this pattern orchestrates
    
    Attributes:
        max_concurrent: Maximum number of concurrent operations
        default_timeout: Default timeout for operations in seconds
        metrics: Pattern execution metrics
        semaphore: Asyncio semaphore for concurrency control
    
    Example:
        >>> class MyPattern(OrchestrationPattern[MyAgent]):
        ...     async def execute(self, tasks: List[Task]) -> List[Result]:
        ...         async with self._acquire_slot():
        ...             # Execute with concurrency limit
        ...             pass
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: float = 30.0,
        enable_telemetry: bool = True,
    ):
        """
        Initialize the orchestration pattern.
        
        Args:
            max_concurrent: Maximum concurrent operations (default: 10)
            default_timeout: Default operation timeout in seconds (default: 30)
            enable_telemetry: Whether to collect execution metrics (default: True)
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.enable_telemetry = enable_telemetry
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._metrics: Optional[PatternMetrics] = None
        self._status = PatternStatus.PENDING
        self._lock = asyncio.Lock()
    
    @property
    def status(self) -> PatternStatus:
        """Current execution status of the pattern."""
        return self._status
    
    @property
    def metrics(self) -> Optional[PatternMetrics]:
        """Execution metrics if telemetry is enabled."""
        return self._metrics
    
    def _init_metrics(self, pattern_name: str) -> None:
        """Initialize metrics collection."""
        if self.enable_telemetry:
            self._metrics = PatternMetrics(pattern_name=pattern_name)
    
    @asynccontextmanager
    async def _acquire_slot(self):
        """
        Context manager for acquiring a concurrency slot.
        
        Yields:
            None when slot is acquired
        
        Example:
            >>> async with self._acquire_slot():
            ...     await agent.execute(task)
        """
        async with self._semaphore:
            yield
    
    async def _execute_with_timeout(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> TaskResult[T]:
        """
        Execute a coroutine with timeout handling.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds (uses default if not specified)
            task_id: Optional task identifier for result tracking
        
        Returns:
            TaskResult containing execution result or error
        """
        task_id = task_id or f"task_{id(coro)}"
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            execution_time = time.time() - start_time
            
            if self._metrics:
                self._metrics.record_completion(execution_time)
            
            return TaskResult(
                task_id=task_id,
                status=PatternStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
            )
        
        except asyncio.TimeoutError as e:
            execution_time = time.time() - start_time
            
            if self._metrics:
                self._metrics.record_timeout()
            
            return TaskResult(
                task_id=task_id,
                status=PatternStatus.TIMEOUT,
                error=e,
                execution_time=execution_time,
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            if self._metrics:
                self._metrics.record_failure()
            
            return TaskResult(
                task_id=task_id,
                status=PatternStatus.FAILED,
                error=e,
                execution_time=execution_time,
            )
    
    async def _execute_parallel(
        self,
        tasks: List[Coroutine[Any, Any, T]],
        timeout: Optional[float] = None,
        return_exceptions: bool = True,
    ) -> List[TaskResult[T]]:
        """
        Execute multiple coroutines in parallel with concurrency control.
        
        Args:
            tasks: List of coroutines to execute
            timeout: Timeout per task in seconds
            return_exceptions: Whether to return exceptions in results
        
        Returns:
            List of TaskResult objects in original order
        """
        timeout = timeout or self.default_timeout
        
        async def bounded_execute(
            coro: Coroutine[Any, Any, T],
            idx: int,
        ) -> TaskResult[T]:
            async with self._acquire_slot():
                return await self._execute_with_timeout(
                    coro,
                    timeout=timeout,
                    task_id=f"parallel_task_{idx}",
                )
        
        # Create bounded task executions
        bounded_tasks = [
            bounded_execute(coro, idx)
            for idx, coro in enumerate(tasks)
        ]
        
        # Execute all with gather
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Handle any exceptions from gather itself
        processed_results: List[TaskResult[T]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=f"parallel_task_{idx}",
                    status=PatternStatus.FAILED,
                    error=result,
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _filter_successful_results(
        self,
        results: List[TaskResult[T]],
    ) -> List[T]:
        """
        Extract successful results, filtering out failures.
        
        Args:
            results: List of TaskResult objects
        
        Returns:
            List of successful result values
        """
        successful = []
        for r in results:
            if r.is_success and r.result is not None:
                successful.append(r.result)
        return successful
    
    def _has_partial_failures(self, results: List[TaskResult[T]]) -> bool:
        """Check if any results indicate partial failure."""
        return any(r.is_failure for r in results)
    
    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Main execution method for the pattern.
        
        This method must be implemented by all concrete pattern classes.
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        
        Returns:
            Pattern-specific result type
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Cleanup resources used by the pattern.
        
        Override to implement pattern-specific cleanup.
        """
        async with self._lock:
            if self._metrics:
                self._metrics.finalize()
            self._status = PatternStatus.COMPLETED
    
    async def __aenter__(self) -> "OrchestrationPattern[AgentType]":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.cleanup()


# Import concrete pattern implementations
from .base import Pattern, PatternConfig
from .leader import LeaderPattern, LeaderConfig
from .pipeline import PipelinePattern, PipelineConfig
from .swarm import SwarmPattern, SwarmPatternConfig
from .council import CouncilPattern, CouncilConfig

# Export all public classes
__all__ = [
    # Base classes
    "OrchestrationPattern",
    "Pattern",
    "PatternConfig",
    "PatternStatus",
    "AgentCapability",
    "TaskResult",
    "PatternMetrics",
    # Concrete patterns
    "LeaderPattern",
    "LeaderConfig",
    "PipelinePattern",
    "PipelineConfig",
    "SwarmPattern",
    "SwarmPatternConfig",
    "CouncilPattern",
    "CouncilConfig",
]
