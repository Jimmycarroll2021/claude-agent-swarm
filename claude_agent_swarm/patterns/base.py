"""Base pattern interface for Claude Agent Swarm.

This module provides the base class and interfaces for implementing
swarm coordination patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import structlog

from claude_agent_swarm.agent import Agent
from claude_agent_swarm.state_manager import StateManager
from claude_agent_swarm.message_queue import MessageQueue
from claude_agent_swarm.telemetry import TelemetryCollector

logger = structlog.get_logger()


class PatternStatus(Enum):
    """Pattern execution status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    EXECUTING = "executing"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class PatternConfig:
    """Configuration for a swarm pattern."""
    
    name: str = "default"
    task_timeout: float = 300.0
    max_iterations: int = 10
    enable_coordination: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pattern(ABC):
    """Base class for swarm coordination patterns.
    
    The Pattern class provides the interface for implementing different
    swarm coordination strategies like leader-follower, pipeline, etc.
    
    Example:
        >>> class MyPattern(Pattern):
        ...     async def execute(self, task: str, context: Dict) -> Dict:
        ...         # Implementation
        ...         pass
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        config: PatternConfig,
        state_manager: Optional[StateManager] = None,
        message_queue: Optional[MessageQueue] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        """Initialize the pattern.
        
        Args:
            name: Pattern name
            agents: List of agents in the swarm
            config: Pattern configuration
            state_manager: Optional state manager
            message_queue: Optional message queue
            telemetry: Optional telemetry collector
        """
        self.name = name
        self.agents = agents
        self.config = config
        self._state_manager = state_manager
        self._message_queue = message_queue
        self._telemetry = telemetry
        
        self._status = PatternStatus.INITIALIZING
        self._created_at = datetime.now()
        self._execution_count = 0
        
        logger.info(
            "pattern_initialized",
            pattern_name=name,
            pattern_type=self.__class__.__name__,
            agent_count=len(agents),
        )
    
    @property
    def status(self) -> PatternStatus:
        """Get current pattern status."""
        return self._status
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using this pattern.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Execution result
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize the pattern. Override for custom initialization."""
        self._status = PatternStatus.ACTIVE
        
        # Initialize all agents
        for agent in self.agents:
            # Agent initialization if needed
            pass
        
        logger.info("pattern_ready", pattern_name=self.name)
    
    async def terminate(self) -> None:
        """Terminate the pattern and cleanup resources."""
        self._status = PatternStatus.TERMINATING
        
        # Terminate all agents
        for agent in self.agents:
            await agent.terminate()
        
        self._status = PatternStatus.TERMINATED
        
        logger.info(
            "pattern_terminated",
            pattern_name=self.name,
            execution_count=self._execution_count,
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get pattern status information.
        
        Returns:
            Status dictionary
        """
        agent_statuses = []
        for agent in self.agents:
            agent_statuses.append(await agent.get_status())
        
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "status": self._status.value,
            "agent_count": len(self.agents),
            "agents": agent_statuses,
            "created_at": self._created_at.isoformat(),
            "execution_count": self._execution_count,
            "config": {
                "task_timeout": self.config.task_timeout,
                "max_iterations": self.config.max_iterations,
            },
        }
    
    def _track_execution(self, operation: str) -> Any:
        """Track execution with telemetry.
        
        Args:
            operation: Operation name
            
        Returns:
            Telemetry context manager
        """
        if self._telemetry:
            return self._telemetry.track_operation(
                operation,
                pattern=self.__class__.__name__,
                swarm=self.name,
            )
        
        # Return a no-op context manager if no telemetry
        from contextlib import nullcontext
        return nullcontext()
