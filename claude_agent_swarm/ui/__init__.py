"""
UI Components for Claude Agent Swarm Framework

Provides terminal dashboard and task board functionality.
"""

from .dashboard import (
    SwarmDashboard,
    AsyncSwarmDashboard,
    AgentDisplayInfo,
    SwarmStats,
    create_dashboard
)

from .task_board import (
    TaskBoard,
    Task,
    TaskResult,
    TaskStatus
)

__all__ = [
    # Dashboard components
    "SwarmDashboard",
    "AsyncSwarmDashboard",
    "AgentDisplayInfo",
    "SwarmStats",
    "create_dashboard",
    
    # Task board components
    "TaskBoard",
    "Task",
    "TaskResult",
    "TaskStatus"
]
