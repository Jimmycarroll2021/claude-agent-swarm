"""
Task Board for Claude Agent Swarm Framework

Provides task status tracking, markdown-based status file generation,
and state persistence for swarm task management.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    
    def get_icon(self) -> str:
        """Get status icon."""
        icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫",
            TaskStatus.BLOCKED: "⛔"
        }
        return icons.get(self, "❓")
    
    def get_color(self) -> str:
        """Get status color for markdown."""
        colors = {
            TaskStatus.PENDING: "#FFA500",
            TaskStatus.IN_PROGRESS: "#1E90FF",
            TaskStatus.COMPLETED: "#32CD32",
            TaskStatus.FAILED: "#DC143C",
            TaskStatus.CANCELLED: "#808080",
            TaskStatus.BLOCKED: "#8B0000"
        }
        return colors.get(self, "#000000")


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Task:
    """Represents a task in the task board."""
    task_id: str
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    agent_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more important
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[TaskResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None  # in seconds
    actual_duration: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.task_id:
            raise ValueError("Task ID is required")
        if not self.name:
            raise ValueError("Task name is required")
    
    def start(self, agent_id: Optional[str] = None) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = time.time()
        if agent_id:
            self.agent_id = agent_id
    
    def complete(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.completed_at = time.time()
        self.result = result
        if self.started_at:
            self.actual_duration = self.completed_at - self.started_at
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.result = TaskResult(success=False, error=error)
        if self.started_at:
            self.actual_duration = self.completed_at - self.started_at
    
    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
    
    def block(self) -> None:
        """Block the task (waiting for dependencies)."""
        self.status = TaskStatus.BLOCKED
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.actual_duration:
            return self.actual_duration
        if self.started_at:
            return time.time() - self.started_at
        return None
    
    def get_duration_str(self) -> str:
        """Get formatted duration string."""
        duration = self.get_duration()
        if duration is None:
            return "-"
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "agent_id": self.agent_id,
            "parent_task_id": self.parent_task_id,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
            "tags": self.tags,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Create task from dictionary."""
        task_data = data.copy()
        task_data["status"] = TaskStatus(task_data.get("status", "pending"))
        if task_data.get("result"):
            task_data["result"] = TaskResult.from_dict(task_data["result"])
        return cls(**{k: v for k, v in task_data.items() if k in cls.__dataclass_fields__})


class TaskBoard:
    """
    Task board for tracking and managing swarm tasks.
    
    Provides:
    - Task creation and status tracking
    - Agent assignment tracking
    - Markdown-based status file generation
    - State persistence
    - Dependency management
    
    Example:
        >>> board = TaskBoard()
        >>> board.add_task("task_1", "Implement feature", priority=8)
        >>> board.update_task("task_1", status=TaskStatus.IN_PROGRESS, agent_id="agent_1")
        >>> board.save_to_file("tasks.md")
    """
    
    def __init__(
        self,
        board_id: Optional[str] = None,
        name: str = "Task Board",
        description: str = ""
    ):
        """
        Initialize the task board.
        
        Args:
            board_id: Unique board identifier
            name: Board name
            description: Board description
        """
        self.board_id = board_id or f"board_{int(time.time())}"
        self.name = name
        self.description = description
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # Task storage
        self._tasks: Dict[str, Task] = {}
        self._tasks_by_agent: Dict[str, List[str]] = defaultdict(list)
        self._tasks_by_status: Dict[TaskStatus, List[str]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Persistence
        self._autosave_path: Optional[Path] = None
        self._autosave_enabled = False
    
    def add_task(
        self,
        task_id: str,
        name: str,
        description: str = "",
        status: TaskStatus = TaskStatus.PENDING,
        agent_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 5,
        tags: Optional[List[str]] = None,
        estimated_duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Add a new task to the board.
        
        Args:
            task_id: Unique task identifier
            name: Task name
            description: Task description
            status: Initial task status
            agent_id: Assigned agent ID
            parent_task_id: Parent task ID for subtasks
            dependencies: List of task IDs this task depends on
            priority: Task priority (1-10)
            tags: List of tags
            estimated_duration: Estimated duration in seconds
            metadata: Additional metadata
            
        Returns:
            Created task
            
        Raises:
            ValueError: If task_id already exists
        """
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            task = Task(
                task_id=task_id,
                name=name,
                description=description,
                status=status,
                agent_id=agent_id,
                parent_task_id=parent_task_id,
                dependencies=dependencies or [],
                priority=priority,
                tags=tags or [],
                estimated_duration=estimated_duration,
                metadata=metadata or {}
            )
            
            self._tasks[task_id] = task
            self._update_indexes(task)
            self.updated_at = time.time()
            
            if self._autosave_enabled:
                self._autosave()
            
            return task
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        agent_id: Optional[str] = None,
        result: Optional[TaskResult] = None,
        progress: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """
        Update an existing task.
        
        Args:
            task_id: Task identifier
            status: New status
            agent_id: New agent assignment
            result: Task result
            progress: Progress percentage (0-100)
            metadata: Additional metadata to merge
            
        Returns:
            Updated task or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            
            # Update status
            if status:
                old_status = task.status
                if status == TaskStatus.IN_PROGRESS and task.status != TaskStatus.IN_PROGRESS:
                    task.start(agent_id)
                elif status == TaskStatus.COMPLETED and result:
                    task.complete(result)
                elif status == TaskStatus.FAILED:
                    if result and result.error:
                        task.fail(result.error)
                    else:
                        task.fail("Unknown error")
                else:
                    task.status = status
                
                # Update indexes
                if old_status != task.status:
                    self._tasks_by_status[old_status].remove(task_id)
                    self._tasks_by_status[task.status].append(task_id)
            
            # Update agent
            if agent_id and agent_id != task.agent_id:
                if task.agent_id:
                    self._tasks_by_agent[task.agent_id].remove(task_id)
                task.agent_id = agent_id
                self._tasks_by_agent[agent_id].append(task_id)
            
            # Update result
            if result:
                task.result = result
            
            # Update progress in metadata
            if progress is not None:
                task.metadata["progress"] = progress
            
            # Merge metadata
            if metadata:
                task.metadata.update(metadata)
            
            self.updated_at = time.time()
            
            if self._autosave_enabled:
                self._autosave()
            
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        agent_id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Task]:
        """
        Get tasks filtered by criteria.
        
        Args:
            status: Filter by status
            agent_id: Filter by agent ID
            tag: Filter by tag
            
        Returns:
            List of matching tasks
        """
        with self._lock:
            tasks = list(self._tasks.values())
            
            if status:
                tasks = [t for t in tasks if t.status == status]
            
            if agent_id:
                tasks = [t for t in tasks if t.agent_id == agent_id]
            
            if tag:
                tasks = [t for t in tasks if tag in t.tags]
            
            return sorted(tasks, key=lambda t: (-t.priority, t.created_at))
    
    def get_board(self) -> Dict[str, Any]:
        """
        Get complete board state.
        
        Returns:
            Dictionary with board state
        """
        with self._lock:
            return {
                "board_id": self.board_id,
                "name": self.name,
                "description": self.description,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
                "summary": self._get_summary()
            }
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get board summary statistics."""
        status_counts = {status: 0 for status in TaskStatus}
        for task in self._tasks.values():
            status_counts[task.status] += 1
        
        total_duration = sum(
            (t.actual_duration or 0) for t in self._tasks.values()
        )
        
        return {
            "total_tasks": len(self._tasks),
            "by_status": {s.value: c for s, c in status_counts.items()},
            "completion_rate": (
                status_counts[TaskStatus.COMPLETED] / len(self._tasks)
                if self._tasks else 0
            ),
            "total_duration_seconds": total_duration,
            "agents_assigned": len(self._tasks_by_agent)
        }
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the board.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            del self._tasks[task_id]
            
            if task.agent_id:
                self._tasks_by_agent[task.agent_id].remove(task_id)
            
            self._tasks_by_status[task.status].remove(task_id)
            
            self.updated_at = time.time()
            
            if self._autosave_enabled:
                self._autosave()
            
            return True
    
    def _update_indexes(self, task: Task) -> None:
        """Update internal indexes for a task."""
        self._tasks_by_status[task.status].append(task.task_id)
        if task.agent_id:
            self._tasks_by_agent[task.agent_id].append(task.task_id)
    
    def enable_autosave(self, path: Union[str, Path]) -> None:
        """
        Enable automatic saving to file.
        
        Args:
            path: Path to save file
        """
        self._autosave_path = Path(path)
        self._autosave_enabled = True
    
    def disable_autosave(self) -> None:
        """Disable automatic saving."""
        self._autosave_enabled = False
    
    def _autosave(self) -> None:
        """Perform autosave."""
        if self._autosave_path:
            try:
                self.save_to_file(self._autosave_path)
            except Exception:
                pass  # Silently fail autosave
    
    def save_to_file(
        self,
        filepath: Union[str, Path],
        format: str = "markdown"
    ) -> Path:
        """
        Save board to file.
        
        Args:
            filepath: Output file path
            format: Output format ("markdown", "json")
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "markdown":
            content = self.to_markdown()
        elif format == "json":
            content = json.dumps(self.get_board(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        filepath.write_text(content, encoding="utf-8")
        return filepath
    
    def load_from_file(self, filepath: Union[str, Path]) -> TaskBoard:
        """
        Load board from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded task board
        """
        filepath = Path(filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        
        self.board_id = data.get("board_id", self.board_id)
        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)
        self.created_at = data.get("created_at", self.created_at)
        self.updated_at = data.get("updated_at", self.updated_at)
        
        self._tasks.clear()
        self._tasks_by_agent.clear()
        self._tasks_by_status.clear()
        
        for task_id, task_data in data.get("tasks", {}).items():
            task = Task.from_dict(task_data)
            self._tasks[task_id] = task
            self._update_indexes(task)
        
        return self
    
    def to_markdown(self) -> str:
        """
        Generate markdown representation of the board.
        
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append(f"# {self.name}")
        lines.append("")
        if self.description:
            lines.append(self.description)
            lines.append("")
        
        # Summary
        summary = self._get_summary()
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Tasks:** {summary['total_tasks']}")
        lines.append(f"- **Completion Rate:** {summary['completion_rate']*100:.1f}%")
        lines.append(f"- **Agents Assigned:** {summary['agents_assigned']}")
        lines.append("")
        
        # Status breakdown
        lines.append("### Status Breakdown")
        lines.append("")
        for status in TaskStatus:
            count = summary['by_status'].get(status.value, 0)
            icon = status.get_icon()
            lines.append(f"- {icon} **{status.value.replace('_', ' ').title()}:** {count}")
        lines.append("")
        
        # Tasks by status
        for status in TaskStatus:
            tasks = self.get_tasks(status=status)
            if tasks:
                lines.append(f"## {status.get_icon()} {status.value.replace('_', ' ').title()} ({len(tasks)})")
                lines.append("")
                
                for task in tasks:
                    lines.extend(self._task_to_markdown(task))
                    lines.append("")
        
        # Agent assignments
        if self._tasks_by_agent:
            lines.append("## Agent Assignments")
            lines.append("")
            
            for agent_id, task_ids in sorted(self._tasks_by_agent.items()):
                lines.append(f"### {agent_id}")
                lines.append("")
                for task_id in task_ids:
                    task = self._tasks.get(task_id)
                    if task:
                        status_icon = task.status.get_icon()
                        lines.append(f"- {status_icon} [{task.name}](#{task.task_id})")
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Last updated: {datetime.fromtimestamp(self.updated_at).isoformat()}*")
        
        return "\n".join(lines)
    
    def _task_to_markdown(self, task: Task) -> List[str]:
        """Convert a task to markdown lines."""
        lines = []
        
        # Task header with anchor
        lines.append(f"<a id=\"{task.task_id}\"></a>")
        lines.append(f"### {task.name}")
        lines.append("")
        
        # Task metadata
        meta = []
        meta.append(f"**ID:** `{task.task_id}`")
        meta.append(f"**Status:** {task.status.value}")
        if task.agent_id:
            meta.append(f"**Agent:** {task.agent_id}")
        meta.append(f"**Priority:** {task.priority}/10")
        
        lines.append(" | ".join(meta))
        lines.append("")
        
        # Description
        if task.description:
            lines.append(task.description)
            lines.append("")
        
        # Tags
        if task.tags:
            lines.append(f"**Tags:** {', '.join(task.tags)}")
            lines.append("")
        
        # Dependencies
        if task.dependencies:
            lines.append(f"**Dependencies:** {', '.join(task.dependencies)}")
            lines.append("")
        
        # Duration
        if task.estimated_duration:
            lines.append(f"**Estimated Duration:** {self._format_duration(task.estimated_duration)}")
        if task.actual_duration:
            lines.append(f"**Actual Duration:** {self._format_duration(task.actual_duration)}")
        
        # Progress
        if "progress" in task.metadata:
            progress = task.metadata["progress"]
            bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            lines.append(f"**Progress:** {bar} {progress:.0f}%")
            lines.append("")
        
        # Result
        if task.result:
            lines.append("#### Result")
            lines.append("")
            if task.result.success:
                lines.append("✅ **Success**")
            else:
                lines.append("❌ **Failed**")
            
            if task.result.output:
                lines.append("")
                lines.append("**Output:**")
                lines.append("```")
                lines.append(task.result.output[:500])  # Limit output
                if len(task.result.output) > 500:
                    lines.append("... (truncated)")
                lines.append("```")
            
            if task.result.error:
                lines.append("")
                lines.append(f"**Error:** {task.result.error}")
            
            if task.result.artifacts:
                lines.append("")
                lines.append(f"**Artifacts:** {', '.join(task.result.artifacts)}")
        
        return lines
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute (dependencies met).
        
        Returns:
            List of ready tasks
        """
        with self._lock:
            ready = []
            for task in self._tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Check if all dependencies are completed
                deps_met = all(
                    self._tasks.get(dep_id, Task(dep_id, "")).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if deps_met:
                    ready.append(task)
            
            return sorted(ready, key=lambda t: -t.priority)
    
    def get_agent_workload(self, agent_id: str) -> Dict[str, Any]:
        """
        Get workload information for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Workload statistics
        """
        with self._lock:
            tasks = self.get_tasks(agent_id=agent_id)
            
            return {
                "agent_id": agent_id,
                "total_tasks": len(tasks),
                "in_progress": len([t for t in tasks if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
                "pending": len([t for t in tasks if t.status == TaskStatus.PENDING])
            }


# Import defaultdict at module level for type hints
from collections import defaultdict
