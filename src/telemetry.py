"""
Telemetry System for Claude Agent Swarm Framework

Provides comprehensive metrics collection, token usage tracking,
execution time measurements, and cost calculations for the swarm.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path


class EventType(Enum):
    """Types of telemetry events."""
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    TOOL_CALL = "tool_call"
    TOKEN_USAGE = "token_usage"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    RATE_LIMIT_HIT = "rate_limit_hit"


@dataclass
class TokenUsage:
    """Tracks token usage for an entity."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int = 0, completion: int = 0) -> None:
        """Add token usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Tracks execution metrics for tasks/agents."""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self, success: bool = True, error: Optional[str] = None) -> None:
        """Stop timing and record result."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000 if self.start_time else 0
        self.success = success
        self.error_message = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: EventType
    timestamp: float
    agent_id: Optional[str] = None
    swarm_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "agent_id": self.agent_id,
            "swarm_id": self.swarm_id,
            "task_id": self.task_id,
            "data": self.data
        }


class TelemetryCollector:
    """
    Comprehensive telemetry collector for the Claude Agent Swarm.
    
    Collects metrics on:
    - Token usage per agent and swarm
    - Execution times
    - Success/failure rates
    - Tool call counts
    - Cost calculations
    - Rate limit events
    
    Example:
        >>> telemetry = TelemetryCollector()
        >>> telemetry.record_event(
        ...     EventType.AGENT_START,
        ...     agent_id="agent_1",
        ...     swarm_id="swarm_1",
        ...     data={"task": "code_review"}
        ... )
        >>> metrics = telemetry.get_metrics()
        >>> telemetry.export_metrics("metrics.json")
    """
    
    # Cost per 1K tokens (approximate, can be updated)
    DEFAULT_COSTS = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(
        self,
        model_costs: Optional[Dict[str, Dict[str, float]]] = None,
        max_events: int = 10000
    ):
        """
        Initialize the telemetry collector.
        
        Args:
            model_costs: Custom cost per 1K tokens for models
            max_events: Maximum events to keep in memory
        """
        self.model_costs = model_costs or self.DEFAULT_COSTS.copy()
        self.max_events = max_events
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Events storage
        self._events: List[TelemetryEvent] = []
        
        # Token usage by agent
        self._agent_tokens: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        
        # Token usage by swarm
        self._swarm_tokens: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        
        # Token usage by model
        self._model_tokens: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        
        # Execution metrics by agent
        self._agent_metrics: Dict[str, List[ExecutionMetrics]] = defaultdict(list)
        
        # Tool call counts
        self._tool_calls: Dict[str, int] = defaultdict(int)
        
        # Rate limit events
        self._rate_limit_events: List[Dict[str, Any]] = []
        
        # Active agents tracking
        self._active_agents: Dict[str, Dict[str, Any]] = {}
        
        # Workflow tracking
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks for real-time updates
        self._callbacks: List[Callable[[TelemetryEvent], None]] = []
        
        # Start time
        self._start_time = time.time()
    
    def record_event(
        self,
        event_type: EventType,
        agent_id: Optional[str] = None,
        swarm_id: Optional[str] = None,
        task_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> TelemetryEvent:
        """
        Record a telemetry event.
        
        Args:
            event_type: Type of event
            agent_id: Optional agent identifier
            swarm_id: Optional swarm identifier
            task_id: Optional task identifier
            data: Additional event data
            
        Returns:
            The recorded event
        """
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            agent_id=agent_id,
            swarm_id=swarm_id,
            task_id=task_id,
            data=data or {}
        )
        
        with self._lock:
            self._events.append(event)
            
            # Enforce max events limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
            
            # Process event for metrics
            self._process_event(event)
        
        # Notify callbacks (outside lock to prevent deadlocks)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass
        
        return event
    
    def _process_event(self, event: TelemetryEvent) -> None:
        """Process event and update internal metrics."""
        if event.event_type == EventType.TOKEN_USAGE:
            self._process_token_usage(event)
        elif event.event_type == EventType.TOOL_CALL:
            self._process_tool_call(event)
        elif event.event_type == EventType.AGENT_START:
            self._process_agent_start(event)
        elif event.event_type == EventType.AGENT_COMPLETE:
            self._process_agent_complete(event)
        elif event.event_type == EventType.AGENT_ERROR:
            self._process_agent_error(event)
        elif event.event_type == EventType.RATE_LIMIT_HIT:
            self._process_rate_limit(event)
        elif event.event_type == EventType.WORKFLOW_START:
            self._process_workflow_start(event)
        elif event.event_type == EventType.WORKFLOW_COMPLETE:
            self._process_workflow_complete(event)
    
    def _process_token_usage(self, event: TelemetryEvent) -> None:
        """Process token usage event."""
        data = event.data
        prompt = data.get("prompt_tokens", 0)
        completion = data.get("completion_tokens", 0)
        model = data.get("model", "unknown")
        
        # Update agent tokens
        if event.agent_id:
            self._agent_tokens[event.agent_id].add(prompt, completion)
        
        # Update swarm tokens
        if event.swarm_id:
            self._swarm_tokens[event.swarm_id].add(prompt, completion)
        
        # Update model tokens
        self._model_tokens[model].add(prompt, completion)
    
    def _process_tool_call(self, event: TelemetryEvent) -> None:
        """Process tool call event."""
        tool_name = event.data.get("tool_name", "unknown")
        self._tool_calls[tool_name] += 1
    
    def _process_agent_start(self, event: TelemetryEvent) -> None:
        """Process agent start event."""
        if event.agent_id:
            self._active_agents[event.agent_id] = {
                "start_time": event.timestamp,
                "task_id": event.task_id,
                "status": "running"
            }
    
    def _process_agent_complete(self, event: TelemetryEvent) -> None:
        """Process agent complete event."""
        if event.agent_id:
            # Update active agents
            if event.agent_id in self._active_agents:
                self._active_agents[event.agent_id]["status"] = "completed"
                self._active_agents[event.agent_id]["end_time"] = event.timestamp
            
            # Record execution metrics
            metrics = ExecutionMetrics()
            metrics.start_time = event.data.get("start_time", event.timestamp)
            metrics.end_time = event.timestamp
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.success = True
            self._agent_metrics[event.agent_id].append(metrics)
    
    def _process_agent_error(self, event: TelemetryEvent) -> None:
        """Process agent error event."""
        if event.agent_id:
            if event.agent_id in self._active_agents:
                self._active_agents[event.agent_id]["status"] = "error"
                self._active_agents[event.agent_id]["end_time"] = event.timestamp
                self._active_agents[event.agent_id]["error"] = event.data.get("error")
            
            # Record failed execution
            metrics = ExecutionMetrics()
            metrics.start_time = event.data.get("start_time", event.timestamp)
            metrics.end_time = event.timestamp
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.success = False
            metrics.error_message = event.data.get("error")
            self._agent_metrics[event.agent_id].append(metrics)
    
    def _process_rate_limit(self, event: TelemetryEvent) -> None:
        """Process rate limit event."""
        self._rate_limit_events.append({
            "timestamp": event.timestamp,
            "agent_id": event.agent_id,
            "swarm_id": event.swarm_id,
            "retry_after": event.data.get("retry_after")
        })
    
    def _process_workflow_start(self, event: TelemetryEvent) -> None:
        """Process workflow start event."""
        if event.swarm_id:
            self._active_workflows[event.swarm_id] = {
                "start_time": event.timestamp,
                "status": "running"
            }
    
    def _process_workflow_complete(self, event: TelemetryEvent) -> None:
        """Process workflow complete event."""
        if event.swarm_id and event.swarm_id in self._active_workflows:
            self._active_workflows[event.swarm_id]["status"] = "completed"
            self._active_workflows[event.swarm_id]["end_time"] = event.timestamp
    
    def record_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "claude-3-sonnet",
        agent_id: Optional[str] = None,
        swarm_id: Optional[str] = None
    ) -> None:
        """
        Record token usage.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name
            agent_id: Optional agent identifier
            swarm_id: Optional swarm identifier
        """
        self.record_event(
            EventType.TOKEN_USAGE,
            agent_id=agent_id,
            swarm_id=swarm_id,
            data={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model
            }
        )
    
    def record_tool_call(
        self,
        tool_name: str,
        agent_id: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Record a tool call.
        
        Args:
            tool_name: Name of the tool called
            agent_id: Optional agent identifier
            duration_ms: Optional duration of the tool call
        """
        self.record_event(
            EventType.TOOL_CALL,
            agent_id=agent_id,
            data={"tool_name": tool_name, "duration_ms": duration_ms}
        )
    
    def calculate_cost(self, model: str = "claude-3-sonnet") -> Dict[str, float]:
        """
        Calculate costs for a model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with cost breakdown
        """
        costs = self.model_costs.get(model, self.model_costs["claude-3-sonnet"])
        usage = self._model_tokens.get(model, TokenUsage())
        
        input_cost = (usage.prompt_tokens / 1000) * costs["input"]
        output_cost = (usage.completion_tokens / 1000) * costs["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": "USD"
        }
    
    def calculate_total_cost(self) -> Dict[str, float]:
        """
        Calculate total costs across all models.
        
        Returns:
            Dictionary with total cost breakdown
        """
        total = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0, "currency": "USD"}
        
        for model in self._model_tokens.keys():
            costs = self.calculate_cost(model)
            total["input_cost"] += costs["input_cost"]
            total["output_cost"] += costs["output_cost"]
            total["total_cost"] += costs["total_cost"]
        
        return total
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            total_executions = sum(len(m) for m in self._agent_metrics.values())
            successful_executions = sum(
                1 for metrics in self._agent_metrics.values()
                for m in metrics if m.success
            )
            failed_executions = total_executions - successful_executions
            
            # Calculate average execution time
            all_durations = [
                m.duration_ms
                for metrics in self._agent_metrics.values()
                for m in metrics
            ]
            avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
            
            # Total tokens
            total_tokens = TokenUsage()
            for usage in self._agent_tokens.values():
                total_tokens.add(usage.prompt_tokens, usage.completion_tokens)
            
            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "executions": {
                    "total": total_executions,
                    "successful": successful_executions,
                    "failed": failed_executions,
                    "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                    "average_duration_ms": avg_duration
                },
                "tokens": total_tokens.to_dict(),
                "tokens_by_agent": {
                    k: v.to_dict() for k, v in self._agent_tokens.items()
                },
                "tokens_by_swarm": {
                    k: v.to_dict() for k, v in self._swarm_tokens.items()
                },
                "tokens_by_model": {
                    k: v.to_dict() for k, v in self._model_tokens.items()
                },
                "tool_calls": dict(self._tool_calls),
                "total_tool_calls": sum(self._tool_calls.values()),
                "costs": self.calculate_total_cost(),
                "costs_by_model": {
                    model: self.calculate_cost(model)
                    for model in self._model_tokens.keys()
                },
                "rate_limit_events": len(self._rate_limit_events),
                "active_agents": len(self._active_agents),
                "active_workflows": len(self._active_workflows),
                "total_events": len(self._events)
            }
    
    def export_metrics(
        self,
        filepath: Optional[Union[str, Path]] = None,
        format: str = "json"
    ) -> str:
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path (optional)
            format: Export format ("json" or "prometheus")
            
        Returns:
            Path to exported file or exported content
        """
        metrics = self.get_metrics()
        
        if format == "json":
            content = json.dumps(metrics, indent=2)
            if filepath:
                Path(filepath).write_text(content)
                return str(filepath)
            return content
        
        elif format == "prometheus":
            content = self._to_prometheus_format(metrics)
            if filepath:
                Path(filepath).write_text(content)
                return str(filepath)
            return content
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _to_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Convert metrics to Prometheus exposition format."""
        lines = []
        
        # Execution metrics
        lines.append("# HELP swarm_executions_total Total number of executions")
        lines.append("# TYPE swarm_executions_total counter")
        lines.append(f'swarm_executions_total {{type="total"}} {metrics["executions"]["total"]}')
        lines.append(f'swarm_executions_total {{type="successful"}} {metrics["executions"]["successful"]}')
        lines.append(f'swarm_executions_total {{type="failed"}} {metrics["executions"]["failed"]}')
        
        # Token metrics
        lines.append("# HELP swarm_tokens_total Total token usage")
        lines.append("# TYPE swarm_tokens_total counter")
        lines.append(f'swarm_tokens_total {{type="prompt"}} {metrics["tokens"]["prompt_tokens"]}')
        lines.append(f'swarm_tokens_total {{type="completion"}} {metrics["tokens"]["completion_tokens"]}')
        lines.append(f'swarm_tokens_total {{type="total"}} {metrics["tokens"]["total_tokens"]}')
        
        # Tool calls
        lines.append("# HELP swarm_tool_calls_total Total tool calls")
        lines.append("# TYPE swarm_tool_calls_total counter")
        lines.append(f'swarm_tool_calls_total {metrics["total_tool_calls"]}')
        
        # Cost metrics
        lines.append("# HELP swarm_cost_dollars Total cost in dollars")
        lines.append("# TYPE swarm_cost_dollars gauge")
        lines.append(f'swarm_cost_dollars {{type="total"}} {metrics["costs"]["total_cost"]:.6f}')
        
        # Active agents
        lines.append("# HELP swarm_active_agents Number of active agents")
        lines.append("# TYPE swarm_active_agents gauge")
        lines.append(f'swarm_active_agents {metrics["active_agents"]}')
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics and clear stored data."""
        with self._lock:
            self._events.clear()
            self._agent_tokens.clear()
            self._swarm_tokens.clear()
            self._model_tokens.clear()
            self._agent_metrics.clear()
            self._tool_calls.clear()
            self._rate_limit_events.clear()
            self._active_agents.clear()
            self._active_workflows.clear()
            self._start_time = time.time()
    
    def register_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """
        Register a callback for real-time event notifications.
        
        Args:
            callback: Function to call when events are recorded
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """
        Unregister a callback.
        
        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent-specific metrics
        """
        with self._lock:
            metrics = self._agent_metrics.get(agent_id, [])
            tokens = self._agent_tokens.get(agent_id, TokenUsage())
            
            if not metrics:
                return {
                    "agent_id": agent_id,
                    "executions": 0,
                    "success_rate": 0,
                    "tokens": tokens.to_dict()
                }
            
            successful = sum(1 for m in metrics if m.success)
            
            return {
                "agent_id": agent_id,
                "executions": len(metrics),
                "successful": successful,
                "failed": len(metrics) - successful,
                "success_rate": successful / len(metrics),
                "average_duration_ms": sum(m.duration_ms for m in metrics) / len(metrics),
                "tokens": tokens.to_dict()
            }
    
    def get_swarm_metrics(self, swarm_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific swarm.
        
        Args:
            swarm_id: Swarm identifier
            
        Returns:
            Swarm-specific metrics
        """
        with self._lock:
            tokens = self._swarm_tokens.get(swarm_id, TokenUsage())
            workflow = self._active_workflows.get(swarm_id, {})
            
            return {
                "swarm_id": swarm_id,
                "tokens": tokens.to_dict(),
                "workflow_status": workflow.get("status", "unknown"),
                "start_time": workflow.get("start_time"),
                "agents": [
                    aid for aid, info in self._active_agents.items()
                    if info.get("swarm_id") == swarm_id
                ]
            }
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TelemetryEvent]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type
            agent_id: Filter by agent ID
            limit: Maximum number of events
            
        Returns:
            List of matching events
        """
        with self._lock:
            events = self._events
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if agent_id:
                events = [e for e in events if e.agent_id == agent_id]
            
            return events[-limit:]


# Global telemetry instance for convenience
_global_telemetry: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    """Get or create the global telemetry instance."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryCollector()
    return _global_telemetry


def set_telemetry(telemetry: TelemetryCollector) -> None:
    """Set the global telemetry instance."""
    global _global_telemetry
    _global_telemetry = telemetry
