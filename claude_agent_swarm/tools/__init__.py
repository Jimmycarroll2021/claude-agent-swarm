"""
Tool system for Claude Agent Swarm.

Provides base classes, registries, and executors for agent tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union
from enum import Enum
import asyncio
import json
import time
from datetime import datetime


class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool arguments are invalid."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found in the registry."""
    pass


class ToolAccessDeniedError(ToolError):
    """Raised when an agent doesn't have permission to use a tool."""
    pass


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tool_name: str = ""
    agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "agent_id": self.agent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create result from dictionary."""
        return cls(
            success=data["success"],
            data=data.get("data"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_name=data.get("tool_name", ""),
            agent_id=data.get("agent_id")
        )


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "returns": self.returns,
            "examples": self.examples
        }
    
    def to_claude_schema(self) -> Dict[str, Any]:
        """Convert to Claude function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            }
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._schema: Optional[ToolSchema] = None
        self._execution_count = 0
        self._total_execution_time_ms = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool's schema definition."""
        pass
    
    def validate_args(self, args: Dict[str, Any]) -> List[str]:
        """Validate tool arguments against schema."""
        schema = self.get_schema()
        errors = []
        
        # Check required parameters
        for param in schema.required:
            if param not in args:
                errors.append(f"Missing required parameter: {param}")
        
        # Check parameter types
        for param, value in args.items():
            if param in schema.parameters:
                param_schema = schema.parameters[param]
                param_type = param_schema.get("type")
                
                if param_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter '{param}' must be a string")
                elif param_type == "integer" and not isinstance(value, int):
                    errors.append(f"Parameter '{param}' must be an integer")
                elif param_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{param}' must be a number")
                elif param_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter '{param}' must be a boolean")
                elif param_type == "array" and not isinstance(value, list):
                    errors.append(f"Parameter '{param}' must be an array")
                elif param_type == "object" and not isinstance(value, dict):
                    errors.append(f"Parameter '{param}' must be an object")
        
        return errors
    
    async def execute_with_validation(self, agent_id: Optional[str] = None, **kwargs) -> ToolResult:
        """Execute tool with argument validation and metrics."""
        start_time = time.time()
        
        # Validate arguments
        errors = self.validate_args(kwargs)
        if errors:
            return ToolResult(
                success=False,
                error=f"Validation failed: {'; '.join(errors)}",
                tool_name=self.name,
                agent_id=agent_id,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Execute the tool
            result = await self.execute(**kwargs)
            result.tool_name = self.name
            result.agent_id = agent_id
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._execution_count += 1
            self._total_execution_time_ms += result.execution_time_ms
            
            return result
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Execution error: {str(e)}",
                tool_name=self.name,
                agent_id=agent_id,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tool execution metrics."""
        avg_time = (self._total_execution_time_ms / self._execution_count 
                   if self._execution_count > 0 else 0)
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time_ms,
            "average_execution_time_ms": avg_time
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._access_control: Dict[str, List[str]] = {}  # agent_id -> [tool_names]
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return self._tools[name]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.get_schema().to_claude_schema() for tool in self._tools.values()]
    
    def grant_access(self, agent_id: str, tool_names: List[str]) -> None:
        """Grant an agent access to specific tools."""
        self._access_control[agent_id] = tool_names
    
    def revoke_access(self, agent_id: str) -> None:
        """Revoke all tool access for an agent."""
        if agent_id in self._access_control:
            del self._access_control[agent_id]
    
    def check_access(self, agent_id: str, tool_name: str) -> bool:
        """Check if an agent has access to a tool."""
        # If no access control defined, allow all
        if agent_id not in self._access_control:
            return True
        return tool_name in self._access_control[agent_id]
    
    def get_agent_tools(self, agent_id: str) -> List[BaseTool]:
        """Get all tools an agent has access to."""
        if agent_id not in self._access_control:
            return list(self._tools.values())
        return [self._tools[name] for name in self._access_control[agent_id] 
                if name in self._tools]


class ToolExecutor:
    """Executor for running tools with proper error handling."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    async def execute(
        self, 
        tool_name: str, 
        args: Dict[str, Any], 
        agent_id: Optional[str] = None
    ) -> ToolResult:
        """Execute a tool with access control."""
        # Check access
        if not self.registry.check_access(agent_id, tool_name):
            raise ToolAccessDeniedError(
                f"Agent '{agent_id}' does not have access to tool '{tool_name}'"
            )
        
        # Get tool
        tool = self.registry.get(tool_name)
        
        # Execute
        result = await tool.execute_with_validation(agent_id=agent_id, **args)
        
        # Record in history
        self._execution_history.append({
            "tool_name": tool_name,
            "agent_id": agent_id,
            "args": args,
            "result": result.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Trim history if needed
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
        
        return result
    
    def get_execution_history(
        self, 
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history with optional filtering."""
        history = self._execution_history
        
        if agent_id:
            history = [h for h in history if h["agent_id"] == agent_id]
        
        if tool_name:
            history = [h for h in history if h["tool_name"] == tool_name]
        
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history = []


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """Register a tool in the global registry."""
    get_global_registry().register(tool)


def get_tool(name: str) -> BaseTool:
    """Get a tool from the global registry."""
    return get_global_registry().get(name)


__all__ = [
    "ToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolAccessDeniedError",
    "ToolResult",
    "ToolSchema",
    "BaseTool",
    "ToolRegistry",
    "ToolExecutor",
    "get_global_registry",
    "register_tool",
    "get_tool"
]
