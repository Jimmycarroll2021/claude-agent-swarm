"""Agent implementation for Claude Agent Swarm.

This module provides the Agent class that wraps Claude API interactions
and provides capabilities for tool use, state management, and task execution.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from anthropic import AsyncAnthropic
import structlog

from claude_agent_swarm.tools import Tool, ToolResult

logger = structlog.get_logger()


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentConfig:
    """Configuration for an Agent."""
    
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 3


@dataclass
class AgentMessage:
    """Message from/to an agent."""
    
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Claude-powered agent for swarm operations.
    
    The Agent class wraps Claude API interactions and provides a consistent
    interface for task execution, tool use, and state management.
    
    Example:
        >>> config = AgentConfig(model="claude-3-5-sonnet-20241022")
        >>> agent = Agent(name="researcher", role="research", config=config)
        >>> result = await agent.execute("Research AI trends")
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the agent.
        
        Args:
            name: Unique agent name
            role: Agent role/purpose
            config: Optional agent configuration
            api_key: Optional Anthropic API key
        """
        self.name = name
        self.role = role
        self.config = config or AgentConfig()
        self._client = AsyncAnthropic(api_key=api_key)
        self._status = AgentStatus.IDLE
        self._tools: Dict[str, Tool] = {}
        self._message_history: List[AgentMessage] = []
        self._lock = asyncio.Lock()
        
        # Register default system prompt
        if not self.config.system_prompt:
            self.config.system_prompt = self._default_system_prompt()
        
        logger.info(
            "agent_initialized",
            agent_name=name,
            role=role,
            model=self.config.model,
        )
    
    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on role."""
        return f"""You are {self.name}, an AI agent specializing in {self.role}.

Your responsibilities:
1. Execute tasks related to your role efficiently and accurately
2. Use available tools when appropriate
3. Communicate clearly with other agents in the swarm
4. Report progress and any issues promptly

When working with other agents:
- Be collaborative and supportive
- Share relevant information
- Ask for clarification when needed
- Follow the swarm's coordination pattern

Current status: Ready to assist with {self.role} tasks.
"""
    
    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(
            "tool_registered",
            agent_name=self.name,
            tool_name=tool.name,
        )
    
    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(
                "tool_unregistered",
                agent_name=self.name,
                tool_name=tool_name,
            )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Task execution result
        """
        async with self._lock:
            if self._status == AgentStatus.BUSY:
                raise RuntimeError(f"Agent {self.name} is busy")
            
            self._status = AgentStatus.BUSY
            
            try:
                logger.info(
                    "agent_executing_task",
                    agent_name=self.name,
                    task=task[:100],
                )
                
                # Build messages
                messages = self._build_messages(task, context)
                
                # Prepare tools for API
                tools = self._prepare_tools()
                
                # Call Claude API
                response = await self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.config.system_prompt,
                    messages=messages,
                    tools=tools if tools else None,
                )
                
                # Process response
                result = await self._process_response(response)
                
                # Update message history
                self._message_history.append(AgentMessage(
                    role="user",
                    content=task,
                ))
                self._message_history.append(AgentMessage(
                    role="assistant",
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                ))
                
                self._status = AgentStatus.IDLE
                
                logger.info(
                    "agent_task_completed",
                    agent_name=self.name,
                    success=True,
                )
                
                return {
                    "success": True,
                    "agent": self.name,
                    "content": result.get("content", ""),
                    "tool_calls": result.get("tool_calls", []),
                    "metadata": result.get("metadata", {}),
                }
                
            except Exception as e:
                self._status = AgentStatus.ERROR
                logger.error(
                    "agent_task_failed",
                    agent_name=self.name,
                    error=str(e),
                )
                return {
                    "success": False,
                    "agent": self.name,
                    "error": str(e),
                }
    
    def _build_messages(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Build message list for API call.
        
        Args:
            task: Current task
            context: Optional context
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add context if provided
        if context:
            context_str = "\n".join(
                f"{key}: {value}" for key, value in context.items()
            )
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_str}\n\nTask: {task}",
            })
        else:
            messages.append({
                "role": "user",
                "content": task,
            })
        
        return messages
    
    def _prepare_tools(self) -> List[Dict[str, Any]]:
        """Prepare tools for Claude API.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]
    
    async def _process_response(self, response: Any) -> Dict[str, Any]:
        """Process Claude API response.
        
        Args:
            response: API response
            
        Returns:
            Processed result
        """
        content_parts = []
        tool_calls = []
        
        for content in response.content:
            if content.type == "text":
                content_parts.append(content.text)
            elif content.type == "tool_use":
                tool_calls.append({
                    "name": content.name,
                    "input": content.input,
                })
                
                # Execute tool if registered
                if content.name in self._tools:
                    tool = self._tools[content.name]
                    tool_result = await tool.execute(content.input)
                    content_parts.append(
                        f"\n[Tool {content.name} result: {tool_result}]\n"
                    )
        
        return {
            "content": "\n".join(content_parts),
            "tool_calls": tool_calls,
            "metadata": {
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status.
        
        Returns:
            Status information
        """
        return {
            "name": self.name,
            "role": self.role,
            "status": self._status.value,
            "model": self.config.model,
            "tools": list(self._tools.keys()),
            "message_count": len(self._message_history),
        }
    
    async def reset(self) -> None:
        """Reset agent state."""
        async with self._lock:
            self._message_history.clear()
            self._status = AgentStatus.IDLE
            logger.info("agent_reset", agent_name=self.name)
    
    async def terminate(self) -> None:
        """Terminate the agent."""
        async with self._lock:
            self._status = AgentStatus.TERMINATED
            await self._client.close()
            logger.info("agent_terminated", agent_name=self.name)
