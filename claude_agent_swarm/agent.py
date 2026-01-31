"""Agent implementation for Claude Agent Swarm.

This module provides the ClaudeAgent class that wraps Claude API interactions
and provides capabilities for tool use, state management, and task execution.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from anthropic import AsyncAnthropic
import structlog

from .models import ClaudeModel, TokenUsage, AgentConfig
from .tools import BaseTool, ToolResult

logger = structlog.get_logger()


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentMessage:
    """Message from/to an agent."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaudeAgent:
    """Claude-powered agent for swarm operations.

    The ClaudeAgent class wraps Claude API interactions and provides a consistent
    interface for task execution, tool use, and state management.

    Example:
        >>> agent = await ClaudeAgent.create(
        ...     model="claude-3-7-sonnet-20250219",
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> result = await agent.execute("Research AI trends")
    """

    def __init__(
        self,
        agent_id: str,
        model: ClaudeModel = "claude-3-7-sonnet-20250219",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        """Initialize the agent. Use `create()` for async initialization.

        Args:
            agent_id: Unique agent identifier
            model: Claude model to use
            system_prompt: System prompt for the agent
            max_tokens: Maximum tokens for responses
            temperature: Sampling temperature
            api_key: Optional Anthropic API key
            name: Optional human-readable name for the agent
            role: Optional role/specialization for the agent
        """
        self._agent_id = agent_id
        self._model = model
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._name = name or f"agent_{agent_id[:8]}"
        self._role = role or "general"

        self._client: Optional[AsyncAnthropic] = None
        self._status = AgentStatus.IDLE
        self._tools: Dict[str, BaseTool] = {}
        self._message_history: List[AgentMessage] = []
        self._token_usage = TokenUsage()
        self._lock = asyncio.Lock()
        self._initialized = False

    @classmethod
    async def create(
        cls,
        model: ClaudeModel = "claude-3-7-sonnet-20250219",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> "ClaudeAgent":
        """Async factory method to create and initialize a ClaudeAgent.

        Args:
            model: Claude model to use
            system_prompt: System prompt for the agent
            max_tokens: Maximum tokens for responses
            temperature: Sampling temperature
            api_key: Optional Anthropic API key
            name: Optional human-readable name for the agent
            role: Optional role/specialization for the agent

        Returns:
            An initialized ClaudeAgent instance
        """
        agent_id = str(uuid4())
        instance = cls(
            agent_id=agent_id,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            name=name,
            role=role,
        )

        instance._client = AsyncAnthropic(api_key=instance._api_key)
        instance._initialized = True

        logger.info(
            "agent_created",
            agent_id=agent_id,
            model=model,
        )

        return instance

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return """You are a helpful AI assistant working as part of an agent swarm.

Your responsibilities:
1. Execute tasks efficiently and accurately
2. Use available tools when appropriate
3. Communicate clearly with other agents
4. Report progress and any issues promptly

Be collaborative, share relevant information, and follow the swarm's coordination pattern.
"""

    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self._agent_id

    @property
    def name(self) -> str:
        """Get the agent's human-readable name."""
        return self._name

    @property
    def role(self) -> str:
        """Get the agent's role/specialization."""
        return self._role

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status

    @property
    def token_usage(self) -> TokenUsage:
        """Get token usage for this agent."""
        return self._token_usage

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for the agent to use.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(
            "tool_registered",
            agent_id=self._agent_id,
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
                agent_id=self._agent_id,
                tool_name=tool_name,
            )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[BaseTool]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a task.

        Args:
            task: Task description
            context: Optional task context
            tools: Optional additional tools to use
            timeout: Optional timeout override

        Returns:
            Task execution result
        """
        if not self._initialized or not self._client:
            raise RuntimeError("Agent not initialized. Use ClaudeAgent.create()")

        async with self._lock:
            if self._status == AgentStatus.BUSY:
                raise RuntimeError(f"Agent {self._agent_id} is busy")

            self._status = AgentStatus.BUSY

            try:
                logger.info(
                    "agent_executing_task",
                    agent_id=self._agent_id,
                    task=task[:100],
                )

                # Register additional tools temporarily
                temp_tools = tools or []
                for tool in temp_tools:
                    if tool.name not in self._tools:
                        self._tools[tool.name] = tool

                # Build messages
                messages = self._build_messages(task, context)

                # Prepare tools for API
                tool_definitions = self._prepare_tools()

                # Call Claude API
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=self._system_prompt,
                    messages=messages,
                    tools=tool_definitions if tool_definitions else None,
                )

                # Update token usage
                self._token_usage.input_tokens += response.usage.input_tokens
                self._token_usage.output_tokens += response.usage.output_tokens

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
                    agent_id=self._agent_id,
                    success=True,
                )

                return {
                    "success": True,
                    "agent_id": self._agent_id,
                    "content": result.get("content", ""),
                    "tool_calls": result.get("tool_calls", []),
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "metadata": result.get("metadata", {}),
                }

            except Exception as e:
                self._status = AgentStatus.ERROR
                logger.error(
                    "agent_task_failed",
                    agent_id=self._agent_id,
                    error=str(e),
                )
                return {
                    "success": False,
                    "agent_id": self._agent_id,
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
            tool.get_schema().to_claude_schema()
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
                    tool_result = await tool.execute_with_validation(**content.input)
                    content_parts.append(
                        f"\n[Tool {content.name} result: {tool_result.data}]\n"
                    )

        return {
            "content": "\n".join(content_parts),
            "tool_calls": tool_calls,
            "metadata": {
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status.

        Returns:
            Status information
        """
        return {
            "agent_id": self._agent_id,
            "status": self._status.value,
            "model": self._model,
            "tools": list(self._tools.keys()),
            "message_count": len(self._message_history),
            "token_usage": self._token_usage.to_dict(),
        }

    async def reset(self) -> None:
        """Reset agent state."""
        async with self._lock:
            self._message_history.clear()
            self._status = AgentStatus.IDLE
            logger.info("agent_reset", agent_id=self._agent_id)

    async def close(self) -> None:
        """Close the agent and release resources."""
        async with self._lock:
            self._status = AgentStatus.TERMINATED
            if self._client:
                await self._client.close()
                self._client = None
            self._initialized = False
            logger.info("agent_closed", agent_id=self._agent_id)

    async def terminate(self) -> None:
        """Terminate the agent. Alias for close() for backwards compatibility."""
        await self.close()

    async def __aenter__(self) -> "ClaudeAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Backwards compatibility alias
Agent = ClaudeAgent
