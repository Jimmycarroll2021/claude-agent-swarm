"""
MCP (Model Context Protocol) connector for Claude Agent Swarm.

Provides integration with MCP servers for tool discovery and execution.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
from datetime import datetime
import subprocess
import os

from . import BaseTool, ToolSchema, ToolResult, ToolRegistry


logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when MCP server connection fails."""
    pass


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""
    pass


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "sse"
    url: Optional[str] = None  # For SSE transport
    timeout: int = 30
    auto_connect: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "transport": self.transport,
            "url": self.url,
            "timeout": self.timeout,
            "auto_connect": self.auto_connect
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env", {}),
            transport=data.get("transport", "stdio"),
            url=data.get("url"),
            timeout=data.get("timeout", 30),
            auto_connect=data.get("auto_connect", True)
        )


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    def to_tool_schema(self) -> ToolSchema:
        """Convert to ToolSchema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.get("properties", {}),
            required=self.input_schema.get("required", [])
        )


class MCPConnection:
    """Manages connection to an MCP server."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._connected = False
        self._tools: List[MCPTool] = []
        self._message_id = 0
        self._pending_responses: Dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected and self._process is not None
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get available tools from this server."""
        return self._tools.copy()
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        async with self._lock:
            if self._connected:
                return
            
            try:
                if self.config.transport == "stdio":
                    await self._connect_stdio()
                elif self.config.transport == "sse":
                    await self._connect_sse()
                else:
                    raise MCPConnectionError(f"Unknown transport: {self.config.transport}")
                
                self._connected = True
                logger.info(f"Connected to MCP server: {self.config.name}")
                
                # Discover tools
                await self._discover_tools()
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
                raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        env = os.environ.copy()
        env.update(self.config.env)
        
        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Start reading responses
        self._read_task = asyncio.create_task(self._read_responses())
        
        # Initialize connection
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "claude-agent-swarm", "version": "1.0.0"}
        })
    
    async def _connect_sse(self) -> None:
        """Connect via SSE transport."""
        # SSE transport implementation would go here
        # For now, raise not implemented
        raise NotImplementedError("SSE transport not yet implemented")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        async with self._lock:
            self._connected = False
            
            if self._read_task:
                self._read_task.cancel()
                try:
                    await self._read_task
                except asyncio.CancelledError:
                    pass
            
            if self._process:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._process.kill()
                
                self._process = None
            
            # Clear pending responses
            for future in self._pending_responses.values():
                future.cancel()
            self._pending_responses.clear()
            
            logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    async def _read_responses(self) -> None:
        """Read responses from the MCP server."""
        if not self._process or not self._process.stdout:
            return
        
        try:
            while self._connected:
                line = await self._process.stdout.readline()
                if not line:
                    break
                
                try:
                    message = json.loads(line.decode("utf-8").strip())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {self.config.name}: {line}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading from MCP server {self.config.name}: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming message from MCP server."""
        msg_id = message.get("id")
        
        if msg_id and msg_id in self._pending_responses:
            future = self._pending_responses.pop(msg_id)
            if not future.done():
                if "error" in message:
                    future.set_exception(MCPToolError(message["error"]))
                else:
                    future.set_result(message.get("result"))
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request to the MCP server."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected")
        
        self._message_id += 1
        msg_id = self._message_id
        
        message = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }
        
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future
        
        # Send message
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode("utf-8"))
        await self._process.stdin.drain()
        
        # Wait for response
        try:
            return await asyncio.wait_for(future, timeout=self.config.timeout)
        except asyncio.TimeoutError:
            self._pending_responses.pop(msg_id, None)
            raise MCPError(f"Timeout waiting for response from {self.config.name}")
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        
        tools_data = result.get("tools", [])
        self._tools = [
            MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                server_name=self.config.name
            )
            for tool in tools_data
        ]
        
        logger.info(f"Discovered {len(self._tools)} tools from {self.config.name}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        return result


class MCPConnector:
    """Manages connections to multiple MCP servers."""
    
    def __init__(self):
        self._connections: Dict[str, MCPConnection] = {}
        self._tool_registry: Optional[ToolRegistry] = None
    
    async def connect_server(self, config: MCPServerConfig) -> MCPConnection:
        """Connect to an MCP server."""
        if config.name in self._connections:
            logger.warning(f"Already connected to MCP server: {config.name}")
            return self._connections[config.name]
        
        connection = MCPConnection(config)
        await connection.connect()
        self._connections[config.name] = connection
        
        return connection
    
    async def disconnect_server(self, name: str) -> None:
        """Disconnect from an MCP server."""
        if name in self._connections:
            await self._connections[name].disconnect()
            del self._connections[name]
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for connection in list(self._connections.values()):
            await connection.disconnect()
        self._connections.clear()
    
    def get_connection(self, name: str) -> Optional[MCPConnection]:
        """Get a connection by server name."""
        return self._connections.get(name)
    
    def list_servers(self) -> List[str]:
        """List all connected server names."""
        return list(self._connections.keys())
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for connection in self._connections.values():
            tools.extend(connection.tools)
        return tools
    
    def find_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Find a tool by name across all servers."""
        for connection in self._connections.values():
            for tool in connection.tools:
                if tool.name == tool_name:
                    return tool
        return None
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> ToolResult:
        """Call a tool on an MCP server."""
        if server_name:
            connection = self._connections.get(server_name)
            if not connection:
                raise MCPError(f"Server not connected: {server_name}")
        else:
            # Find tool in any server
            tool = self.find_tool(tool_name)
            if not tool:
                raise MCPError(f"Tool not found: {tool_name}")
            connection = self._connections.get(tool.server_name)
        
        try:
            result = await connection.call_tool(tool_name, arguments)
            
            # Parse result
            content = result.get("content", [])
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "")
            
            is_error = result.get("isError", False)
            
            return ToolResult(
                success=not is_error,
                data=text_content if not is_error else None,
                error=text_content if is_error else None
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"MCP tool call failed: {str(e)}"
            )
    
    def register_tools_with_registry(self, registry: ToolRegistry) -> None:
        """Register all MCP tools with a ToolRegistry."""
        self._tool_registry = registry
        
        for tool in self.get_all_tools():
            mcp_tool_wrapper = MCPToolWrapper(tool, self)
            registry.register(mcp_tool_wrapper)
        
        logger.info(f"Registered {len(self.get_all_tools())} MCP tools with registry")


class MCPToolWrapper(BaseTool):
    """Wraps an MCP tool as a BaseTool."""
    
    def __init__(self, mcp_tool: MCPTool, connector: MCPConnector):
        super().__init__(mcp_tool.name, mcp_tool.description)
        self._mcp_tool = mcp_tool
        self._connector = connector
        self._schema = mcp_tool.to_tool_schema()
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the MCP tool."""
        return await self._connector.call_tool(self.name, kwargs)
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema."""
        return self._schema


__all__ = [
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "MCPServerConfig",
    "MCPTool",
    "MCPConnection",
    "MCPConnector",
    "MCPToolWrapper"
]
