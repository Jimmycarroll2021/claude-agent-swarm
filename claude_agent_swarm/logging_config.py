"""
Structured Logging Configuration for Claude Agent Swarm Framework

Provides JSON-formatted, contextual logging with agent/swarm tracking,
log rotation, and multiple output handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from logging.handlers import RotatingFileHandler


class AgentContextFilter(logging.Filter):
    """Filter that adds agent and swarm context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to log record."""
        record.agent_id = getattr(record, "agent_id", None)
        record.swarm_id = getattr(record, "swarm_id", None)
        record.task_id = getattr(record, "task_id", None)
        return True


def configure_logging(
    level: Union[str, int] = "INFO",
    json_format: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    additional_processors: Optional[List] = None
) -> structlog.BoundLogger:
    """
    Configure structured logging for the Claude Agent Swarm.
    
    Args:
        level: Log level (DEBUG, INFO, WARN, ERROR)
        json_format: Use JSON formatting
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Enable console output
        additional_processors: Additional structlog processors
        
    Returns:
        Configured structlog logger
        
    Example:
        >>> logger = configure_logging(
        ...     level="DEBUG",
        ...     json_format=True,
        ...     log_file="swarm.log"
        ... )
        >>> logger.info("Swarm started", swarm_id="swarm_1")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Shared processors for both stdlib and structlog
    shared_processors: List = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    # Add additional processors if provided
    if additional_processors:
        shared_processors.extend(additional_processors)
    
    # Structlog-specific processors
    structlog_processors = shared_processors + [
        structlog.stdlib.ExtraAdder(),
    ]
    
    # Renderer depends on format
    if json_format:
        structlog_processors.append(structlog.processors.JSONRenderer())
    else:
        structlog_processors.append(
            structlog.dev.ConsoleRenderer(colors=True, pad_event=False)
        )
    
    # Configure structlog
    structlog.configure(
        processors=structlog_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    handlers: List[logging.Handler] = []
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(AgentContextFilter())
        handlers.append(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.addFilter(AgentContextFilter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s" if json_format else "%(asctime)s [%(levelname)s] %(message)s",
        force=True
    )
    
    # Get and return the logger
    return structlog.get_logger("claude_agent_swarm")


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a configured structlog logger.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Structlog logger instance
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger("claude_agent_swarm")


class LogContext:
    """
    Context manager for adding contextual data to logs.
    
    Example:
        >>> with LogContext(agent_id="agent_1", swarm_id="swarm_1"):
        ...     logger.info("Processing task")
        # Output includes agent_id and swarm_id
    """
    
    def __init__(self, **context: Any):
        """
        Initialize context with key-value pairs.
        
        Args:
            **context: Context key-value pairs to add to logs
        """
        self.context = context
        self.token = None
    
    def __enter__(self) -> LogContext:
        """Enter context and bind variables."""
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and unbind variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent log calls.
    
    Args:
        **kwargs: Context key-value pairs
        
    Example:
        >>> bind_context(agent_id="agent_1")
        >>> logger.info("Task started")  # Includes agent_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Unbind context variables.
    
    Args:
        *keys: Keys to unbind
        
    Example:
        >>> unbind_context("agent_id", "task_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


class AgentLogger:
    """
    Logger wrapper with automatic agent/swarm context.
    
    Provides a convenient interface for logging with agent context
    without manually binding context variables each time.
    
    Example:
        >>> agent_logger = AgentLogger(agent_id="agent_1", swarm_id="swarm_1")
        >>> agent_logger.info("Task started", task="code_review")
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        swarm_id: Optional[str] = None,
        task_id: Optional[str] = None,
        logger: Optional[structlog.BoundLogger] = None
    ):
        """
        Initialize agent logger.
        
        Args:
            agent_id: Agent identifier
            swarm_id: Swarm identifier
            task_id: Task identifier
            logger: Underlying logger (optional)
        """
        self.agent_id = agent_id
        self.swarm_id = swarm_id
        self.task_id = task_id
        self._logger = logger or get_logger()
    
    def _bind_context(self, **kwargs: Any) -> structlog.BoundLogger:
        """Bind context to logger."""
        context = {}
        if self.agent_id:
            context["agent_id"] = self.agent_id
        if self.swarm_id:
            context["swarm_id"] = self.swarm_id
        if self.task_id:
            context["task_id"] = self.task_id
        context.update(kwargs)
        return self._logger.bind(**context)
    
    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._bind_context(**kwargs).debug(msg)
    
    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._bind_context(**kwargs).info(msg)
    
    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._bind_context(**kwargs).warning(msg)
    
    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        self._bind_context(**kwargs).error(msg)
    
    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log exception message with stack trace."""
        self._bind_context(**kwargs).exception(msg)
    
    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._bind_context(**kwargs).critical(msg)
    
    def bind(self, **kwargs: Any) -> AgentLogger:
        """
        Create new logger with additional bound context.
        
        Args:
            **kwargs: Additional context
            
        Returns:
            New AgentLogger with combined context
        """
        new_logger = AgentLogger(
            agent_id=self.agent_id,
            swarm_id=self.swarm_id,
            task_id=self.task_id,
            logger=self._logger
        )
        return new_logger


def log_agent_event(
    event_type: str,
    agent_id: str,
    swarm_id: Optional[str] = None,
    task_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    success: Optional[bool] = None,
    error: Optional[str] = None,
    **extra: Any
) -> None:
    """
    Log a standardized agent event.
    
    Args:
        event_type: Type of event (start, complete, error, etc.)
        agent_id: Agent identifier
        swarm_id: Swarm identifier
        task_id: Task identifier
        duration_ms: Event duration in milliseconds
        success: Whether the event was successful
        error: Error message if failed
        **extra: Additional event data
    """
    logger = get_logger()
    
    event_data = {
        "event_type": event_type,
        "agent_id": agent_id,
        "swarm_id": swarm_id,
        "task_id": task_id,
        "duration_ms": duration_ms,
        "success": success,
        "error": error,
        **extra
    }
    
    # Remove None values
    event_data = {k: v for k, v in event_data.items() if v is not None}
    
    if success is False or error:
        logger.error("agent_event", **event_data)
    elif event_type in ("start", "complete"):
        logger.info("agent_event", **event_data)
    else:
        logger.debug("agent_event", **event_data)


def log_tool_call(
    tool_name: str,
    agent_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log a tool call event.
    
    Args:
        tool_name: Name of the tool
        agent_id: Agent identifier
        duration_ms: Call duration in milliseconds
        success: Whether the call succeeded
        error: Error message if failed
        **kwargs: Additional tool call data
    """
    logger = get_logger()
    
    event_data = {
        "event_type": "tool_call",
        "tool_name": tool_name,
        "agent_id": agent_id,
        "duration_ms": duration_ms,
        "success": success,
        "error": error,
        **kwargs
    }
    
    # Remove None values
    event_data = {k: v for k, v in event_data.items() if v is not None}
    
    if not success or error:
        logger.warning("tool_call", **event_data)
    else:
        logger.debug("tool_call", **event_data)


def log_llm_request(
    model: str,
    agent_id: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    **kwargs: Any
) -> None:
    """
    Log an LLM request event.
    
    Args:
        model: Model name
        agent_id: Agent identifier
        prompt_tokens: Number of prompt tokens
        **kwargs: Additional request data
    """
    logger = get_logger()
    
    event_data = {
        "event_type": "llm_request",
        "model": model,
        "agent_id": agent_id,
        "prompt_tokens": prompt_tokens,
        **kwargs
    }
    
    event_data = {k: v for k, v in event_data.items() if v is not None}
    logger.debug("llm_request", **event_data)


def log_llm_response(
    model: str,
    agent_id: Optional[str] = None,
    completion_tokens: Optional[int] = None,
    duration_ms: Optional[float] = None,
    **kwargs: Any
) -> None:
    """
    Log an LLM response event.
    
    Args:
        model: Model name
        agent_id: Agent identifier
        completion_tokens: Number of completion tokens
        duration_ms: Response time in milliseconds
        **kwargs: Additional response data
    """
    logger = get_logger()
    
    event_data = {
        "event_type": "llm_response",
        "model": model,
        "agent_id": agent_id,
        "completion_tokens": completion_tokens,
        "duration_ms": duration_ms,
        **kwargs
    }
    
    event_data = {k: v for k, v in event_data.items() if v is not None}
    logger.debug("llm_response", **event_data)


# Default logger instance
_default_logger: Optional[structlog.BoundLogger] = None


def init_default_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Initialize default logging configuration.
    
    Args:
        level: Log level
        json_format: Use JSON formatting
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    global _default_logger
    _default_logger = configure_logging(
        level=level,
        json_format=json_format,
        log_file=log_file
    )
    return _default_logger


def get_default_logger() -> structlog.BoundLogger:
    """Get the default logger, initializing if necessary."""
    global _default_logger
    if _default_logger is None:
        _default_logger = configure_logging()
    return _default_logger
