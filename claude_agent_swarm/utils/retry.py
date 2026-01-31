"""Retry utilities for Claude Agent Swarm.

This module provides retry functionality with exponential backoff
for handling transient failures.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Optional, Type, Tuple, Union
from dataclasses import dataclass
from functools import wraps

import structlog

logger = structlog.get_logger()


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_max: float = 1.0
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )
    on_retry: Optional[Callable[[Exception, int], None]] = None
    on_giveup: Optional[Callable[[Exception], None]] = None


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None) -> None:
        """Initialize the error.
        
        Args:
            message: Error message
            last_exception: The last exception that caused the retry to fail
        """
        super().__init__(message)
        self.last_exception = last_exception


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback called on each retry
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry(max_attempts=3, initial_delay=1.0)
        ... async def fetch_data():
        ...     return await api_call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await _retry_async(func, config, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return _retry_sync(func, config, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


async def _retry_async(
    func: Callable,
    config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute async function with retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryError: If all attempts fail
    """
    last_exception: Optional[Exception] = None
    delay = config.initial_delay
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
            
            if attempt > 1:
                logger.info(
                    "retry_succeeded",
                    function=func.__name__,
                    attempt=attempt,
                )
            
            return result
            
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=config.max_attempts,
                    error=str(e),
                )
                
                if config.on_giveup:
                    config.on_giveup(e)
                
                raise RetryError(
                    f"Function {func.__name__} failed after {config.max_attempts} attempts",
                    last_exception=e,
                ) from e
            
            logger.warning(
                "retry_attempt",
                function=func.__name__,
                attempt=attempt,
                next_attempt=attempt + 1,
                delay=delay,
                error=str(e),
            )
            
            if config.on_retry:
                config.on_retry(e, attempt)
            
            # Wait before retry
            await asyncio.sleep(delay)
            
            # Calculate next delay with exponential backoff
            delay = min(
                delay * config.backoff_multiplier,
                config.max_delay,
            )
            
            # Add jitter if enabled
            if config.jitter:
                delay += random.uniform(0, config.jitter_max)
    
    # Should never reach here
    raise RetryError("Unexpected end of retry loop", last_exception=last_exception)


def _retry_sync(
    func: Callable,
    config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute sync function with retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryError: If all attempts fail
    """
    last_exception: Optional[Exception] = None
    delay = config.initial_delay
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            
            if attempt > 1:
                logger.info(
                    "retry_succeeded",
                    function=func.__name__,
                    attempt=attempt,
                )
            
            return result
            
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=config.max_attempts,
                    error=str(e),
                )
                
                if config.on_giveup:
                    config.on_giveup(e)
                
                raise RetryError(
                    f"Function {func.__name__} failed after {config.max_attempts} attempts",
                    last_exception=e,
                ) from e
            
            logger.warning(
                "retry_attempt",
                function=func.__name__,
                attempt=attempt,
                next_attempt=attempt + 1,
                delay=delay,
                error=str(e),
            )
            
            if config.on_retry:
                config.on_retry(e, attempt)
            
            # Wait before retry
            time.sleep(delay)
            
            # Calculate next delay with exponential backoff
            delay = min(
                delay * config.backoff_multiplier,
                config.max_delay,
            )
            
            # Add jitter if enabled
            if config.jitter:
                delay += random.uniform(0, config.jitter_max)
    
    # Should never reach here
    raise RetryError("Unexpected end of retry loop", last_exception=last_exception)


class RetryHandler:
    """Handler for retry operations.
    
    The RetryHandler provides a class-based interface for retry logic
    with configurable behavior.
    
    Example:
        >>> handler = RetryHandler(RetryConfig(max_attempts=5))
        >>> result = await handler.execute(async_function)
    """
    
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        """Initialize the retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if asyncio.iscoroutinefunction(func):
            return await _retry_async(func, self.config, *args, **kwargs)
        else:
            return _retry_sync(func, self.config, *args, **kwargs)
    
    def wrap(self, func: Callable) -> Callable:
        """Wrap a function with retry logic.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await self.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return self.execute(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
