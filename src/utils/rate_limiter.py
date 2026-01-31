"""Rate limiter for Claude Agent Swarm.

This module provides rate limiting functionality to prevent
exceeding API rate limits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
from collections import deque

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    
    requests_per_minute: int = 60
    burst_size: int = 10
    enabled: bool = True


class RateLimiter:
    """Token bucket rate limiter.
    
    The RateLimiter implements a token bucket algorithm to control
    the rate of requests and prevent exceeding API limits.
    
    Example:
        >>> limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        >>> async with limiter.acquire():
        ...     await make_api_call()
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        enabled: bool = True,
    ) -> None:
        """Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size
            enabled: Whether rate limiting is enabled
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.enabled = enabled
        
        # Token bucket
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
        
        # Request history for tracking
        self._request_times: deque = deque(maxlen=1000)
        
        # Calculate token rate (tokens per second)
        self._token_rate = requests_per_minute / 60.0
        
        logger.info(
            "rate_limiter_initialized",
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            enabled=enabled,
        )
    
    async def acquire(self, tokens: int = 1) -> "RateLimiterContext":
        """Acquire permission to make a request.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Context manager for the acquired permission
        """
        if not self.enabled:
            return RateLimiterContext(self, 0)
        
        async with self._lock:
            await self._add_tokens()
            
            # Wait until we have enough tokens
            while self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self._token_rate
                logger.debug("rate_limit_waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)
                await self._add_tokens()
            
            # Consume tokens
            self._tokens -= tokens
            self._request_times.append(time.time())
        
        return RateLimiterContext(self, tokens)
    
    async def _add_tokens(self) -> None:
        """Add tokens to the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        
        # Add tokens based on elapsed time
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self._token_rate
        )
        
        self._last_update = now
    
    def get_current_rate(self) -> float:
        """Get current request rate (requests per minute).
        
        Returns:
            Current rate
        """
        if not self._request_times:
            return 0.0
        
        now = time.time()
        one_minute_ago = now - 60
        
        # Count requests in the last minute
        recent_requests = sum(
            1 for t in self._request_times
            if t > one_minute_ago
        )
        
        return float(recent_requests)
    
    def get_available_tokens(self) -> float:
        """Get available tokens.
        
        Returns:
            Available tokens
        """
        return self._tokens
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "current_rate": self.get_current_rate(),
            "available_tokens": self.get_available_tokens(),
            "total_requests": len(self._request_times),
        }
    
    async def reset(self) -> None:
        """Reset the rate limiter."""
        async with self._lock:
            self._tokens = float(self.burst_size)
            self._last_update = time.time()
            self._request_times.clear()
        
        logger.info("rate_limiter_reset")


class RateLimiterContext:
    """Context manager for rate limited operations."""
    
    def __init__(self, limiter: RateLimiter, tokens: int) -> None:
        """Initialize the context.
        
        Args:
            limiter: Parent rate limiter
            tokens: Acquired tokens
        """
        self._limiter = limiter
        self._tokens = tokens
    
    async def __aenter__(self) -> RateLimiterContext:
        """Enter the context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        # Tokens are already consumed, nothing to do
        pass
    
    def __enter__(self) -> RateLimiterContext:
        """Synchronous enter (for compatibility)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Synchronous exit (for compatibility)."""
        pass
