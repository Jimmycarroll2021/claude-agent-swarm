"""Context manager for Claude Agent Swarm.

This module provides context management for maintaining state across
operations and sharing information between agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass, field
from collections import deque
from contextlib import contextmanager
import copy

import structlog

logger = structlog.get_logger()


@dataclass
class ContextScope:
    """A context scope with metadata."""
    
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    max_size: int = 1000


class ContextManager:
    """Manager for hierarchical contexts.
    
    The ContextManager maintains a stack of contexts that can be used
to share information across operations and between agents.
    
    Example:
        >>> ctx = ContextManager()
        >>> ctx.set("key", "value")
        >>> with ctx.scope("operation"):
        ...     ctx.set("local_key", "local_value")
        ...     value = ctx.get("key")  # Can access parent scope
    """
    
    def __init__(self, max_history: int = 100) -> None:
        """Initialize the context manager.
        
        Args:
            max_history: Maximum number of context snapshots to keep
        """
        self._scopes: Dict[str, ContextScope] = {}
        self._scope_stack: List[str] = []
        self._history: deque = deque(maxlen=max_history)
        
        # Create root scope
        self._root_scope = "root"
        self._scopes[self._root_scope] = ContextScope(name=self._root_scope)
        self._scope_stack.append(self._root_scope)
        
        logger.debug("context_manager_initialized")
    
    @property
    def current_scope(self) -> str:
        """Get current scope name."""
        return self._scope_stack[-1] if self._scope_stack else self._root_scope
    
    def get(self, key: str, default: Any = None, scope: Optional[str] = None) -> Any:
        """Get a value from context.
        
        Args:
            key: Key to look up
            default: Default value if not found
            scope: Optional scope to search (defaults to current scope and parents)
            
        Returns:
            Value or default
        """
        if scope:
            # Search specific scope only
            ctx_scope = self._scopes.get(scope)
            if ctx_scope:
                return ctx_scope.data.get(key, default)
            return default
        
        # Search current scope and parents
        for scope_name in reversed(self._scope_stack):
            ctx_scope = self._scopes.get(scope_name)
            if ctx_scope and key in ctx_scope.data:
                return ctx_scope.data[key]
        
        return default
    
    def set(self, key: str, value: Any, scope: Optional[str] = None) -> None:
        """Set a value in context.
        
        Args:
            key: Key to set
            value: Value to set
            scope: Optional scope (defaults to current scope)
        """
        target_scope = scope or self.current_scope
        ctx_scope = self._scopes.get(target_scope)
        
        if not ctx_scope:
            logger.warning("scope_not_found", scope=target_scope)
            return
        
        # Check size limit
        if len(ctx_scope.data) >= ctx_scope.max_size and key not in ctx_scope.data:
            logger.warning("scope_full", scope=target_scope)
            return
        
        ctx_scope.data[key] = value
        logger.debug("context_set", key=key, scope=target_scope)
    
    def delete(self, key: str, scope: Optional[str] = None) -> bool:
        """Delete a value from context.
        
        Args:
            key: Key to delete
            scope: Optional scope (defaults to current scope)
            
        Returns:
            True if deleted, False if not found
        """
        target_scope = scope or self.current_scope
        ctx_scope = self._scopes.get(target_scope)
        
        if ctx_scope and key in ctx_scope.data:
            del ctx_scope.data[key]
            logger.debug("context_deleted", key=key, scope=target_scope)
            return True
        
        return False
    
    def has(self, key: str, scope: Optional[str] = None) -> bool:
        """Check if a key exists in context.
        
        Args:
            key: Key to check
            scope: Optional scope (defaults to current scope and parents)
            
        Returns:
            True if key exists
        """
        if scope:
            ctx_scope = self._scopes.get(scope)
            return ctx_scope is not None and key in ctx_scope.data
        
        for scope_name in reversed(self._scope_stack):
            ctx_scope = self._scopes.get(scope_name)
            if ctx_scope and key in ctx_scope.data:
                return True
        
        return False
    
    @contextmanager
    def scope(self, name: str, inherit: bool = True) -> Iterator[ContextManager]:
        """Create a new context scope.
        
        Args:
            name: Scope name
            inherit: Whether to inherit from parent scope
            
        Yields:
            Context manager (self)
        """
        # Create new scope
        parent = self.current_scope if inherit else None
        self._scopes[name] = ContextScope(name=name, parent=parent)
        self._scope_stack.append(name)
        
        logger.debug("scope_entered", name=name, parent=parent)
        
        try:
            yield self
        finally:
            # Exit scope
            self._scope_stack.pop()
            
            # Save snapshot to history
            self._history.append({
                "scope": name,
                "data": copy.deepcopy(self._scopes[name].data),
            })
            
            logger.debug("scope_exited", name=name)
    
    def get_all(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Get all values from a scope.
        
        Args:
            scope: Optional scope (defaults to current scope)
            
        Returns:
            Dictionary of all values
        """
        target_scope = scope or self.current_scope
        ctx_scope = self._scopes.get(target_scope)
        
        if ctx_scope:
            return ctx_scope.data.copy()
        
        return {}
    
    def merge(self, data: Dict[str, Any], scope: Optional[str] = None) -> None:
        """Merge dictionary into context.
        
        Args:
            data: Dictionary to merge
            scope: Optional scope (defaults to current scope)
        """
        target_scope = scope or self.current_scope
        ctx_scope = self._scopes.get(target_scope)
        
        if ctx_scope:
            ctx_scope.data.update(data)
            logger.debug("context_merged", keys=list(data.keys()), scope=target_scope)
    
    def clear(self, scope: Optional[str] = None) -> None:
        """Clear context values.
        
        Args:
            scope: Optional scope (defaults to current scope)
        """
        target_scope = scope or self.current_scope
        ctx_scope = self._scopes.get(target_scope)
        
        if ctx_scope:
            ctx_scope.data.clear()
            logger.debug("context_cleared", scope=target_scope)
    
    def get_scope_names(self) -> List[str]:
        """Get all scope names.
        
        Returns:
            List of scope names
        """
        return list(self._scopes.keys())
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get context history.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of historical context snapshots
        """
        return list(self._history)[-limit:]
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of all scopes.
        
        Returns:
            Snapshot dictionary
        """
        return {
            name: copy.deepcopy(scope.data)
            for name, scope in self._scopes.items()
        }
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from a snapshot.
        
        Args:
            snapshot: Snapshot to restore from
        """
        self._scopes.clear()
        
        for name, data in snapshot.items():
            self._scopes[name] = ContextScope(name=name, data=copy.deepcopy(data))
        
        # Reset stack to root
        self._scope_stack = [self._root_scope]
        
        logger.info("context_restored", scopes=list(snapshot.keys()))
