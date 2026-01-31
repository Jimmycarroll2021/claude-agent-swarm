"""
State Manager for Claude Agent Swarm Framework.

Provides thread-safe shared state management with SQLite backend
and optional Redis support for distributed deployments.
"""

import asyncio
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class StateEntry:
    """Represents a single state entry with metadata."""
    key: str
    value: Any
    namespace: str
    created_at: datetime
    updated_at: datetime
    ttl: Optional[int] = None  # Time-to-live in seconds
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ttl": self.ttl,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateEntry":
        """Create entry from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            namespace=data["namespace"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {})
        )


class StateBackend(ABC):
    """Abstract base class for state backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the backend."""
        pass

    @abstractmethod
    async def get(self, namespace: str, key: str) -> Optional[StateEntry]:
        """Get a state entry by namespace and key."""
        pass

    @abstractmethod
    async def set(self, entry: StateEntry) -> None:
        """Set a state entry."""
        pass

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a state entry. Returns True if deleted."""
        pass

    @abstractmethod
    async def get_all(self, namespace: str) -> Dict[str, StateEntry]:
        """Get all state entries in a namespace."""
        pass

    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace. Returns count deleted."""
        pass

    @abstractmethod
    async def list_namespaces(self) -> List[str]:
        """List all namespaces."""
        pass


class SQLiteBackend(StateBackend):
    """SQLite-based state backend with async support."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = Path(db_path) if db_path != ":memory:" else ":memory:"
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize SQLite connection and create tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"SQLite backend connected: {self.db_path}")

    async def disconnect(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite backend disconnected")

    async def _create_tables(self) -> None:
        """Create necessary database tables."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS state_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ttl INTEGER,
                metadata TEXT,
                PRIMARY KEY (namespace, key)
            )
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_namespace 
            ON state_entries(namespace)
        """)
        await self._connection.commit()

    def _serialize_value(self, value: Any) -> str:
        """Serialize value to JSON string."""
        return json.dumps(value, default=str)

    def _deserialize_value(self, value_str: str) -> Any:
        """Deserialize JSON string to value."""
        return json.loads(value_str)

    def _row_to_entry(self, row: aiosqlite.Row) -> StateEntry:
        """Convert database row to StateEntry."""
        return StateEntry(
            key=row["key"],
            value=self._deserialize_value(row["value"]),
            namespace=row["namespace"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            ttl=row["ttl"],
            metadata=self._deserialize_value(row["metadata"]) if row["metadata"] else {}
        )

    async def get(self, namespace: str, key: str) -> Optional[StateEntry]:
        """Get a state entry by namespace and key."""
        async with self._lock:
            cursor = await self._connection.execute(
                "SELECT * FROM state_entries WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            row = await cursor.fetchone()
            if row:
                entry = self._row_to_entry(row)
                # Check TTL
                if entry.ttl is not None:
                    age = (datetime.now() - entry.updated_at).total_seconds()
                    if age > entry.ttl:
                        await self.delete(namespace, key)
                        return None
                return entry
            return None

    async def set(self, entry: StateEntry) -> None:
        """Set a state entry."""
        async with self._lock:
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO state_entries 
                (namespace, key, value, created_at, updated_at, ttl, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.namespace,
                    entry.key,
                    self._serialize_value(entry.value),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.ttl,
                    self._serialize_value(entry.metadata) if entry.metadata else None
                )
            )
            await self._connection.commit()

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a state entry. Returns True if deleted."""
        async with self._lock:
            cursor = await self._connection.execute(
                "DELETE FROM state_entries WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            await self._connection.commit()
            return cursor.rowcount > 0

    async def get_all(self, namespace: str) -> Dict[str, StateEntry]:
        """Get all state entries in a namespace."""
        async with self._lock:
            cursor = await self._connection.execute(
                "SELECT * FROM state_entries WHERE namespace = ?",
                (namespace,)
            )
            rows = await cursor.fetchall()
            entries = {}
            for row in rows:
                entry = self._row_to_entry(row)
                # Check TTL
                if entry.ttl is not None:
                    age = (datetime.now() - entry.updated_at).total_seconds()
                    if age > entry.ttl:
                        await self.delete(namespace, entry.key)
                        continue
                entries[entry.key] = entry
            return entries

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace. Returns count deleted."""
        async with self._lock:
            cursor = await self._connection.execute(
                "DELETE FROM state_entries WHERE namespace = ?",
                (namespace,)
            )
            await self._connection.commit()
            return cursor.rowcount

    async def list_namespaces(self) -> List[str]:
        """List all namespaces."""
        async with self._lock:
            cursor = await self._connection.execute(
                "SELECT DISTINCT namespace FROM state_entries"
            )
            rows = await cursor.fetchall()
            return [row["namespace"] for row in rows]

    async def checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Create a checkpoint backup of the database."""
        checkpoint_path = Path(checkpoint_path)
        async with self._lock:
            # Use SQLite backup API
            backup_conn = await aiosqlite.connect(checkpoint_path)
            await self._connection.backup(backup_conn)
            await backup_conn.close()
            logger.info(f"Checkpoint created: {checkpoint_path}")


class RedisBackend(StateBackend):
    """Redis-based state backend for distributed deployments."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "swarm:state:"
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()

    def _make_key(self, namespace: str, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{namespace}:{key}"

    def _parse_key(self, redis_key: str) -> tuple:
        """Parse Redis key into namespace and key."""
        parts = redis_key[len(self.key_prefix):].split(":", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            await self._redis.ping()
            logger.info(f"Redis backend connected: {self.host}:{self.port}")
        except ImportError:
            raise ImportError(
                "Redis support requires 'redis' package. "
                "Install with: pip install redis"
            )

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis backend disconnected")

    def _serialize_value(self, value: Any) -> str:
        """Serialize value to JSON string."""
        return json.dumps(value, default=str)

    def _deserialize_value(self, value_str: str) -> Any:
        """Deserialize JSON string to value."""
        return json.loads(value_str)

    async def get(self, namespace: str, key: str) -> Optional[StateEntry]:
        """Get a state entry by namespace and key."""
        async with self._lock:
            redis_key = self._make_key(namespace, key)
            data = await self._redis.hgetall(redis_key)
            if not data:
                return None

            return StateEntry(
                key=key,
                value=self._deserialize_value(data.get("value", "null")),
                namespace=namespace,
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
                ttl=int(data["ttl"]) if data.get("ttl") else None,
                metadata=self._deserialize_value(data.get("metadata", "{}"))
            )

    async def set(self, entry: StateEntry) -> None:
        """Set a state entry."""
        async with self._lock:
            redis_key = self._make_key(entry.namespace, entry.key)
            data = {
                "value": self._serialize_value(entry.value),
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "metadata": self._serialize_value(entry.metadata) if entry.metadata else "{}"
            }
            if entry.ttl:
                data["ttl"] = str(entry.ttl)

            await self._redis.hset(redis_key, mapping=data)
            if entry.ttl:
                await self._redis.expire(redis_key, entry.ttl)

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a state entry. Returns True if deleted."""
        async with self._lock:
            redis_key = self._make_key(namespace, key)
            result = await self._redis.delete(redis_key)
            return result > 0

    async def get_all(self, namespace: str) -> Dict[str, StateEntry]:
        """Get all state entries in a namespace."""
        async with self._lock:
            pattern = f"{self.key_prefix}{namespace}:*"
            keys = await self._redis.keys(pattern)
            entries = {}
            for redis_key in keys:
                data = await self._redis.hgetall(redis_key)
                if data:
                    _, key = self._parse_key(redis_key)
                    entries[key] = StateEntry(
                        key=key,
                        value=self._deserialize_value(data.get("value", "null")),
                        namespace=namespace,
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
                        ttl=int(data["ttl"]) if data.get("ttl") else None,
                        metadata=self._deserialize_value(data.get("metadata", "{}"))
                    )
            return entries

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace. Returns count deleted."""
        async with self._lock:
            pattern = f"{self.key_prefix}{namespace}:*"
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0

    async def list_namespaces(self) -> List[str]:
        """List all namespaces."""
        async with self._lock:
            pattern = f"{self.key_prefix}*"
            keys = await self._redis.keys(pattern)
            namespaces = set()
            for key in keys:
                ns, _ = self._parse_key(key)
                namespaces.add(ns)
            return list(namespaces)


class StateManager:
    """
    Thread-safe state manager for Claude Agent Swarm.
    
    Provides key-value storage with namespacing, persistence,
    and optional distributed backend support.
    
    Example:
        >>> manager = StateManager(backend="sqlite", db_path="state.db")
        >>> await manager.connect()
        >>> await manager.set("agent_1", "status", "active")
        >>> status = await manager.get("agent_1", "status")
        >>> await manager.disconnect()
    """

    def __init__(
        self,
        backend: str = "sqlite",
        namespace_prefix: str = "swarm",
        **backend_kwargs
    ):
        """
        Initialize StateManager.
        
        Args:
            backend: Backend type ("sqlite" or "redis")
            namespace_prefix: Default prefix for all namespaces
            **backend_kwargs: Backend-specific configuration
        """
        self.namespace_prefix = namespace_prefix
        self._lock = asyncio.Lock()
        self._connected = False

        if backend == "sqlite":
            self._backend: StateBackend = SQLiteBackend(**backend_kwargs)
        elif backend == "redis":
            self._backend = RedisBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _make_namespace(self, namespace: str) -> str:
        """Create full namespace with prefix."""
        return f"{self.namespace_prefix}:{namespace}"

    async def connect(self) -> None:
        """Connect to the state backend."""
        async with self._lock:
            await self._backend.connect()
            self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the state backend."""
        async with self._lock:
            await self._backend.disconnect()
            self._connected = False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["StateManager", None]:
        """Context manager for automatic connection handling."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    async def get(
        self,
        namespace: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a value from the state store.
        
        Args:
            namespace: The namespace (e.g., agent ID)
            key: The key to retrieve
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        full_namespace = self._make_namespace(namespace)
        entry = await self._backend.get(full_namespace, key)
        return entry.value if entry else default

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set a value in the state store.
        
        Args:
            namespace: The namespace (e.g., agent ID)
            key: The key to set
            value: The value to store
            ttl: Optional time-to-live in seconds
            metadata: Optional metadata dictionary
        """
        full_namespace = self._make_namespace(namespace)
        now = datetime.now()
        
        # Check if entry exists to preserve created_at
        existing = await self._backend.get(full_namespace, key)
        created_at = existing.created_at if existing else now
        
        entry = StateEntry(
            key=key,
            value=value,
            namespace=full_namespace,
            created_at=created_at,
            updated_at=now,
            ttl=ttl,
            metadata=metadata
        )
        await self._backend.set(entry)

    async def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a value from the state store.
        
        Args:
            namespace: The namespace
            key: The key to delete
            
        Returns:
            True if deleted, False if not found
        """
        full_namespace = self._make_namespace(namespace)
        return await self._backend.delete(full_namespace, key)

    async def get_all(self, namespace: str) -> Dict[str, Any]:
        """
        Get all key-value pairs in a namespace.
        
        Args:
            namespace: The namespace to query
            
        Returns:
            Dictionary of key-value pairs
        """
        full_namespace = self._make_namespace(namespace)
        entries = await self._backend.get_all(full_namespace)
        return {k: v.value for k, v in entries.items()}

    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all entries in a namespace.
        
        Args:
            namespace: The namespace to clear
            
        Returns:
            Number of entries deleted
        """
        full_namespace = self._make_namespace(namespace)
        return await self._backend.clear_namespace(full_namespace)

    async def get_entry(self, namespace: str, key: str) -> Optional[StateEntry]:
        """
        Get the full StateEntry including metadata.
        
        Args:
            namespace: The namespace
            key: The key to retrieve
            
        Returns:
            StateEntry or None
        """
        full_namespace = self._make_namespace(namespace)
        return await self._backend.get(full_namespace, key)

    async def update_metadata(
        self,
        namespace: str,
        key: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for an existing entry.
        
        Args:
            namespace: The namespace
            key: The key to update
            metadata: New metadata dictionary
            
        Returns:
            True if updated, False if not found
        """
        full_namespace = self._make_namespace(namespace)
        entry = await self._backend.get(full_namespace, key)
        if entry:
            entry.metadata = {**(entry.metadata or {}), **metadata}
            entry.updated_at = datetime.now()
            await self._backend.set(entry)
            return True
        return False

    async def increment(
        self,
        namespace: str,
        key: str,
        amount: int = 1
    ) -> int:
        """
        Atomically increment a numeric value.
        
        Args:
            namespace: The namespace
            key: The key to increment
            amount: Amount to increment by
            
        Returns:
            New value after increment
        """
        async with self._lock:
            current = await self.get(namespace, key, 0)
            new_value = current + amount
            await self.set(namespace, key, new_value)
            return new_value

    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces.
        
        Returns:
            List of namespace names
        """
        namespaces = await self._backend.list_namespaces()
        prefix_len = len(self.namespace_prefix) + 1
        return [ns[prefix_len:] if ns.startswith(self.namespace_prefix) else ns 
                for ns in namespaces]

    async def checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create a checkpoint of the current state.
        
        Args:
            checkpoint_path: Path for checkpoint (SQLite only)
        """
        if isinstance(self._backend, SQLiteBackend):
            if checkpoint_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f"state_checkpoint_{timestamp}.db"
            await self._backend.checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not supported for Redis backend")

    async def restore(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if isinstance(self._backend, SQLiteBackend):
            # Close current connection and copy checkpoint
            await self.disconnect()
            import shutil
            shutil.copy(checkpoint_path, self._backend.db_path)
            await self.connect()
            logger.info(f"State restored from: {checkpoint_path}")
        else:
            raise NotImplementedError("Restore only supported for SQLite backend")

    def is_connected(self) -> bool:
        """Check if manager is connected to backend."""
        return self._connected


# Singleton instance for global state management
_state_manager_instance: Optional[StateManager] = None


def get_state_manager(
    backend: str = "sqlite",
    **kwargs
) -> StateManager:
    """Get or create global state manager instance."""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager(backend=backend, **kwargs)
    return _state_manager_instance


def reset_state_manager() -> None:
    """Reset global state manager instance."""
    global _state_manager_instance
    _state_manager_instance = None
