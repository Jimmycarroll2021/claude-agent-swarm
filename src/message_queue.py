"""
Message Queue for Claude Agent Swarm Framework.

Provides async message queue system with direct messaging,
broadcast messaging, and message persistence.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system."""
    TASK = "task"           # Task assignment
    RESULT = "result"       # Task result
    CONTROL = "control"     # Control commands
    BROADCAST = "broadcast" # Broadcast message
    HEARTBEAT = "heartbeat" # Heartbeat/keepalive
    ERROR = "error"         # Error notification


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Message:
    """Represents a message in the queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK
    sender: str = ""
    recipient: Optional[str] = None  # None for broadcast
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None  # For request-response
    ttl: Optional[int] = None  # Time-to-live in seconds
    delivered: bool = False
    acknowledged: bool = False
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "delivered": self.delivered,
            "acknowledged": self.acknowledged,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            recipient=data.get("recipient"),
            content=data.get("content", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MessagePriority(data.get("priority", 2)),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl"),
            delivered=data.get("delivered", False),
            acknowledged=data.get("acknowledged", False),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {})
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl


class MessageStore:
    """Persistent message store using SQLite."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize database connection."""
        self._connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info(f"Message store connected: {self.db_path}")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Message store disconnected")

    async def _create_tables(self) -> None:
        """Create database tables."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                priority INTEGER NOT NULL,
                correlation_id TEXT,
                ttl INTEGER,
                delivered BOOLEAN DEFAULT 0,
                acknowledged BOOLEAN DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                metadata TEXT
            )
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_recipient 
            ON messages(recipient)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_correlation 
            ON messages(correlation_id)
        """)
        await self._connection.commit()

    async def save_message(self, message: Message) -> None:
        """Save a message to persistent storage."""
        async with self._lock:
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO messages 
                (id, type, sender, recipient, content, timestamp, priority,
                 correlation_id, ttl, delivered, acknowledged, retry_count, 
                 max_retries, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.type.value,
                    message.sender,
                    message.recipient,
                    json.dumps(message.content),
                    message.timestamp.isoformat(),
                    message.priority.value,
                    message.correlation_id,
                    message.ttl,
                    message.delivered,
                    message.acknowledged,
                    message.retry_count,
                    message.max_retries,
                    json.dumps(message.metadata)
                )
            )
            await self._connection.commit()

    async def load_pending_messages(self, recipient: str) -> List[Message]:
        """Load pending (undelivered) messages for a recipient."""
        async with self._lock:
            cursor = await self._connection.execute(
                """
                SELECT * FROM messages 
                WHERE recipient = ? AND delivered = 0
                ORDER BY priority ASC, timestamp ASC
                """,
                (recipient,)
            )
            rows = await cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    async def load_broadcast_messages(self, since: datetime) -> List[Message]:
        """Load broadcast messages since a timestamp."""
        async with self._lock:
            cursor = await self._connection.execute(
                """
                SELECT * FROM messages 
                WHERE recipient IS NULL AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                (since.isoformat(),)
            )
            rows = await cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    async def update_message_status(
        self,
        message_id: str,
        delivered: Optional[bool] = None,
        acknowledged: Optional[bool] = None,
        retry_count: Optional[int] = None
    ) -> None:
        """Update message status."""
        async with self._lock:
            updates = []
            params = []
            if delivered is not None:
                updates.append("delivered = ?")
                params.append(delivered)
            if acknowledged is not None:
                updates.append("acknowledged = ?")
                params.append(acknowledged)
            if retry_count is not None:
                updates.append("retry_count = ?")
                params.append(retry_count)
            
            if updates:
                params.append(message_id)
                await self._connection.execute(
                    f"UPDATE messages SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                await self._connection.commit()

    async def delete_message(self, message_id: str) -> None:
        """Delete a message from storage."""
        async with self._lock:
            await self._connection.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
            await self._connection.commit()

    async def cleanup_expired(self) -> int:
        """Remove expired messages. Returns count deleted."""
        async with self._lock:
            now = datetime.now().isoformat()
            cursor = await self._connection.execute(
                """
                DELETE FROM messages 
                WHERE ttl IS NOT NULL AND 
                datetime(timestamp, '+' || ttl || ' seconds') < datetime(?)
                """,
                (now,)
            )
            await self._connection.commit()
            return cursor.rowcount

    async def cleanup_acknowledged(self, older_than_hours: int = 24) -> int:
        """Remove old acknowledged messages. Returns count deleted."""
        async with self._lock:
            cursor = await self._connection.execute(
                """
                DELETE FROM messages 
                WHERE acknowledged = 1 AND 
                datetime(timestamp, '+' || ? || ' hours') < datetime('now')
                """,
                (older_than_hours,)
            )
            await self._connection.commit()
            return cursor.rowcount

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        """Convert database row to Message."""
        return Message(
            id=row["id"],
            type=MessageType(row["type"]),
            sender=row["sender"],
            recipient=row["recipient"],
            content=json.loads(row["content"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            priority=MessagePriority(row["priority"]),
            correlation_id=row["correlation_id"],
            ttl=row["ttl"],
            delivered=bool(row["delivered"]),
            acknowledged=bool(row["acknowledged"]),
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )


class MessageQueue:
    """
    Async message queue for inter-agent communication.
    
    Supports direct messaging (point-to-point) and broadcast
    messaging (publish-subscribe) with message persistence.
    
    Example:
        >>> queue = MessageQueue(persistence=True)
        >>> await queue.connect()
        >>> 
        >>> # Send direct message
        >>> msg = await queue.send(
        ...     sender="agent_1",
        ...     recipient="agent_2",
        ...     content={"task": "process_data"}
        ... )
        >>> 
        >>> # Subscribe to broadcasts
        >>> await queue.subscribe("agent_1", handler)
        >>> 
        >>> # Receive messages
        >>> messages = await queue.receive("agent_2")
        >>> 
        >>> await queue.disconnect()
    """

    def __init__(
        self,
        persistence: bool = True,
        db_path: Union[str, Path] = ":memory:",
        max_queue_size: int = 10000,
        cleanup_interval: int = 300  # seconds
    ):
        """
        Initialize MessageQueue.
        
        Args:
            persistence: Enable message persistence
            db_path: Path for persistent storage
            max_queue_size: Maximum in-memory queue size per agent
            cleanup_interval: Interval for cleanup tasks in seconds
        """
        self.persistence = persistence
        self.max_queue_size = max_queue_size
        self.cleanup_interval = cleanup_interval
        
        self._store: Optional[MessageStore] = None
        if persistence:
            self._store = MessageStore(db_path)
        
        # In-memory queues: recipient -> asyncio.Queue
        self._queues: Dict[str, asyncio.Queue] = {}
        
        # Subscribers: channel -> set of recipient IDs
        self._subscribers: Dict[str, Set[str]] = {}
        
        # Broadcast handlers: recipient -> handler callback
        self._handlers: Dict[str, Callable[[Message], None]] = {}
        
        # Locks
        self._queue_lock = asyncio.Lock()
        self._sub_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self) -> None:
        """Initialize the message queue."""
        if self._store:
            await self._store.connect()
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Message queue connected")

    async def disconnect(self) -> None:
        """Shutdown the message queue."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._store:
            await self._store.disconnect()
        
        # Clear in-memory queues
        self._queues.clear()
        self._subscribers.clear()
        self._handlers.clear()
        
        logger.info("Message queue disconnected")

    def _get_queue(self, recipient: str) -> asyncio.Queue:
        """Get or create queue for recipient."""
        if recipient not in self._queues:
            self._queues[recipient] = asyncio.Queue(maxsize=self.max_queue_size)
        return self._queues[recipient]

    async def send(
        self,
        sender: str,
        recipient: str,
        content: Dict[str, Any],
        msg_type: MessageType = MessageType.TASK,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Send a direct message to a recipient.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content dictionary
            msg_type: Type of message
            priority: Message priority
            correlation_id: ID for request-response correlation
            ttl: Time-to-live in seconds
            metadata: Additional metadata
            
        Returns:
            The sent Message object
        """
        message = Message(
            type=msg_type,
            sender=sender,
            recipient=recipient,
            content=content,
            priority=priority,
            correlation_id=correlation_id,
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # Persist message
        if self._store:
            await self._store.save_message(message)
        
        # Add to in-memory queue
        async with self._queue_lock:
            queue = self._get_queue(recipient)
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {recipient}, dropping message")
                raise
        
        logger.debug(f"Message sent: {message.id} from {sender} to {recipient}")
        return message

    async def broadcast(
        self,
        sender: str,
        content: Dict[str, Any],
        channel: str = "default",
        msg_type: MessageType = MessageType.BROADCAST,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Broadcast a message to all subscribers of a channel.
        
        Args:
            sender: Sender agent ID
            content: Message content dictionary
            channel: Broadcast channel name
            msg_type: Type of message
            priority: Message priority
            ttl: Time-to-live in seconds
            metadata: Additional metadata
            
        Returns:
            The sent Message object
        """
        message = Message(
            type=msg_type,
            sender=sender,
            recipient=None,  # Broadcast has no specific recipient
            content=content,
            priority=priority,
            ttl=ttl,
            metadata={**(metadata or {}), "channel": channel}
        )
        
        # Persist message
        if self._store:
            await self._store.save_message(message)
        
        # Deliver to subscribers
        async with self._sub_lock:
            subscribers = self._subscribers.get(channel, set())
            for subscriber_id in subscribers:
                # Create copy for each subscriber
                msg_copy = Message(
                    id=message.id,
                    type=message.type,
                    sender=message.sender,
                    recipient=subscriber_id,
                    content=message.content,
                    timestamp=message.timestamp,
                    priority=message.priority,
                    correlation_id=message.correlation_id,
                    ttl=message.ttl,
                    metadata={**message.metadata, "broadcast": True, "channel": channel}
                )
                
                async with self._queue_lock:
                    queue = self._get_queue(subscriber_id)
                    try:
                        queue.put_nowait(msg_copy)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for {subscriber_id}")
                
                # Trigger handler if registered
                if subscriber_id in self._handlers:
                    try:
                        self._handlers[subscriber_id](msg_copy)
                    except Exception as e:
                        logger.error(f"Handler error for {subscriber_id}: {e}")
        
        logger.debug(f"Broadcast sent: {message.id} from {sender} to channel {channel}")
        return message

    async def receive(
        self,
        recipient: str,
        timeout: Optional[float] = None,
        message_type: Optional[MessageType] = None
    ) -> Optional[Message]:
        """
        Receive a message for a recipient.
        
        Args:
            recipient: Recipient agent ID
            timeout: Timeout in seconds (None for blocking)
            message_type: Filter by message type
            
        Returns:
            Message or None if timeout
        """
        async with self._queue_lock:
            queue = self._get_queue(recipient)
        
        try:
            if timeout is not None:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            
            # Check if message type matches
            if message_type and message.type != message_type:
                # Put back and try again
                await queue.put(message)
                return None
            
            # Update delivery status
            message.delivered = True
            if self._store:
                await self._store.update_message_status(
                    message.id, delivered=True
                )
            
            return message
            
        except asyncio.TimeoutError:
            return None

    async def receive_batch(
        self,
        recipient: str,
        max_messages: int = 10,
        timeout: float = 0.0
    ) -> List[Message]:
        """
        Receive multiple messages for a recipient.
        
        Args:
            recipient: Recipient agent ID
            max_messages: Maximum number of messages
            timeout: Timeout in seconds
            
        Returns:
            List of messages
        """
        messages = []
        
        # Try to get first message with timeout
        first = await self.receive(recipient, timeout=timeout)
        if first:
            messages.append(first)
        
        # Get remaining messages without blocking
        async with self._queue_lock:
            queue = self._get_queue(recipient)
            while len(messages) < max_messages:
                try:
                    message = queue.get_nowait()
                    message.delivered = True
                    if self._store:
                        await self._store.update_message_status(
                            message.id, delivered=True
                        )
                    messages.append(message)
                except asyncio.QueueEmpty:
                    break
        
        return messages

    async def subscribe(
        self,
        recipient: str,
        channel: str = "default",
        handler: Optional[Callable[[Message], None]] = None
    ) -> None:
        """
        Subscribe to a broadcast channel.
        
        Args:
            recipient: Subscriber agent ID
            channel: Channel to subscribe to
            handler: Optional callback for broadcast messages
        """
        async with self._sub_lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = set()
            self._subscribers[channel].add(recipient)
            
            if handler:
                self._handlers[recipient] = handler
        
        logger.debug(f"{recipient} subscribed to channel {channel}")

    async def unsubscribe(self, recipient: str, channel: str = "default") -> None:
        """
        Unsubscribe from a broadcast channel.
        
        Args:
            recipient: Subscriber agent ID
            channel: Channel to unsubscribe from
        """
        async with self._sub_lock:
            if channel in self._subscribers:
                self._subscribers[channel].discard(recipient)
            
            if recipient in self._handlers:
                del self._handlers[recipient]
        
        logger.debug(f"{recipient} unsubscribed from channel {channel}")

    async def acknowledge(self, message_id: str) -> bool:
        """
        Acknowledge receipt of a message.
        
        Args:
            message_id: ID of the message to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        if self._store:
            await self._store.update_message_status(
                message_id, acknowledged=True
            )
            return True
        return False

    async def reply(
        self,
        original_message: Message,
        sender: str,
        content: Dict[str, Any],
        msg_type: MessageType = MessageType.RESULT
    ) -> Message:
        """
        Reply to a message.
        
        Args:
            original_message: The message being replied to
            sender: Sender agent ID
            content: Reply content
            msg_type: Type of reply
            
        Returns:
            The reply Message object
        """
        return await self.send(
            sender=sender,
            recipient=original_message.sender,
            content=content,
            msg_type=msg_type,
            correlation_id=original_message.id
        )

    async def get_pending_count(self, recipient: str) -> int:
        """
        Get count of pending messages for a recipient.
        
        Args:
            recipient: Recipient agent ID
            
        Returns:
            Number of pending messages
        """
        async with self._queue_lock:
            queue = self._queues.get(recipient)
            return queue.qsize() if queue else 0

    async def get_subscriber_count(self, channel: str = "default") -> int:
        """
        Get count of subscribers for a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            Number of subscribers
        """
        async with self._sub_lock:
            return len(self._subscribers.get(channel, set()))

    async def clear_queue(self, recipient: str) -> int:
        """
        Clear all messages in a recipient's queue.
        
        Args:
            recipient: Recipient agent ID
            
        Returns:
            Number of messages cleared
        """
        async with self._queue_lock:
            queue = self._queues.get(recipient)
            if queue:
                count = queue.qsize()
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                return count
            return 0

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if self._store:
                    # Clean up expired messages
                    expired = await self._store.cleanup_expired()
                    if expired:
                        logger.debug(f"Cleaned up {expired} expired messages")
                    
                    # Clean up old acknowledged messages
                    old = await self._store.cleanup_acknowledged()
                    if old:
                        logger.debug(f"Cleaned up {old} old acknowledged messages")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get message queue statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "queues": {},
            "subscribers": {},
            "persistence_enabled": self.persistence
        }
        
        async with self._queue_lock:
            for recipient, queue in self._queues.items():
                stats["queues"][recipient] = queue.qsize()
        
        async with self._sub_lock:
            for channel, subs in self._subscribers.items():
                stats["subscribers"][channel] = len(subs)
        
        return stats


# Singleton instance
_message_queue_instance: Optional[MessageQueue] = None


def get_message_queue(
    persistence: bool = True,
    **kwargs
) -> MessageQueue:
    """Get or create global message queue instance."""
    global _message_queue_instance
    if _message_queue_instance is None:
        _message_queue_instance = MessageQueue(persistence=persistence, **kwargs)
    return _message_queue_instance


def reset_message_queue() -> None:
    """Reset global message queue instance."""
    global _message_queue_instance
    _message_queue_instance = None
