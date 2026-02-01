"""
Tests for state management components.
"""

import pytest
import asyncio
import tempfile
import os

from claude_agent_swarm.state_manager import StateManager
from claude_agent_swarm.message_queue import MessageQueue, MessageType


class TestStateManager:
    """Test cases for StateManager."""

    @pytest.mark.asyncio
    async def test_state_manager_initialization(self):
        """Test state manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)

            assert manager._connected is False
            assert manager.namespace_prefix == "swarm"

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("test", "key1", "value1")
            value = await manager.get("test", "key1")

            assert value == "value1"

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("test", "key1", "value1")
            await manager.delete("test", "key1")
            value = await manager.get("test", "key1")

            assert value is None

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_namespacing(self):
        """Test namespace isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("ns1", "key", "value1")
            await manager.set("ns2", "key", "value2")

            value1 = await manager.get("ns1", "key")
            value2 = await manager.get("ns2", "key")

            assert value1 == "value1"
            assert value2 == "value2"

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_increment(self):
        """Test increment operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("test", "counter", 10)
            new_value = await manager.increment("test", "counter", 5)

            assert new_value == 15

            value = await manager.get("test", "counter")
            assert value == 15

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_get_all(self):
        """Test getting all values in a namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("test", "key1", "value1")
            await manager.set("test", "key2", "value2")

            all_values = await manager.get_all("test")

            assert "key1" in all_values
            assert "key2" in all_values
            assert all_values["key1"] == "value1"
            assert all_values["key2"] == "value2"

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_clear_namespace(self):
        """Test clearing a namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.connect()

            await manager.set("test", "key1", "value1")
            await manager.set("test", "key2", "value2")

            deleted = await manager.clear_namespace("test")
            assert deleted == 2

            value = await manager.get("test", "key1")
            assert value is None

            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        """Test session context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)

            async with manager.session():
                await manager.set("test", "key", "value")
                value = await manager.get("test", "key")
                assert value == "value"

            # Should be disconnected after context
            assert manager._connected is False


class TestMessageQueue:
    """Test cases for MessageQueue."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test message queue initialization."""
        queue = MessageQueue(persistence=False)

        assert queue._queues == {}
        assert queue._subscribers == {}

    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        """Test sending and receiving messages."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        await queue.send(
            sender="sender_agent",
            recipient="agent1",
            content={"task": "test"},
            msg_type=MessageType.TASK
        )

        message = await queue.receive("agent1", timeout=1.0)

        assert message is not None
        assert message.type == MessageType.TASK
        assert message.content["task"] == "test"

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcast messaging."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        # Subscribe agents
        await queue.subscribe("agent1", "updates")
        await queue.subscribe("agent2", "updates")

        # Broadcast
        await queue.broadcast(
            sender="broadcaster",
            channel="updates",
            content={"update": "news"},
            msg_type=MessageType.BROADCAST
        )

        # Both agents should receive
        msg1 = await queue.receive("agent1", timeout=1.0)
        msg2 = await queue.receive("agent2", timeout=1.0)

        assert msg1 is not None
        assert msg2 is not None
        assert msg1.content["update"] == "news"
        assert msg2.content["update"] == "news"

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_reply(self):
        """Test message replies."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        # Send original message
        original = await queue.send(
            sender="agent1",
            recipient="agent2",
            content={"task": "do something"},
            msg_type=MessageType.TASK
        )

        # Receive and reply
        message = await queue.receive("agent2", timeout=1.0)
        await queue.reply(
            original_message=message,
            sender="agent2",
            content={"result": "done"},
            msg_type=MessageType.RESULT
        )

        # Original sender receives reply
        reply = await queue.receive("agent1", timeout=1.0)

        assert reply is not None
        assert reply.content["result"] == "done"
        assert reply.correlation_id == original.id

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self):
        """Test subscribe and unsubscribe."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        await queue.subscribe("agent1", "channel1")
        count = await queue.get_subscriber_count("channel1")
        assert count == 1

        await queue.unsubscribe("agent1", "channel1")
        count = await queue.get_subscriber_count("channel1")
        assert count == 0

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_pending_count(self):
        """Test pending message count."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        await queue.send(
            sender="sender",
            recipient="agent1",
            content={"msg": 1},
            msg_type=MessageType.TASK
        )
        await queue.send(
            sender="sender",
            recipient="agent1",
            content={"msg": 2},
            msg_type=MessageType.TASK
        )

        count = await queue.get_pending_count("agent1")
        assert count == 2

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_clear_queue(self):
        """Test clearing a queue."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        await queue.send(
            sender="sender",
            recipient="agent1",
            content={"msg": 1},
            msg_type=MessageType.TASK
        )

        cleared = await queue.clear_queue("agent1")
        assert cleared == 1

        count = await queue.get_pending_count("agent1")
        assert count == 0

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        """Test receive timeout."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        # Try to receive with short timeout - should return None
        message = await queue.receive("agent1", timeout=0.1)
        assert message is None

        await queue.disconnect()

    @pytest.mark.asyncio
    async def test_receive_batch(self):
        """Test receiving multiple messages."""
        queue = MessageQueue(persistence=False)
        await queue.connect()

        # Send multiple messages
        for i in range(5):
            await queue.send(
                sender="sender",
                recipient="agent1",
                content={"msg": i},
                msg_type=MessageType.TASK
            )

        # Receive batch
        messages = await queue.receive_batch("agent1", max_messages=3, timeout=1.0)
        assert len(messages) == 3

        await queue.disconnect()
