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
            
            assert manager._backend == "sqlite"
            assert manager._db_path == db_path
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.initialize()
            
            await manager.set("key1", "value1", namespace="test")
            value = await manager.get("key1", namespace="test")
            
            assert value == "value1"
            
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.initialize()
            
            await manager.set("key1", "value1", namespace="test")
            await manager.delete("key1", namespace="test")
            value = await manager.get("key1", namespace="test")
            
            assert value is None
            
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_namespacing(self):
        """Test namespace isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.initialize()
            
            await manager.set("key", "value1", namespace="ns1")
            await manager.set("key", "value2", namespace="ns2")
            
            value1 = await manager.get("key", namespace="ns1")
            value2 = await manager.get("key", namespace="ns2")
            
            assert value1 == "value1"
            assert value2 == "value2"
            
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_increment(self):
        """Test increment operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = StateManager(backend="sqlite", db_path=db_path)
            await manager.initialize()
            
            await manager.set("counter", 10, namespace="test")
            new_value = await manager.increment("counter", 5, namespace="test")
            
            assert new_value == 15
            
            value = await manager.get("counter", namespace="test")
            assert value == 15
            
            await manager.close()


class TestMessageQueue:
    """Test cases for MessageQueue."""
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test message queue initialization."""
        queue = MessageQueue()
        
        assert queue._queues == {}
        assert queue._subscribers == {}
    
    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        """Test sending and receiving messages."""
        queue = MessageQueue()
        await queue.initialize()
        
        await queue.send(
            recipient="agent1",
            message_type=MessageType.TASK,
            payload={"task": "test"}
        )
        
        message = await queue.receive("agent1")
        
        assert message is not None
        assert message["type"] == MessageType.TASK.value
        assert message["payload"]["task"] == "test"
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcast messaging."""
        queue = MessageQueue()
        await queue.initialize()
        
        # Subscribe agents
        await queue.subscribe("agent1", "updates")
        await queue.subscribe("agent2", "updates")
        
        # Broadcast
        await queue.broadcast(
            channel="updates",
            message_type=MessageType.BROADCAST,
            payload={"update": "news"}
        )
        
        # Both agents should receive
        msg1 = await queue.receive("agent1")
        msg2 = await queue.receive("agent2")
        
        assert msg1 is not None
        assert msg2 is not None
        assert msg1["payload"]["update"] == "news"
        
        await queue.close()
    
    @pytest.mark.asyncio
    async def test_reply(self):
        """Test message replies."""
        queue = MessageQueue()
        await queue.initialize()
        
        # Send original message
        await queue.send(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.TASK,
            payload={"task": "do something"}
        )
        
        # Receive and reply
        message = await queue.receive("agent2")
        await queue.reply(
            original_message=message,
            message_type=MessageType.RESULT,
            payload={"result": "done"}
        )
        
        # Original sender receives reply
        reply = await queue.receive("agent1")
        
        assert reply is not None
        assert reply["payload"]["result"] == "done"
        
        await queue.close()
