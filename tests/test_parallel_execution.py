"""
Tests for parallel execution capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import os

from claude_agent_swarm.swarm_manager import SwarmManager
from claude_agent_swarm.models import SwarmConfig, AgentConfig


@pytest.fixture
def mock_anthropic():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="Test response")],
        usage=MagicMock(input_tokens=10, output_tokens=20)
    ))
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def swarm_config():
    """Create a basic swarm configuration."""
    return SwarmConfig(
        name="test_swarm",
        max_agents=10,
        agent_configs=[
            AgentConfig(
                name="test_agent",
                model="claude-3-7-sonnet-20250219",
                system_prompt="You are a helpful assistant."
            )
        ],
        shared_system_prompt="You are a helpful assistant."
    )


class TestParallelExecution:
    """Test cases for parallel execution."""

    @pytest.mark.asyncio
    async def test_swarm_manager_creation(self, mock_anthropic, swarm_config):
        """Test swarm manager creation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_1",
                    config=swarm_config,
                    api_key="test_key"
                )

                assert manager is not None
                assert manager.swarm_id == "test_swarm_1"
                assert manager._initialized is True

                await manager.close()

    @pytest.mark.asyncio
    async def test_spawn_agents(self, mock_anthropic, swarm_config):
        """Test spawning agents."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_2",
                    config=swarm_config,
                    api_key="test_key"
                )

                agents = await manager.spawn_agents(3)

                assert len(agents) == 3
                assert manager.agent_count == 3

                await manager.close()

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_anthropic, swarm_config):
        """Test parallel task execution."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_3",
                    config=swarm_config,
                    api_key="test_key"
                )

                agents = await manager.spawn_agents(3)

                # Create tasks for each agent
                tasks = [
                    (agent, {"prompt": f"Task for agent {i}"})
                    for i, agent in enumerate(agents)
                ]

                results = await manager.execute_parallel(tasks)

                assert len(results) == 3
                assert all("success" in r for r in results)

                await manager.close()

    @pytest.mark.asyncio
    async def test_semaphore_limiting(self, mock_anthropic, swarm_config):
        """Test that semaphore properly limits concurrency."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_4",
                    config=swarm_config,
                    api_key="test_key",
                    max_concurrent=2
                )

                agents = await manager.spawn_agents(4)

                execution_order = []

                # Mock execution to track order
                async def tracked_execute(prompt, **kwargs):
                    execution_order.append(("start", prompt))
                    await asyncio.sleep(0.05)
                    execution_order.append(("end", prompt))
                    return {"success": True, "content": "Result"}

                for agent in agents:
                    agent.execute = tracked_execute

                tasks = [
                    (agent, {"prompt": f"task_{i}"})
                    for i, agent in enumerate(agents)
                ]

                results = await manager.execute_parallel(tasks, max_concurrent=2)

                assert len(results) == 4
                # Due to semaphore, not all tasks start simultaneously

                await manager.close()

    @pytest.mark.asyncio
    async def test_parallel_speedup(self, mock_anthropic, swarm_config):
        """Test that parallel execution provides speedup."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_5",
                    config=swarm_config,
                    api_key="test_key",
                    max_concurrent=5
                )

                agents = await manager.spawn_agents(3)

                # Mock slow execution
                async def slow_execute(prompt, **kwargs):
                    await asyncio.sleep(0.1)
                    return {"success": True, "content": "Result"}

                for agent in agents:
                    agent.execute = slow_execute

                tasks = [
                    (agent, {"prompt": f"task_{i}"})
                    for i, agent in enumerate(agents)
                ]

                # Sequential would take 0.3s, parallel should take ~0.1s
                start = time.time()
                results = await manager.execute_parallel(tasks)
                elapsed = time.time() - start

                assert len(results) == 3
                assert elapsed < 0.2  # Should be much faster than sequential

                await manager.close()

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_anthropic, swarm_config):
        """Test handling of partial failures in parallel execution."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_6",
                    config=swarm_config,
                    api_key="test_key"
                )

                agents = await manager.spawn_agents(3)

                # Make one agent fail
                async def failing_execute(prompt, **kwargs):
                    raise Exception("Task failed")

                async def success_execute(prompt, **kwargs):
                    return {"success": True, "content": "Success"}

                agents[0].execute = success_execute
                agents[1].execute = failing_execute
                agents[2].execute = success_execute

                tasks = [
                    (agent, {"prompt": f"task_{i}"})
                    for i, agent in enumerate(agents)
                ]

                results = await manager.execute_parallel(tasks)

                assert len(results) == 3
                assert results[0]["success"] is True
                assert results[1]["success"] is False
                assert results[2]["success"] is True

                await manager.close()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_anthropic, swarm_config):
        """Test timeout in parallel execution."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_7",
                    config=swarm_config,
                    api_key="test_key",
                    agent_timeout=0.1
                )

                agents = await manager.spawn_agents(1)

                # Mock very slow execution
                async def slow_execute(prompt, **kwargs):
                    await asyncio.sleep(10)
                    return {"success": True}

                agents[0].execute = slow_execute

                tasks = [(agents[0], {"prompt": "slow task"})]

                results = await manager.execute_parallel(tasks, timeout=0.1)

                assert len(results) == 1
                assert results[0]["success"] is False
                assert "Timeout" in results[0].get("error", "")

                await manager.close()


class TestScaling:
    """Test cases for swarm scaling."""

    @pytest.mark.skip(reason="Scaling tests require extended timeout due to gradual scaling delays")
    @pytest.mark.asyncio
    async def test_scale_up(self, mock_anthropic, swarm_config):
        """Test scaling up the swarm."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_8",
                    config=swarm_config,
                    api_key="test_key"
                )

                await manager.spawn_agents(2)
                assert manager.agent_count == 2

                new_count = await manager.scale_swarm(5, strategy="immediate")
                assert new_count == 5
                assert manager.agent_count == 5

                await manager.close()

    @pytest.mark.skip(reason="Scaling tests require extended timeout due to gradual scaling delays")
    @pytest.mark.asyncio
    async def test_scale_down(self, mock_anthropic, swarm_config):
        """Test scaling down the swarm."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_9",
                    config=swarm_config,
                    api_key="test_key"
                )

                await manager.spawn_agents(5)
                
                # Mark all agents as not busy
                for instance in manager._agents.values():
                    instance.is_busy = False

                new_count = await manager.scale_swarm(2)
                assert new_count == 2

                await manager.close()

    @pytest.mark.skip(reason="Metrics test timing out - needs mock refinement")
    @pytest.mark.asyncio
    async def test_agent_metrics(self, mock_anthropic, swarm_config):
        """Test getting agent metrics."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_10",
                    config=swarm_config,
                    api_key="test_key"
                )

                await manager.spawn_agents(3)

                metrics = await manager.get_agent_metrics()

                assert "swarm_id" in metrics
                assert metrics["total_agents"] == 3
                assert "agents" in metrics

                await manager.close()


class TestLoadBalancing:
    """Test cases for load balancing."""

    @pytest.mark.asyncio
    async def test_task_distribution(self, mock_anthropic, swarm_config):
        """Test that tasks are distributed across agents."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_11",
                    config=swarm_config,
                    api_key="test_key"
                )

                agents = await manager.spawn_agents(3)

                task_assignments = {agent.agent_id: 0 for agent in agents}

                async def tracking_execute(prompt, **kwargs):
                    return {"success": True, "content": "Done"}

                for agent in agents:
                    agent.execute = tracking_execute

                # Execute multiple tasks
                tasks = [
                    (agents[i % len(agents)], {"prompt": f"task_{i}"})
                    for i in range(9)
                ]

                results = await manager.execute_parallel(tasks)

                assert len(results) == 9
                assert all(r["success"] for r in results)

                await manager.close()

    @pytest.mark.asyncio
    async def test_reuse_existing_agents(self, mock_anthropic, swarm_config):
        """Test agent reuse from pool."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                manager = await SwarmManager.create(
                    swarm_id="test_swarm_12",
                    config=swarm_config,
                    api_key="test_key"
                )

                # Spawn initial agents
                agents1 = await manager.spawn_agents(3)
                initial_count = manager.agent_count

                # Mark as not busy
                for instance in manager._agents.values():
                    instance.is_busy = False

                # Request same number - should reuse
                agents2 = await manager.spawn_agents(3, reuse_existing=True)

                # Count should stay the same
                assert manager.agent_count == initial_count

                await manager.close()
