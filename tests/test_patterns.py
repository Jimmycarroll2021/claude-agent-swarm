"""
Tests for orchestration patterns.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import os

from claude_agent_swarm.patterns import (
    LeaderPattern, SwarmPattern, PipelinePattern, CouncilPattern,
    PatternConfig
)
from claude_agent_swarm.patterns.leader import LeaderConfig
from claude_agent_swarm.patterns.swarm import SwarmPatternConfig
from claude_agent_swarm.patterns.pipeline import PipelineConfig
from claude_agent_swarm.patterns.council import CouncilConfig


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "mock_agent"
    agent.role = "general"
    agent.agent_id = "agent_123"
    agent.execute = AsyncMock(return_value={
        "success": True,
        "content": "Test result",
        "agent_id": "agent_123",
    })
    agent.get_status = AsyncMock(return_value={
        "agent_id": "agent_123",
        "status": "idle",
    })
    agent.terminate = AsyncMock()
    agent.close = AsyncMock()
    return agent


@pytest.fixture
def mock_agents(mock_agent):
    """Create multiple mock agents."""
    agents = []
    for i in range(3):
        agent = MagicMock()
        agent.name = f"agent_{i}"
        agent.role = "general"
        agent.agent_id = f"agent_{i}"
        agent.execute = AsyncMock(return_value={
            "success": True,
            "content": f"Result from agent {i}",
            "agent_id": f"agent_{i}",
        })
        agent.get_status = AsyncMock(return_value={
            "agent_id": f"agent_{i}",
            "status": "idle",
        })
        agent.terminate = AsyncMock()
        agent.close = AsyncMock()
        agents.append(agent)
    return agents


class TestSwarmPattern:
    """Test cases for SwarmPattern."""

    @pytest.mark.asyncio
    async def test_swarm_initialization(self, mock_agents):
        """Test swarm pattern initialization."""
        config = SwarmPatternConfig()
        swarm = SwarmPattern("test_swarm", mock_agents, config)

        assert swarm.name == "test_swarm"
        assert len(swarm.agents) == 3
        assert swarm.swarm_config.consensus_threshold == 0.5

    @pytest.mark.asyncio
    async def test_execute_swarm(self, mock_agents):
        """Test swarm task execution."""
        config = SwarmPatternConfig(enable_voting=False)
        swarm = SwarmPattern("test_swarm", mock_agents, config)

        result = await swarm.execute("Test task")

        assert result["success"] is True
        assert result["pattern"] == "swarm"
        assert result["agent_count"] == 3

    @pytest.mark.asyncio
    async def test_execute_with_consensus(self, mock_agents):
        """Test swarm execution with voting."""
        config = SwarmPatternConfig(enable_voting=True)
        swarm = SwarmPattern("test_swarm", mock_agents, config)

        result = await swarm.execute("Topic to discuss")

        assert result["success"] is True
        assert result["pattern"] == "swarm"

    @pytest.mark.asyncio
    async def test_swarm_handles_agent_failures(self, mock_agents):
        """Test swarm handles partial agent failures."""
        # Make one agent fail
        mock_agents[1].execute = AsyncMock(side_effect=Exception("Agent failed"))

        config = SwarmPatternConfig(enable_voting=False)
        swarm = SwarmPattern("test_swarm", mock_agents, config)

        result = await swarm.execute("Test task")

        # Should still succeed with remaining agents
        assert result["success"] is True


class TestLeaderPattern:
    """Test cases for LeaderPattern."""

    @pytest.mark.asyncio
    async def test_leader_initialization(self, mock_agents):
        """Test leader pattern initialization."""
        config = LeaderConfig()
        leader = LeaderPattern("test_leader", mock_agents, config)

        assert leader.name == "test_leader"
        assert leader._leader == mock_agents[0]
        assert len(leader._followers) == 2

    @pytest.mark.asyncio
    async def test_leader_with_custom_index(self, mock_agents):
        """Test leader pattern with custom leader index."""
        config = LeaderConfig(leader_agent_index=1)
        leader = LeaderPattern("test_leader", mock_agents, config)

        assert leader._leader == mock_agents[1]
        assert mock_agents[0] in leader._followers
        assert mock_agents[2] in leader._followers

    @pytest.mark.asyncio
    async def test_execute_without_decomposition(self, mock_agents):
        """Test leader execution without decomposition."""
        config = LeaderConfig(enable_decomposition=False)
        leader = LeaderPattern("test_leader", mock_agents, config)

        result = await leader.execute("Simple task")

        assert result["pattern"] == "leader"
        assert result["leader"] == mock_agents[0].name

    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_agents):
        """Test leader execution with context."""
        config = LeaderConfig(enable_decomposition=False)
        leader = LeaderPattern("test_leader", mock_agents, config)

        result = await leader.execute(
            "Task with context",
            context={"key": "value"}
        )

        assert result["pattern"] == "leader"


class TestPipelinePattern:
    """Test cases for PipelinePattern."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_agents):
        """Test pipeline pattern initialization."""
        config = PipelineConfig(stages=["research", "write", "review"])
        pipeline = PipelinePattern("test_pipeline", mock_agents, config)

        assert pipeline.name == "test_pipeline"
        assert len(pipeline._stages) == 3
        assert pipeline._stages == ["research", "write", "review"]

    @pytest.mark.asyncio
    async def test_pipeline_uses_agent_roles(self, mock_agents):
        """Test pipeline uses agent roles when no stages specified."""
        # Set different roles for agents
        mock_agents[0].role = "researcher"
        mock_agents[1].role = "writer"
        mock_agents[2].role = "reviewer"

        config = PipelineConfig()
        pipeline = PipelinePattern("test_pipeline", mock_agents, config)

        assert pipeline._stages == ["researcher", "writer", "reviewer"]

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, mock_agents):
        """Test pipeline execution."""
        config = PipelineConfig(stages=["research", "write"])
        pipeline = PipelinePattern("test_pipeline", mock_agents[:2], config)

        result = await pipeline.execute("Create a report")

        assert result["success"] is True
        assert result["pattern"] == "pipeline"
        assert result["stages_completed"] == 2

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_error(self, mock_agents):
        """Test pipeline stops on error by default."""
        mock_agents[0].execute = AsyncMock(return_value={
            "success": False,
            "error": "Stage failed"
        })

        config = PipelineConfig(
            stages=["stage1", "stage2"],
            stop_on_error=True
        )
        pipeline = PipelinePattern("test_pipeline", mock_agents[:2], config)

        result = await pipeline.execute("Test task")

        assert result["success"] is False
        assert result["failed_stage"] == "stage1"


class TestCouncilPattern:
    """Test cases for CouncilPattern."""

    @pytest.mark.asyncio
    async def test_council_initialization(self, mock_agents):
        """Test council pattern initialization."""
        config = CouncilConfig(discussion_rounds=3)
        council = CouncilPattern("test_council", mock_agents, config)

        assert council.name == "test_council"
        assert council.council_config.discussion_rounds == 3
        assert len(council._member_roles) == 3

    @pytest.mark.asyncio
    async def test_council_assigns_roles(self, mock_agents):
        """Test council assigns default roles."""
        config = CouncilConfig()
        council = CouncilPattern("test_council", mock_agents, config)

        # Check that roles are assigned
        assert all(
            council._member_roles[agent.name] in [
                "analyst", "critic", "creative", "pragmatist", "synthesizer"
            ]
            for agent in mock_agents
        )

    @pytest.mark.asyncio
    async def test_execute_council(self, mock_agents):
        """Test council execution."""
        config = CouncilConfig(discussion_rounds=2)
        council = CouncilPattern("test_council", mock_agents, config)

        result = await council.execute("Should we proceed?")

        assert result["success"] is True
        assert result["pattern"] == "council"
        assert result["member_count"] == 3
        assert result["discussion_rounds"] == 2

    @pytest.mark.asyncio
    async def test_council_with_voting(self, mock_agents):
        """Test council execution with different voting methods."""
        config = CouncilConfig(voting_method="majority")
        council = CouncilPattern("test_council", mock_agents, config)

        result = await council.execute("Decision topic")

        assert result["success"] is True
        assert "decision" in result


class TestPatternTermination:
    """Test cases for pattern termination."""

    @pytest.mark.asyncio
    async def test_pattern_terminate(self, mock_agents):
        """Test pattern terminates agents."""
        config = SwarmPatternConfig()
        pattern = SwarmPattern("test_swarm", mock_agents, config)
        await pattern.initialize()

        await pattern.terminate()

        # All agents should have terminate called
        for agent in mock_agents:
            agent.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_pattern_close(self, mock_agents):
        """Test pattern close method."""
        config = SwarmPatternConfig()
        pattern = SwarmPattern("test_swarm", mock_agents, config)
        await pattern.initialize()

        await pattern.close()

        # close() should call terminate()
        for agent in mock_agents:
            agent.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_pattern_context_manager(self, mock_agents):
        """Test pattern as context manager."""
        config = SwarmPatternConfig()

        async with SwarmPattern("test_swarm", mock_agents, config) as pattern:
            assert pattern is not None

        # After exit, agents should be terminated
        for agent in mock_agents:
            agent.terminate.assert_called()
