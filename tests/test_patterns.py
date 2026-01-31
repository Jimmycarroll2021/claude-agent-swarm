"""
Tests for orchestration patterns.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from claude_agent_swarm.patterns import (
    LeaderPattern, SwarmPattern, PipelinePattern, CouncilPattern
)


class TestSwarmPattern:
    """Test cases for SwarmPattern."""
    
    @pytest.mark.asyncio
    async def test_swarm_initialization(self):
        """Test swarm pattern initialization."""
        orchestrator = Mock()
        swarm = SwarmPattern(orchestrator, max_agents=5)
        
        assert swarm.orchestrator == orchestrator
        assert swarm.max_agents == 5
    
    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """Test parallel task execution."""
        orchestrator = Mock()
        
        # Mock agent execution
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Test result"
        mock_result.execution_time_ms = 100
        
        orchestrator.execute_task = AsyncMock(return_value=mock_result)
        
        swarm = SwarmPattern(orchestrator, max_agents=3)
        
        tasks = ["task1", "task2", "task3"]
        results = await swarm.execute_parallel(tasks)
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_parallel_with_timeout(self):
        """Test parallel execution with timeout."""
        orchestrator = Mock()
        
        # Mock slow execution
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return Mock(success=True)
        
        orchestrator.execute_task = slow_execute
        
        swarm = SwarmPattern(orchestrator, max_agents=2)
        
        tasks = ["task1"]
        results = await swarm.execute_parallel(tasks, timeout=0.1)
        
        assert len(results) == 1
        assert not results[0].success


class TestLeaderPattern:
    """Test cases for LeaderPattern."""
    
    @pytest.mark.asyncio
    async def test_leader_initialization(self):
        """Test leader pattern initialization."""
        orchestrator = Mock()
        leader = LeaderPattern(orchestrator)
        
        assert leader.orchestrator == orchestrator
        assert leader._specialists == {}
    
    @pytest.mark.asyncio
    async def test_register_specialist(self):
        """Test registering a specialist."""
        orchestrator = Mock()
        leader = LeaderPattern(orchestrator)
        
        mock_agent = Mock()
        leader.register_specialist("researcher", mock_agent, ["research"])
        
        assert "researcher" in leader._specialists
    
    @pytest.mark.asyncio
    async def test_delegate(self):
        """Test task delegation."""
        orchestrator = Mock()
        leader = LeaderPattern(orchestrator)
        
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Research result"
        
        mock_agent.execute = AsyncMock(return_value=mock_result)
        leader.register_specialist("researcher", mock_agent, ["research"])
        
        result = await leader.delegate("Research topic", "researcher")
        
        assert result.success


class TestPipelinePattern:
    """Test cases for PipelinePattern."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline pattern initialization."""
        orchestrator = Mock()
        pipeline = PipelinePattern(orchestrator)
        
        assert pipeline.orchestrator == orchestrator
        assert pipeline._stages == []
    
    @pytest.mark.asyncio
    async def test_add_stage(self):
        """Test adding a stage."""
        orchestrator = Mock()
        pipeline = PipelinePattern(orchestrator)
        
        mock_agent = Mock()
        pipeline.add_stage("process", mock_agent)
        
        assert len(pipeline._stages) == 1
        assert pipeline._stages[0]["name"] == "process"
    
    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test pipeline execution."""
        orchestrator = Mock()
        pipeline = PipelinePattern(orchestrator)
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Processed"
        
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=mock_result)
        
        pipeline.add_stage("stage1", mock_agent)
        pipeline.add_stage("stage2", mock_agent)
        
        result = await pipeline.execute_pipeline("input")
        
        assert result.success


class TestCouncilPattern:
    """Test cases for CouncilPattern."""
    
    @pytest.mark.asyncio
    async def test_council_initialization(self):
        """Test council pattern initialization."""
        orchestrator = Mock()
        council = CouncilPattern(orchestrator)
        
        assert council.orchestrator == orchestrator
        assert council._perspectives == []
    
    @pytest.mark.asyncio
    async def test_add_perspective(self):
        """Test adding a perspective."""
        orchestrator = Mock()
        council = CouncilPattern(orchestrator)
        
        council.add_perspective("technical", Mock(), "Technical view")
        
        assert len(council._perspectives) == 1
    
    @pytest.mark.asyncio
    async def test_convene_council(self):
        """Test convening a council."""
        orchestrator = Mock()
        council = CouncilPattern(orchestrator)
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = "Perspective analysis"
        
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=mock_result)
        
        council.add_perspective("view1", mock_agent, "View 1")
        council.add_perspective("view2", mock_agent, "View 2")
        
        result = await council.convene_council("Topic")
        
        assert result.success
