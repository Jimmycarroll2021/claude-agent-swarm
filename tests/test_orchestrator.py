"""
Tests for the SwarmOrchestrator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from claude_agent_swarm import SwarmOrchestrator
from claude_agent_swarm.exceptions import ConfigurationError


class TestSwarmOrchestrator:
    """Test cases for SwarmOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_anthropic_client):
        """Test orchestrator initialization."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(api_key="test_key")
            
            assert orchestrator.api_key == "test_key"
            assert orchestrator.max_agents == 100
            assert orchestrator.parallel_limit == 10
            assert orchestrator._agents == {}
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_config(self, sample_config, mock_anthropic_client):
        """Test orchestrator initialization with config."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(config=sample_config)
            
            assert orchestrator.config == sample_config
            assert orchestrator.max_agents == sample_config["swarm"]["max_agents"]
    
    @pytest.mark.asyncio
    async def test_create_agent(self, mock_anthropic_client):
        """Test agent creation."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(api_key="test_key")
            
            agent_config = {
                "model": "claude-3-7-sonnet-20250219",
                "system_prompt": "You are a test agent."
            }
            
            agent = await orchestrator.create_agent("test_agent", agent_config)
            
            assert agent is not None
            assert "test_agent" in orchestrator._agents
    
    @pytest.mark.asyncio
    async def test_max_agents_limit(self, mock_anthropic_client):
        """Test maximum agents limit."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(api_key="test_key", max_agents=2)
            
            agent_config = {"model": "claude-3-7-sonnet-20250219"}
            
            await orchestrator.create_agent("agent1", agent_config)
            await orchestrator.create_agent("agent2", agent_config)
            
            with pytest.raises(Exception) as exc_info:
                await orchestrator.create_agent("agent3", agent_config)
            
            assert "maximum" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_terminate_all(self, mock_anthropic_client):
        """Test terminating all agents."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(api_key="test_key")
            
            agent_config = {"model": "claude-3-7-sonnet-20250219"}
            await orchestrator.create_agent("agent1", agent_config)
            await orchestrator.create_agent("agent2", agent_config)
            
            await orchestrator.terminate_all()
            
            assert len(orchestrator._agents) == 0
    
    @pytest.mark.asyncio
    async def test_get_status(self, mock_anthropic_client):
        """Test getting orchestrator status."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            orchestrator = SwarmOrchestrator(api_key="test_key")
            
            agent_config = {"model": "claude-3-7-sonnet-20250219"}
            await orchestrator.create_agent("agent1", agent_config)
            
            status = orchestrator.get_status()
            
            assert "agents" in status
            assert status["agents"]["total"] == 1


class TestOrchestratorContextManager:
    """Test async context manager support."""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_anthropic_client):
        """Test using orchestrator as async context manager."""
        with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            async with SwarmOrchestrator(api_key="test_key") as orchestrator:
                assert orchestrator is not None
                agent_config = {"model": "claude-3-7-sonnet-20250219"}
                await orchestrator.create_agent("test", agent_config)
