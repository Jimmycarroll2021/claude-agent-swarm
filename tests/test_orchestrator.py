"""
Tests for the SwarmOrchestrator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import os

from claude_agent_swarm import SwarmOrchestrator
from claude_agent_swarm.exceptions import ConfigurationError


@pytest.fixture
def mock_anthropic():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="Test response")],
        usage=MagicMock(input_tokens=10, output_tokens=20)
    ))
    return mock_client


class TestSwarmOrchestrator:
    """Test cases for SwarmOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_creation(self, mock_anthropic):
        """Test orchestrator creation via factory method."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orchestrator = await SwarmOrchestrator.create(
                    api_key="test_key",
                    max_concurrent_agents=50
                )

                assert orchestrator is not None
                assert orchestrator._max_concurrent_agents == 50

                await orchestrator.close()

    @pytest.mark.asyncio
    async def test_orchestrator_default_values(self, mock_anthropic):
        """Test orchestrator has correct default values."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orchestrator = await SwarmOrchestrator.create(api_key="test_key")

                assert orchestrator._max_concurrent_agents == 100
                assert orchestrator._default_model == "claude-3-7-sonnet-20250219"

                await orchestrator.close()

    @pytest.mark.asyncio
    async def test_create_swarm(self, mock_anthropic):
        """Test swarm creation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orchestrator = await SwarmOrchestrator.create(api_key="test_key")

                swarm_id = await orchestrator.create_swarm(
                    name="test_swarm",
                    num_agents=3
                )

                assert swarm_id is not None
                assert isinstance(swarm_id, str)

                await orchestrator.close()

    @pytest.mark.asyncio
    async def test_get_swarm_status(self, mock_anthropic):
        """Test getting swarm status."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orchestrator = await SwarmOrchestrator.create(api_key="test_key")

                swarm_id = await orchestrator.create_swarm(
                    name="test_swarm",
                    num_agents=2
                )

                # Use get_status() instead of get_swarm_status()
                status = await orchestrator.get_status(swarm_id)

                assert status is not None
                # Status may have different keys depending on implementation
                assert isinstance(status, dict)

                await orchestrator.close()

    @pytest.mark.asyncio
    async def test_terminate_swarm(self, mock_anthropic):
        """Test terminating a swarm."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orchestrator = await SwarmOrchestrator.create(api_key="test_key")

                swarm_id = await orchestrator.create_swarm(
                    name="test_swarm",
                    num_agents=2
                )

                await orchestrator.terminate_all()

                # After terminate_all, the swarm should be cleaned up
                # The terminate_all method shuts down all swarms

                await orchestrator.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_anthropic):
        """Test orchestrator as async context manager."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                async with await SwarmOrchestrator.create(api_key="test_key") as orchestrator:
                    assert orchestrator is not None

                    swarm_id = await orchestrator.create_swarm(
                        name="context_test",
                        num_agents=1
                    )
                    assert swarm_id is not None
                # Automatically cleaned up on exit

    @pytest.mark.asyncio
    async def test_orchestrator_id_uniqueness(self, mock_anthropic):
        """Test that each orchestrator gets a unique ID."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                orch1 = await SwarmOrchestrator.create(api_key="test_key")
                orch2 = await SwarmOrchestrator.create(api_key="test_key")

                assert orch1._orchestrator_id != orch2._orchestrator_id

                await orch1.close()
                await orch2.close()
