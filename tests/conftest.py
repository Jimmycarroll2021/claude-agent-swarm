"""
Pytest configuration and fixtures.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = Mock()
    client.messages = Mock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def mock_telemetry():
    """Create a mock telemetry collector."""
    telemetry = Mock()
    telemetry.record_event = Mock()
    telemetry.get_metrics = Mock(return_value={})
    return telemetry


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    manager = AsyncMock()
    manager.get = AsyncMock(return_value=None)
    manager.set = AsyncMock()
    manager.delete = AsyncMock()
    return manager


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "version": 1,
        "swarm": {
            "name": "Test Swarm",
            "max_agents": 10,
            "parallel_limit": 5,
            "orchestrator": {
                "model": "claude-3-7-sonnet-20250219",
                "description": "Test orchestrator"
            },
            "agent_templates": {
                "test_agent": {
                    "model": "claude-3-7-sonnet-20250219",
                    "system_prompt": "You are a test agent."
                }
            }
        }
    }
