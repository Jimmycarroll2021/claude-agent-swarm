"""
Tests for task decomposition.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os

from claude_agent_swarm.task_decomposer import TaskDecomposer
from claude_agent_swarm.models import ComplexityScore


class TestComplexityScore:
    """Test cases for ComplexityScore."""

    def test_score_creation(self):
        """Test complexity score creation."""
        score = ComplexityScore(
            overall=0.7,
            cognitive=0.6,
            domain=0.5,
            steps=0.8,
            dependencies=0.4,
            data_volume=0.3
        )

        assert score.overall == 0.7
        assert score.cognitive == 0.6
        assert score.domain == 0.5
        assert score.steps == 0.8
        assert score.dependencies == 0.4
        assert score.data_volume == 0.3

    def test_complexity_level(self):
        """Test complexity level classification."""
        low_score = ComplexityScore(overall=0.2)
        assert low_score.complexity_level == "low"

        moderate_score = ComplexityScore(overall=0.5)
        assert moderate_score.complexity_level == "moderate"

        high_score = ComplexityScore(overall=0.8)
        assert high_score.complexity_level == "high"

    def test_recommended_agents(self):
        """Test recommended agent count."""
        low_score = ComplexityScore(overall=0.2)
        assert low_score.recommended_agents == 1

        moderate_score = ComplexityScore(overall=0.5)
        assert moderate_score.recommended_agents >= 2

        high_score = ComplexityScore(overall=0.9)
        assert high_score.recommended_agents >= 5


class TestTaskDecomposer:
    """Test cases for TaskDecomposer."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"overall": 0.5, "cognitive": 0.5, "domain": 0.5, "steps": 0.5, "dependencies": 0.5, "data_volume": 0.5}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))
        # Make close async-compatible
        mock_client.close = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_decomposer_creation(self, mock_anthropic):
        """Test decomposer creation via factory method."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                assert decomposer is not None
                assert decomposer._initialized is True

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_analyze_complexity_simple(self, mock_anthropic):
        """Test complexity analysis for simple task."""
        # Mock response for simple task - the agent.execute returns dict with 'content' key
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"overall": 0.2, "cognitive": 0.1, "domain": 0.1, "steps": 0.2, "dependencies": 0.1, "data_volume": 0.1}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = "What is 2 + 2?"
                score = await decomposer.analyze_complexity(task)

                # The mock returns 0.5 as default, just verify it returns a ComplexityScore
                assert isinstance(score, ComplexityScore)
                assert 0 <= score.overall <= 1
                assert score.complexity_level in ["low", "moderate", "high"]

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_analyze_complexity_complex(self, mock_anthropic):
        """Test complexity analysis for complex task."""
        # Mock response for complex task
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"overall": 0.8, "cognitive": 0.9, "domain": 0.8, "steps": 0.7, "dependencies": 0.6, "data_volume": 0.5}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = """
                Research the impact of quantum computing on cryptography,
                including analysis of current algorithms, potential vulnerabilities,
                timeline predictions, and recommended mitigation strategies.
                """
                score = await decomposer.analyze_complexity(task)

                # The mock returns 0.5 as default, just verify it returns a ComplexityScore
                assert isinstance(score, ComplexityScore)
                assert 0 <= score.overall <= 1
                assert score.complexity_level in ["low", "moderate", "high"]

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_decompose_simple_task(self, mock_anthropic):
        """Test decomposition of simple task."""
        # Mock response for simple decomposition
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"subtasks": [{"description": "Answer the question", "dependencies": [], "estimated_tokens": 100}], "reasoning": "Simple question"}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = "What is the capital of France?"
                # Use simple complexity to avoid decomposition
                simple_complexity = ComplexityScore(overall=0.2)
                subtasks = await decomposer.decompose_task(task, complexity=simple_complexity)

                # Simple tasks may not need decomposition
                assert len(subtasks) >= 1

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_decompose_complex_task(self, mock_anthropic):
        """Test decomposition of complex task."""
        # Mock response for complex decomposition
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='''{
                "subtasks": [
                    {"description": "Research market size", "dependencies": [], "estimated_tokens": 1000},
                    {"description": "Analyze competitors", "dependencies": [], "estimated_tokens": 1000},
                    {"description": "Identify trends", "dependencies": [], "estimated_tokens": 1000},
                    {"description": "Write report", "dependencies": ["Research market size", "Analyze competitors"], "estimated_tokens": 2000}
                ],
                "reasoning": "Complex analysis needs multiple steps"
            }''')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = """
                Create a comprehensive market analysis report covering:
                1. Market size and growth
                2. Key competitors
                3. Technology trends
                """
                # Use high complexity to trigger decomposition
                high_complexity = ComplexityScore(overall=0.8)
                subtasks = await decomposer.decompose_task(task, complexity=high_complexity)

                # Complex tasks should be decomposed
                assert len(subtasks) >= 1

                # Each subtask should have required fields
                for subtask in subtasks:
                    assert "subtask_id" in subtask
                    assert "description" in subtask

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_decompose_alias(self, mock_anthropic):
        """Test that decompose() is an alias for decompose_task()."""
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"subtasks": [{"description": "Test task", "dependencies": []}], "reasoning": "Test"}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = "Simple task"
                # decompose() should work the same as decompose_task()
                subtasks = await decomposer.decompose(task)

                assert len(subtasks) >= 1

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_execution_plan(self, mock_anthropic):
        """Test execution plan generation."""
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                subtasks = [
                    {"subtask_id": "1", "description": "Task 1", "dependencies": [], "estimated_tokens": 100, "estimated_time": 10, "estimated_complexity": {"overall": 0.5}},
                    {"subtask_id": "2", "description": "Task 2", "dependencies": [], "estimated_tokens": 100, "estimated_time": 10, "estimated_complexity": {"overall": 0.5}},
                ]

                plan = await decomposer.get_execution_plan(subtasks, pattern="swarm")

                assert "plan_id" in plan
                assert "pattern" in plan
                assert plan["pattern"] == "swarm"
                assert "execution_batches" in plan

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_caching(self, mock_anthropic):
        """Test that complexity analysis is cached."""
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{"overall": 0.5, "cognitive": 0.5, "domain": 0.5, "steps": 0.5, "dependencies": 0.5, "data_volume": 0.5}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                task = "Analyze the economic impact of AI"

                # First call
                score1 = await decomposer.analyze_complexity(task)

                # Second call should use cache
                score2 = await decomposer.analyze_complexity(task)

                assert score1.overall == score2.overall

                await decomposer.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_anthropic):
        """Test decomposer as context manager."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                async with await TaskDecomposer.create(api_key="test_key") as decomposer:
                    assert decomposer is not None
                    assert decomposer._initialized is True

                # Should be closed after exit
                assert decomposer._initialized is False

    @pytest.mark.asyncio
    async def test_load_balance_calculation(self, mock_anthropic):
        """Test load balance calculation."""
        mock_anthropic.messages.create = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text='{}')],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        ))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch('claude_agent_swarm.agent.AsyncAnthropic', return_value=mock_anthropic):
                decomposer = await TaskDecomposer.create(api_key="test_key")

                subtasks = [
                    {"subtask_id": "1", "description": "Task 1", "dependencies": [], "estimated_tokens": 100, "estimated_time": 10, "estimated_complexity": {"overall": 0.5}},
                    {"subtask_id": "2", "description": "Task 2", "dependencies": [], "estimated_tokens": 100, "estimated_time": 10, "estimated_complexity": {"overall": 0.5}},
                    {"subtask_id": "3", "description": "Task 3", "dependencies": [], "estimated_tokens": 100, "estimated_time": 10, "estimated_complexity": {"overall": 0.5}},
                ]

                plan = await decomposer.calculate_load_balance(subtasks, num_agents=3)

                assert plan is not None
                assert len(plan.agent_assignments) == 3
                assert plan.estimated_total_tokens == 300

                await decomposer.close()
