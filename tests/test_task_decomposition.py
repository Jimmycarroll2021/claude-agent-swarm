"""
Tests for task decomposition.
"""

import pytest
from unittest.mock import Mock

from claude_agent_swarm.task_decomposer import TaskDecomposer, ComplexityScore


class TestComplexityScore:
    """Test cases for ComplexityScore."""
    
    def test_score_calculation(self):
        """Test complexity score calculation."""
        score = ComplexityScore(
            token_estimate=5000,
            num_steps=10,
            required_tools=5,
            domain_complexity=0.7,
            uncertainty_factor=0.5
        )
        
        # Score should be weighted average
        assert 0 <= score.overall_score <= 100
        assert score.token_tier in ["small", "medium", "large", "xlarge"]
    
    def test_token_tiers(self):
        """Test token tier classification."""
        small = ComplexityScore(token_estimate=1000)
        assert small.token_tier == "small"
        
        medium = ComplexityScore(token_estimate=15000)
        assert medium.token_tier == "medium"
        
        large = ComplexityScore(token_estimate=50000)
        assert large.token_tier == "large"
        
        xlarge = ComplexityScore(token_estimate=150000)
        assert xlarge.token_tier == "xlarge"


class TestTaskDecomposer:
    """Test cases for TaskDecomposer."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = TaskDecomposer()
        
        assert decomposer._complexity_cache == {}
        assert decomposer._decomposition_cache == {}
    
    def test_analyze_complexity_simple(self):
        """Test complexity analysis for simple task."""
        decomposer = TaskDecomposer()
        
        task = "What is 2 + 2?"
        score = decomposer.analyze_complexity(task)
        
        assert score.overall_score < 30  # Should be low complexity
        assert score.token_tier == "small"
    
    def test_analyze_complexity_complex(self):
        """Test complexity analysis for complex task."""
        decomposer = TaskDecomposer()
        
        task = """
        Research the impact of quantum computing on cryptography, 
        including analysis of current algorithms, potential vulnerabilities, 
        timeline predictions, and recommended mitigation strategies for 
        organizations with large-scale deployments.
        """
        score = decomposer.analyze_complexity(task)
        
        assert score.overall_score > 50  # Should be high complexity
    
    def test_decompose_simple_task(self):
        """Test decomposition of simple task."""
        decomposer = TaskDecomposer()
        
        task = "What is the capital of France?"
        subtasks = decomposer.decompose_task(task)
        
        # Simple tasks may not need decomposition
        assert len(subtasks) <= 2
    
    def test_decompose_complex_task(self):
        """Test decomposition of complex task."""
        decomposer = TaskDecomposer()
        
        task = """
        Create a comprehensive market analysis report covering:
        1. Market size and growth
        2. Key competitors
        3. Technology trends
        4. Regulatory landscape
        5. Future predictions
        """
        subtasks = decomposer.decompose_task(task)
        
        # Complex tasks should be decomposed
        assert len(subtasks) >= 3
        
        # Each subtask should have required fields
        for subtask in subtasks:
            assert "id" in subtask
            assert "description" in subtask
            assert "estimated_complexity" in subtask
    
    def test_dependency_detection(self):
        """Test dependency detection between subtasks."""
        decomposer = TaskDecomposer()
        
        task = """
        Build a web application that:
        1. Designs the database schema
        2. Implements the backend API
        3. Creates the frontend UI
        4. Deploys to production
        """
        subtasks = decomposer.decompose_task(task)
        
        # Check for dependencies
        # Backend likely depends on schema design
        # Frontend likely depends on backend
        # Deployment depends on everything
        
        # Find subtasks by description keywords
        schema_task = next((s for s in subtasks if "schema" in s["description"].lower()), None)
        backend_task = next((s for s in subtasks if "backend" in s["description"].lower()), None)
        
        if schema_task and backend_task:
            # Backend should depend on schema
            assert backend_task["id"] in [d["to"] for d in schema_task.get("dependencies", [])] or \
                   schema_task["id"] in backend_task.get("depends_on", [])
    
    def test_caching(self):
        """Test that complexity analysis is cached."""
        decomposer = TaskDecomposer()
        
        task = "Analyze the economic impact of AI"
        
        # First call
        score1 = decomposer.analyze_complexity(task)
        
        # Second call should use cache
        score2 = decomposer.analyze_complexity(task)
        
        assert score1.overall_score == score2.overall_score
    
    def test_parallelizability(self):
        """Test parallelizability assessment."""
        decomposer = TaskDecomposer()
        
        # Independent tasks should be highly parallelizable
        independent_task = "Research 5 different topics"
        score1 = decomposer.analyze_complexity(independent_task)
        assert score1.parallelizability_score > 0.7
        
        # Sequential tasks should be less parallelizable
        sequential_task = """
        1. Design the schema
        2. Implement based on the schema
        3. Test the implementation
        """
        score2 = decomposer.analyze_complexity(sequential_task)
        assert score2.parallelizability_score < 0.5
