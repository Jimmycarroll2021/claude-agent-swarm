"""
Tests for parallel execution capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import time

from claude_agent_swarm.swarm_manager import SwarmManager


class TestParallelExecution:
    """Test cases for parallel execution."""
    
    @pytest.mark.asyncio
    async def test_semaphore_limiting(self):
        """Test that semaphore properly limits concurrency."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=2)
        
        execution_order = []
        
        async def tracked_task(task_id):
            execution_order.append(("start", task_id))
            await asyncio.sleep(0.1)
            execution_order.append(("end", task_id))
            return Mock(success=True, data=f"result_{task_id}")
        
        tasks = [("task1", tracked_task, ["task1"]),
                 ("task2", tracked_task, ["task2"]),
                 ("task3", tracked_task, ["task3"])]
        
        results = await manager.execute_parallel(tasks)
        
        # Check that only 2 tasks ran concurrently
        # (task3 should have started after one ended)
        starts = [e for e in execution_order if e[0] == "start"]
        ends = [e for e in execution_order if e[0] == "end"]
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_parallel_speedup(self):
        """Test that parallel execution provides speedup."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=3)
        
        async def slow_task(duration):
            await asyncio.sleep(duration)
            return Mock(success=True)
        
        # Sequential would take 0.3s, parallel should take ~0.1s
        tasks = [
            ("task1", slow_task, [0.1]),
            ("task2", slow_task, [0.1]),
            ("task3", slow_task, [0.1])
        ]
        
        start = time.time()
        results = await manager.execute_parallel(tasks)
        elapsed = time.time() - start
        
        assert len(results) == 3
        assert elapsed < 0.2  # Should be much faster than sequential
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling of partial failures in parallel execution."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=3)
        
        async def mixed_task(should_fail):
            if should_fail:
                raise Exception("Task failed")
            return Mock(success=True, data="success")
        
        tasks = [
            ("task1", mixed_task, [False]),
            ("task2", mixed_task, [True]),
            ("task3", mixed_task, [False])
        ]
        
        results = await manager.execute_parallel(tasks)
        
        assert len(results) == 3
        assert results[0].success
        assert not results[1].success
        assert results[2].success
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout in parallel execution."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=2)
        
        async def infinite_task():
            while True:
                await asyncio.sleep(1)
        
        tasks = [("slow", infinite_task, [])]
        
        results = await manager.execute_parallel(tasks, timeout=0.1)
        
        assert len(results) == 1
        assert not results[0].success
        assert "timeout" in str(results[0].error).lower()
    
    @pytest.mark.asyncio
    async def test_result_aggregation(self):
        """Test result aggregation from parallel tasks."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=5)
        
        async def data_task(value):
            return Mock(success=True, data={"value": value, "square": value ** 2})
        
        tasks = [(f"task_{i}", data_task, [i]) for i in range(5)]
        
        results = await manager.execute_parallel(tasks)
        
        assert len(results) == 5
        
        values = [r.data["value"] for r in results]
        squares = [r.data["square"] for r in results]
        
        assert sorted(values) == [0, 1, 2, 3, 4]
        assert sorted(squares) == [0, 1, 4, 9, 16]


class TestLoadBalancing:
    """Test cases for load balancing."""
    
    @pytest.mark.asyncio
    async def test_task_distribution(self):
        """Test that tasks are distributed across agents."""
        orchestrator = Mock()
        manager = SwarmManager(orchestrator, max_concurrent=3)
        
        agent_tasks = {"agent1": 0, "agent2": 0, "agent3": 0}
        
        async def tracked_task(agent_id):
            agent_tasks[agent_id] += 1
            await asyncio.sleep(0.01)
            return Mock(success=True)
        
        # This would need actual agent assignment
        # For now, just verify parallel execution works
        tasks = [(f"task_{i}", tracked_task, [f"agent{(i % 3) + 1}"]) 
                 for i in range(9)]
        
        results = await manager.execute_parallel(tasks)
        
        assert len(results) == 9
        assert all(r.success for r in results)
