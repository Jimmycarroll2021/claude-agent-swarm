"""Leader pattern for Claude Agent Swarm.

This module implements the leader-follower coordination pattern where
one agent coordinates the work of other agents.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog

from claude_agent_swarm.patterns.base import Pattern, PatternConfig, PatternStatus
from claude_agent_swarm.agent import Agent
from claude_agent_swarm.task_decomposer import TaskDecomposer, Task

logger = structlog.get_logger()


@dataclass
class LeaderConfig(PatternConfig):
    """Configuration for leader pattern."""
    
    leader_agent_index: int = 0
    enable_decomposition: bool = True
    parallel_execution: bool = True
    result_aggregation: bool = True


class LeaderPattern(Pattern):
    """Leader-follower coordination pattern.
    
    In this pattern, one agent acts as the leader and coordinates the
    work of follower agents. The leader decomposes tasks and assigns
    subtasks to followers.
    
    Example:
        >>> leader = agents[0]
        >>> followers = agents[1:]
        >>> pattern = LeaderPattern("my_swarm", agents, config)
        >>> result = await pattern.execute("Complex task")
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        config: Optional[LeaderConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the leader pattern.
        
        Args:
            name: Pattern name
            agents: List of agents (first is leader)
            config: Leader configuration
            **kwargs: Additional arguments for Pattern base class
        """
        super().__init__(name, agents, config or LeaderConfig(), **kwargs)
        
        self.leader_config = config or LeaderConfig()
        self._leader = agents[self.leader_config.leader_agent_index]
        self._followers = [
            a for i, a in enumerate(agents)
            if i != self.leader_config.leader_agent_index
        ]
        self._task_decomposer = TaskDecomposer()
        
        logger.info(
            "leader_pattern_initialized",
            leader=self._leader.name,
            follower_count=len(self._followers),
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using leader-follower pattern.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Execution result
        """
        with self._track_execution("leader_execute"):
            self._status = PatternStatus.EXECUTING
            self._execution_count += 1
            
            try:
                logger.info(
                    "leader_executing_task",
                    task=task[:100],
                    leader=self._leader.name,
                )
                
                # Step 1: Leader analyzes and decomposes task
                if self.leader_config.enable_decomposition:
                    decomposed_task = await self._decompose_task(task, context)
                else:
                    decomposed_task = Task(
                        id=f"task_{self._execution_count}",
                        description=task,
                    )
                
                # Step 2: Assign subtasks to followers
                if decomposed_task.subtasks and self._followers:
                    subtask_results = await self._assign_subtasks(
                        decomposed_task.subtasks
                    )
                else:
                    # Leader handles the task alone
                    result = await self._leader.execute(task, context)
                    subtask_results = [result]
                
                # Step 3: Leader aggregates results
                if self.leader_config.result_aggregation:
                    final_result = await self._aggregate_results(
                        task, subtask_results
                    )
                else:
                    final_result = {
                        "success": all(r.get("success", False) for r in subtask_results),
                        "results": subtask_results,
                    }
                
                self._status = PatternStatus.ACTIVE
                
                return {
                    "success": final_result.get("success", False),
                    "pattern": "leader",
                    "leader": self._leader.name,
                    "result": final_result,
                    "subtask_count": len(decomposed_task.subtasks),
                }
                
            except Exception as e:
                self._status = PatternStatus.ERROR
                logger.error("leader_execution_failed", error=str(e))
                
                return {
                    "success": False,
                    "pattern": "leader",
                    "error": str(e),
                }
    
    async def _decompose_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> Task:
        """Decompose task using leader.
        
        Args:
            task: Task to decompose
            context: Optional context
            
        Returns:
            Decomposed task
        """
        # Use task decomposer
        decomposed = await self._task_decomposer.decompose(
            task,
            context=context,
        )
        
        logger.debug(
            "task_decomposed",
            task_id=decomposed.id,
            subtask_count=len(decomposed.subtasks),
        )
        
        return decomposed
    
    async def _assign_subtasks(
        self,
        subtasks: List[Any],
    ) -> List[Dict[str, Any]]:
        """Assign subtasks to followers.
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            List of results
        """
        results = []
        
        if self.leader_config.parallel_execution:
            # Execute in parallel
            tasks = []
            for i, subtask in enumerate(subtasks):
                follower = self._followers[i % len(self._followers)]
                task = self._execute_subtask(follower, subtask)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            results = [
                r if not isinstance(r, Exception) else {
                    "success": False,
                    "error": str(r),
                }
                for r in results
            ]
        else:
            # Execute sequentially
            for i, subtask in enumerate(subtasks):
                follower = self._followers[i % len(self._followers)]
                result = await self._execute_subtask(follower, subtask)
                results.append(result)
        
        return results
    
    async def _execute_subtask(
        self,
        agent: Agent,
        subtask: Any,
    ) -> Dict[str, Any]:
        """Execute a subtask with an agent.
        
        Args:
            agent: Agent to execute with
            subtask: Subtask to execute
            
        Returns:
            Execution result
        """
        logger.debug(
            "executing_subtask",
            subtask_id=subtask.id,
            agent=agent.name,
        )
        
        result = await agent.execute(
            subtask.description,
            context={"subtask_id": subtask.id},
        )
        
        return result
    
    async def _aggregate_results(
        self,
        original_task: str,
        subtask_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate subtask results.
        
        Args:
            original_task: Original task
            subtask_results: List of subtask results
            
        Returns:
            Aggregated result
        """
        # Build context for aggregation
        aggregation_context = {
            "original_task": original_task,
            "subtask_results": subtask_results,
        }
        
        # Ask leader to aggregate
        aggregation_prompt = f"""Please aggregate the following subtask results into a coherent final response.

Original Task: {original_task}

Subtask Results:
"""
        
        for i, result in enumerate(subtask_results, 1):
            content = result.get("content", "")
            aggregation_prompt += f"\n{i}. {content[:500]}\n"
        
        aggregation_result = await self._leader.execute(
            aggregation_prompt,
            context=aggregation_context,
        )
        
        return aggregation_result
