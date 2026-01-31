"""Swarm pattern for Claude Agent Swarm.

This module implements the decentralized swarm coordination pattern where
agents collaborate without a central leader.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog

from claude_agent_swarm.patterns.base import Pattern, PatternConfig, PatternStatus
from claude_agent_swarm.agent import Agent
from claude_agent_swarm.message_queue import Message, MessageType, MessagePriority

logger = structlog.get_logger()


@dataclass
class SwarmPatternConfig(PatternConfig):
    """Configuration for swarm pattern."""
    
    consensus_threshold: float = 0.5
    max_rounds: int = 5
    enable_voting: bool = True
    broadcast_results: bool = True


class SwarmPattern(Pattern):
    """Decentralized swarm coordination pattern.
    
    In this pattern, agents collaborate without a central leader,
    using message passing and voting to reach consensus.
    
    Example:
        >>> pattern = SwarmPattern("my_swarm", agents, config)
        >>> result = await pattern.execute("Brainstorm ideas")
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        config: Optional[SwarmPatternConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the swarm pattern.
        
        Args:
            name: Pattern name
            agents: List of agents
            config: Swarm configuration
            **kwargs: Additional arguments for Pattern base class
        """
        super().__init__(name, agents, config or SwarmPatternConfig(), **kwargs)
        
        self.swarm_config = config or SwarmPatternConfig()
        
        logger.info(
            "swarm_pattern_initialized",
            agent_count=len(agents),
            consensus_threshold=self.swarm_config.consensus_threshold,
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using swarm pattern.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Execution result
        """
        with self._track_execution("swarm_execute"):
            self._status = PatternStatus.EXECUTING
            self._execution_count += 1
            
            try:
                logger.info(
                    "swarm_executing_task",
                    task=task[:100],
                    agent_count=len(self.agents),
                )
                
                # Step 1: All agents work on the task independently
                agent_results = await self._execute_with_all_agents(task, context)
                
                # Step 2: Share results and reach consensus
                if self.swarm_config.enable_voting:
                    consensus_result = await self._reach_consensus(
                        task, agent_results
                    )
                else:
                    # Simple aggregation without voting
                    consensus_result = self._aggregate_results(agent_results)
                
                # Step 3: Broadcast final result
                if self.swarm_config.broadcast_results and self._message_queue:
                    await self._broadcast_result(consensus_result)
                
                self._status = PatternStatus.ACTIVE
                
                return {
                    "success": True,
                    "pattern": "swarm",
                    "result": consensus_result,
                    "agent_count": len(self.agents),
                    "rounds": 1,
                }
                
            except Exception as e:
                self._status = PatternStatus.ERROR
                logger.error("swarm_execution_failed", error=str(e))
                
                return {
                    "success": False,
                    "pattern": "swarm",
                    "error": str(e),
                }
    
    async def _execute_with_all_agents(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute task with all agents in parallel.
        
        Args:
            task: Task to execute
            context: Optional context
            
        Returns:
            List of agent results
        """
        tasks = [
            self._execute_with_agent(agent, task, context)
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "agent": self.agents[i].name,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_agent(
        self,
        agent: Agent,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute task with a single agent.
        
        Args:
            agent: Agent to execute with
            task: Task to execute
            context: Optional context
            
        Returns:
            Execution result
        """
        logger.debug("agent_executing", agent=agent.name)
        
        result = await agent.execute(task, context)
        result["agent"] = agent.name
        
        return result
    
    async def _reach_consensus(
        self,
        task: str,
        agent_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Reach consensus among agents.
        
        Args:
            task: Original task
            agent_results: Results from all agents
            
        Returns:
            Consensus result
        """
        # Filter successful results
        successful_results = [
            r for r in agent_results if r.get("success", False)
        ]
        
        if not successful_results:
            return {
                "success": False,
                "error": "No successful agent results",
            }
        
        # Simple voting mechanism
        # Count similar responses (simplified)
        votes = {}
        for result in successful_results:
            content = result.get("content", "")[:100]  # Use first 100 chars as key
            votes[content] = votes.get(content, 0) + 1
        
        # Find majority
        total_votes = len(successful_results)
        majority_content = None
        majority_count = 0
        
        for content, count in votes.items():
            if count > majority_count:
                majority_count = count
                majority_content = content
        
        # Check if we have consensus
        consensus_ratio = majority_count / total_votes
        has_consensus = consensus_ratio >= self.swarm_config.consensus_threshold
        
        # Get full result for majority
        majority_result = next(
            r for r in successful_results
            if r.get("content", "")[:100] == majority_content
        )
        
        return {
            "success": True,
            "consensus": has_consensus,
            "consensus_ratio": consensus_ratio,
            "content": majority_result.get("content", ""),
            "all_results": agent_results,
            "voting_summary": {
                "total_votes": total_votes,
                "majority_votes": majority_count,
            },
        }
    
    def _aggregate_results(
        self,
        agent_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate results without voting.
        
        Args:
            agent_results: Results from all agents
            
        Returns:
            Aggregated result
        """
        successful_results = [
            r for r in agent_results if r.get("success", False)
        ]
        
        if not successful_results:
            return {
                "success": False,
                "error": "No successful agent results",
            }
        
        # Combine all results
        contents = [
            r.get("content", "") for r in successful_results
        ]
        
        return {
            "success": True,
            "content": "\n\n---\n\n".join(contents),
            "all_results": agent_results,
        }
    
    async def _broadcast_result(self, result: Dict[str, Any]) -> None:
        """Broadcast result to all agents.
        
        Args:
            result: Result to broadcast
        """
        if not self._message_queue:
            return
        
        from claude_agent_swarm.message_queue import Message
        
        message = Message(
            id=f"broadcast_{self._execution_count}",
            sender=self.name,
            recipient=None,  # Broadcast
            type=MessageType.BROADCAST,
            content={"result": result},
            priority=MessagePriority.NORMAL,
        )
        
        await self._message_queue.send(message)
        
        logger.debug("result_broadcasted", swarm=self.name)
