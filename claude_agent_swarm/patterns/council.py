"""Council pattern for Claude Agent Swarm.

This module implements the council/debate coordination pattern where
agents discuss and debate to reach a collective decision.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog

from claude_agent_swarm.patterns.base import Pattern, PatternConfig, PatternStatus
from claude_agent_swarm.agent import Agent

logger = structlog.get_logger()


@dataclass
class CouncilConfig(PatternConfig):
    """Configuration for council pattern."""
    
    discussion_rounds: int = 3
    consensus_threshold: float = 0.7
    enable_debate: bool = True
    voting_method: str = "majority"  # majority, unanimous, weighted
    final_decision_agent: Optional[str] = None


class CouncilPattern(Pattern):
    """Council/debate coordination pattern.
    
    In this pattern, agents engage in a structured discussion to analyze
    a problem from multiple perspectives and reach a collective decision.
    
    Example:
        >>> config = CouncilConfig(discussion_rounds=5)
        >>> pattern = CouncilPattern("my_council", agents, config)
        >>> result = await pattern.execute("Should we launch this product?")
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        config: Optional[CouncilConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the council pattern.
        
        Args:
            name: Pattern name
            agents: List of agents (council members)
            config: Council configuration
            **kwargs: Additional arguments for Pattern base class
        """
        super().__init__(name, agents, config or CouncilConfig(), **kwargs)
        
        self.council_config = config or CouncilConfig()
        
        # Assign roles to council members if not specified
        self._member_roles = self._assign_roles()
        
        logger.info(
            "council_pattern_initialized",
            member_count=len(agents),
            discussion_rounds=self.council_config.discussion_rounds,
        )
    
    def _assign_roles(self) -> Dict[str, str]:
        """Assign roles to council members.
        
        Returns:
            Dictionary mapping agent names to roles
        """
        default_roles = [
            "analyst",
            "critic",
            "creative",
            "pragmatist",
            "synthesizer",
        ]
        
        roles = {}
        for i, agent in enumerate(self.agents):
            role = default_roles[i % len(default_roles)]
            roles[agent.name] = role
        
        return roles
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using council pattern.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Execution result
        """
        with self._track_execution("council_execute"):
            self._status = PatternStatus.EXECUTING
            self._execution_count += 1
            
            try:
                logger.info(
                    "council_discussing",
                    task=task[:100],
                    member_count=len(self.agents),
                    rounds=self.council_config.discussion_rounds,
                )
                
                # Step 1: Initial perspectives from all members
                perspectives = await self._gather_perspectives(task, context)
                
                # Step 2: Structured discussion rounds
                discussion_history = []
                
                for round_num in range(self.council_config.discussion_rounds):
                    logger.debug("council_round", round=round_num + 1)
                    
                    round_responses = await self._discussion_round(
                        task,
                        perspectives,
                        discussion_history,
                        round_num,
                    )
                    
                    discussion_history.append({
                        "round": round_num + 1,
                        "responses": round_responses,
                    })
                    
                    # Update perspectives with new insights
                    perspectives = round_responses
                
                # Step 3: Final voting/decision
                final_decision = await self._reach_decision(
                    task, discussion_history
                )
                
                self._status = PatternStatus.ACTIVE
                
                return {
                    "success": True,
                    "pattern": "council",
                    "decision": final_decision.get("decision", ""),
                    "confidence": final_decision.get("confidence", 0.0),
                    "reasoning": final_decision.get("reasoning", ""),
                    "member_count": len(self.agents),
                    "discussion_rounds": self.council_config.discussion_rounds,
                    "perspectives": perspectives,
                    "discussion_history": discussion_history,
                }
                
            except Exception as e:
                self._status = PatternStatus.ERROR
                logger.error("council_execution_failed", error=str(e))
                
                return {
                    "success": False,
                    "pattern": "council",
                    "error": str(e),
                }
    
    async def _gather_perspectives(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Gather initial perspectives from all council members.
        
        Args:
            task: Task to discuss
            context: Optional context
            
        Returns:
            List of perspectives
        """
        tasks = []
        for agent in self.agents:
            role = self._member_roles.get(agent.name, "member")
            prompt = self._build_perspective_prompt(task, role)
            tasks.append(self._get_agent_perspective(agent, prompt, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        perspectives = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                perspectives.append({
                    "agent": self.agents[i].name,
                    "role": self._member_roles.get(self.agents[i].name, "member"),
                    "perspective": f"Error: {str(result)}",
                    "success": False,
                })
            else:
                perspectives.append(result)
        
        return perspectives
    
    async def _get_agent_perspective(
        self,
        agent: Agent,
        prompt: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get perspective from a single agent.
        
        Args:
            agent: Agent to get perspective from
            prompt: Perspective prompt
            context: Optional context
            
        Returns:
            Perspective result
        """
        result = await agent.execute(prompt, context)
        
        return {
            "agent": agent.name,
            "role": self._member_roles.get(agent.name, "member"),
            "perspective": result.get("content", ""),
            "success": result.get("success", False),
        }
    
    def _build_perspective_prompt(self, task: str, role: str) -> str:
        """Build perspective prompt based on role.
        
        Args:
            task: Task to discuss
            role: Council member role
            
        Returns:
            Perspective prompt
        """
        role_prompts = {
            "analyst": f"""As the Analyst council member, analyze the following topic objectively:

{task}

Provide data-driven insights, identify key factors, and present a logical breakdown of the situation.""",
            
            "critic": f"""As the Critic council member, examine the following topic for potential issues:

{task}

Identify risks, challenges, weaknesses, and potential problems. Play devil's advocate.""",
            
            "creative": f"""As the Creative council member, explore innovative possibilities for:

{task}

Think outside the box, suggest novel approaches, and consider unconventional solutions.""",
            
            "pragmatist": f"""As the Pragmatist council member, consider practical aspects of:

{task}

Focus on feasibility, implementation challenges, resource requirements, and realistic outcomes.""",
            
            "synthesizer": f"""As the Synthesizer council member, look for connections and patterns in:

{task}

Identify how different aspects relate, find common themes, and help integrate diverse viewpoints.""",
        }
        
        return role_prompts.get(role, f"""As a council member, share your perspective on:

{task}

Provide your analysis and recommendations.""")
    
    async def _discussion_round(
        self,
        task: str,
        perspectives: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        round_num: int,
    ) -> List[Dict[str, Any]]:
        """Execute one round of discussion.
        
        Args:
            task: Original task
            perspectives: Current perspectives
            history: Discussion history
            round_num: Current round number
            
        Returns:
            Round responses
        """
        # Build context from previous rounds
        context = {
            "task": task,
            "previous_perspectives": perspectives,
            "discussion_history": history,
            "round": round_num + 1,
        }
        
        # Each agent responds to others' perspectives
        tasks = []
        for agent in self.agents:
            role = self._member_roles.get(agent.name, "member")
            prompt = self._build_discussion_prompt(task, perspectives, role, round_num)
            tasks.append(self._get_agent_response(agent, prompt, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append({
                    "agent": self.agents[i].name,
                    "role": self._member_roles.get(self.agents[i].name, "member"),
                    "response": f"Error: {str(result)}",
                    "success": False,
                })
            else:
                responses.append(result)
        
        return responses
    
    def _build_discussion_prompt(
        self,
        task: str,
        perspectives: List[Dict[str, Any]],
        role: str,
        round_num: int,
    ) -> str:
        """Build discussion prompt.
        
        Args:
            task: Original task
            perspectives: Other members' perspectives
            role: Agent role
            round_num: Round number
            
        Returns:
            Discussion prompt
        """
        # Format other perspectives
        perspectives_text = "\n\n".join([
            f"{p['agent']} ({p['role']}): {p.get('perspective', '')[:200]}..."
            for p in perspectives
        ])
        
        return f"""Round {round_num + 1} of Council Discussion

Topic: {task}

Other Council Members' Perspectives:
{perspectives_text}

As the {role}, respond to these perspectives:
- Acknowledge points you agree with
- Respectfully challenge points you disagree with
- Build upon ideas that seem promising
- Help move the discussion toward a conclusion

Your response:"""
    
    async def _get_agent_response(
        self,
        agent: Agent,
        prompt: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get response from an agent.
        
        Args:
            agent: Agent to get response from
            prompt: Discussion prompt
            context: Discussion context
            
        Returns:
            Response result
        """
        result = await agent.execute(prompt, context)
        
        return {
            "agent": agent.name,
            "role": self._member_roles.get(agent.name, "member"),
            "response": result.get("content", ""),
            "success": result.get("success", False),
        }
    
    async def _reach_decision(
        self,
        task: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Reach final decision based on discussion.
        
        Args:
            task: Original task
            history: Discussion history
            
        Returns:
            Final decision
        """
        # Use designated decision agent or synthesize
        if self.council_config.final_decision_agent:
            decision_agent = next(
                (a for a in self.agents if a.name == self.council_config.final_decision_agent),
                self.agents[0],
            )
        else:
            # Use synthesizer if available, otherwise first agent
            decision_agent = next(
                (a for a in self.agents if self._member_roles.get(a.name) == "synthesizer"),
                self.agents[0],
            )
        
        # Build decision prompt
        decision_prompt = self._build_decision_prompt(task, history)
        
        result = await decision_agent.execute(decision_prompt)
        
        # Parse decision
        content = result.get("content", "")
        
        return {
            "decision": content,
            "confidence": 0.8,  # Would be calculated from voting
            "reasoning": "Based on council discussion",
            "decision_agent": decision_agent.name,
        }
    
    def _build_decision_prompt(
        self,
        task: str,
        history: List[Dict[str, Any]],
    ) -> str:
        """Build final decision prompt.
        
        Args:
            task: Original task
            history: Discussion history
            
        Returns:
            Decision prompt
        """
        # Format discussion summary
        discussion_summary = ""
        for round_data in history:
            discussion_summary += f"\nRound {round_data['round']}:\n"
            for response in round_data['responses']:
                discussion_summary += f"- {response['agent']}: {response.get('response', '')[:150]}...\n"
        
        return f"""As the council's decision maker, review the full discussion and provide a final decision.

Original Topic: {task}

Discussion Summary:
{discussion_summary}

Please provide:
1. The council's final decision or recommendation
2. Key reasoning behind this decision
3. Any dissenting viewpoints to acknowledge
4. Recommended next steps

Final Decision:"""
