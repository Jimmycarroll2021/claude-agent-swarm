"""Pipeline pattern for Claude Agent Swarm.

This module implements the pipeline coordination pattern where
tasks flow through a sequence of agents, each performing a specific stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog

from claude_agent_swarm.patterns.base import Pattern, PatternConfig, PatternStatus
from claude_agent_swarm.agent import Agent

logger = structlog.get_logger()


@dataclass
class PipelineConfig(PatternConfig):
    """Configuration for pipeline pattern."""
    
    stages: Optional[List[str]] = None
    pass_context: bool = True
    stop_on_error: bool = True
    collect_intermediate: bool = True


class PipelinePattern(Pattern):
    """Pipeline coordination pattern.
    
    In this pattern, tasks flow through a sequence of agents, each
    performing a specific stage of processing. The output of one stage
    becomes the input of the next.
    
    Example:
        >>> config = PipelineConfig(stages=["research", "write", "review"])
        >>> pattern = PipelinePattern("my_pipeline", agents, config)
        >>> result = await pattern.execute("Create a report")
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        config: Optional[PipelineConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the pipeline pattern.
        
        Args:
            name: Pattern name
            agents: List of agents (one per stage)
            config: Pipeline configuration
            **kwargs: Additional arguments for Pattern base class
        """
        super().__init__(name, agents, config or PipelineConfig(), **kwargs)
        
        self.pipeline_config = config or PipelineConfig()
        
        # Assign stages to agents
        if self.pipeline_config.stages:
            self._stages = self.pipeline_config.stages
        else:
            # Use agent roles as stages
            self._stages = [agent.role for agent in agents]
        
        if len(self._stages) != len(agents):
            logger.warning(
                "stage_agent_mismatch",
                stage_count=len(self._stages),
                agent_count=len(agents),
            )
        
        logger.info(
            "pipeline_pattern_initialized",
            stage_count=len(self._stages),
            stages=self._stages,
        )
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task through the pipeline.
        
        Args:
            task: Task description
            context: Optional task context
            
        Returns:
            Execution result
        """
        with self._track_execution("pipeline_execute"):
            self._status = PatternStatus.EXECUTING
            self._execution_count += 1
            
            try:
                logger.info(
                    "pipeline_executing_task",
                    task=task[:100],
                    stage_count=len(self._stages),
                )
                
                current_input = task
                pipeline_context = context or {}
                intermediate_results = []
                
                # Process through each stage
                for i, (stage, agent) in enumerate(zip(self._stages, self.agents)):
                    logger.debug(
                        "pipeline_stage_executing",
                        stage=stage,
                        agent=agent.name,
                        stage_num=i + 1,
                    )
                    
                    # Build stage prompt
                    stage_prompt = self._build_stage_prompt(
                        stage, current_input, pipeline_context
                    )
                    
                    # Execute stage
                    stage_result = await agent.execute(
                        stage_prompt,
                        context=pipeline_context if self.pipeline_config.pass_context else None,
                    )
                    
                    # Store intermediate result
                    if self.pipeline_config.collect_intermediate:
                        intermediate_results.append({
                            "stage": stage,
                            "agent": agent.name,
                            "result": stage_result,
                        })
                    
                    # Check for errors
                    if not stage_result.get("success", False):
                        logger.error(
                            "pipeline_stage_failed",
                            stage=stage,
                            agent=agent.name,
                            error=stage_result.get("error"),
                        )
                        
                        if self.pipeline_config.stop_on_error:
                            self._status = PatternStatus.ERROR
                            return {
                                "success": False,
                                "pattern": "pipeline",
                                "failed_stage": stage,
                                "failed_agent": agent.name,
                                "error": stage_result.get("error"),
                                "intermediate_results": intermediate_results,
                            }
                    
                    # Update input for next stage
                    current_input = stage_result.get("content", "")
                    
                    # Update context with stage output
                    if self.pipeline_config.pass_context:
                        pipeline_context[f"stage_{stage}_output"] = current_input
                
                self._status = PatternStatus.ACTIVE
                
                return {
                    "success": True,
                    "pattern": "pipeline",
                    "final_output": current_input,
                    "stages_completed": len(intermediate_results),
                    "intermediate_results": intermediate_results,
                }
                
            except Exception as e:
                self._status = PatternStatus.ERROR
                logger.error("pipeline_execution_failed", error=str(e))
                
                return {
                    "success": False,
                    "pattern": "pipeline",
                    "error": str(e),
                }
    
    def _build_stage_prompt(
        self,
        stage: str,
        input_data: str,
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for a pipeline stage.
        
        Args:
            stage: Stage name
            input_data: Input for this stage
            context: Pipeline context
            
        Returns:
            Stage prompt
        """
        stage_prompts = {
            "research": f"""Please research the following topic thoroughly:

{input_data}

Provide comprehensive findings including key facts, statistics, and sources.""",
            
            "analyze": f"""Please analyze the following information:

{input_data}

Provide insights, identify patterns, and draw conclusions.""",
            
            "write": f"""Please write content based on the following:

{input_data}

Create well-structured, clear, and engaging content.""",
            
            "review": f"""Please review the following content:

{input_data}

Check for accuracy, clarity, grammar, and suggest improvements.""",
            
            "edit": f"""Please edit the following content:

{input_data}

Improve clarity, fix errors, and enhance the overall quality.""",
            
            "summarize": f"""Please summarize the following:

{input_data}

Create a concise summary that captures the key points.""",
            
            "format": f"""Please format the following content:

{input_data}

Apply appropriate formatting and structure.""",
        }
        
        # Use predefined prompt if available
        if stage.lower() in stage_prompts:
            return stage_prompts[stage.lower()]
        
        # Generic stage prompt
        return f"""Stage: {stage}

Input:
{input_data}

Please process this input according to your role as the '{stage}' stage of the pipeline."""
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status.
        
        Returns:
            Pipeline status information
        """
        agent_statuses = []
        for agent, stage in zip(self.agents, self._stages):
            status = await agent.get_status()
            status["stage"] = stage
            agent_statuses.append(status)
        
        return {
            "name": self.name,
            "type": "pipeline",
            "status": self._status.value,
            "stages": self._stages,
            "agents": agent_statuses,
            "execution_count": self._execution_count,
        }
