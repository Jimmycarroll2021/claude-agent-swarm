"""Task Decomposer for the Claude Agent Swarm framework."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from agent import ClaudeAgent, ClaudeModel, TokenUsage
from exceptions import (
    ComplexityAnalysisError,
    DependencyError,
    TaskDecomposerError,
)

# Type variables
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class ComplexityScore:
    """Complexity score for a task."""
    
    overall: float  # 0.0 to 1.0
    cognitive: float  # Cognitive complexity
    domain: float  # Domain knowledge required
    steps: float  # Number of steps required
    dependencies: float  # Dependency complexity
    data_volume: float  # Amount of data to process
    
    @property
    def is_complex(self) -> bool:
        """Determine if task is complex enough to benefit from swarm."""
        return self.overall > 0.6
    
    @property
    def recommended_agents(self) -> int:
        """Recommend number of agents based on complexity."""
        if self.overall < 0.3:
            return 1
        elif self.overall < 0.5:
            return 2
        elif self.overall < 0.7:
            return 3
        elif self.overall < 0.85:
            return 5
        else:
            return min(10, int(self.overall * 10))


@dataclass
class Subtask:
    """Represents a decomposed subtask."""
    
    subtask_id: str
    description: str
    estimated_complexity: ComplexityScore
    dependencies: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    estimated_time: float = 0.0
    required_specialization: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """Represents task dependencies."""
    
    nodes: dict[str, Subtask] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)  # task_id -> dependencies
    
    def get_execution_order(self) -> list[list[str]]:
        """
        Get execution order as batches of independent tasks.
        
        Returns:
            List of batches, where each batch contains task IDs that can execute in parallel
        """
        remaining = set(self.nodes.keys())
        completed: set[str] = set()
        batches: list[list[str]] = []
        
        while remaining:
            # Find tasks with all dependencies satisfied
            batch = []
            for task_id in remaining:
                deps = self.edges.get(task_id, [])
                if all(dep in completed for dep in deps):
                    batch.append(task_id)
            
            if not batch:
                # Circular dependency detected
                raise DependencyError(
                    "Circular dependency detected in task graph",
                    error_code="CIRCULAR_DEPENDENCY"
                )
            
            batches.append(batch)
            completed.update(batch)
            remaining -= set(batch)
        
        return batches
    
    def get_critical_path(self) -> list[str]:
        """
        Get the critical path (longest dependency chain).
        
        Returns:
            List of task IDs in critical path order
        """
        def path_length(task_id: str, memo: dict[str, int]) -> int:
            if task_id in memo:
                return memo[task_id]
            
            deps = self.edges.get(task_id, [])
            if not deps:
                memo[task_id] = 1
                return 1
            
            max_dep_length = max(path_length(dep, memo) for dep in deps)
            memo[task_id] = max_dep_length + 1
            return memo[task_id]
        
        memo: dict[str, int] = {}
        lengths = {task_id: path_length(task_id, memo) for task_id in self.nodes}
        
        # Reconstruct critical path
        if not lengths:
            return []
        
        critical_path = []
        current = max(lengths, key=lengths.get)
        
        while current:
            critical_path.append(current)
            deps = self.edges.get(current, [])
            if not deps:
                break
            current = max(deps, key=lambda d: lengths.get(d, 0))
        
        return list(reversed(critical_path))


@dataclass
class LoadBalancePlan:
    """Plan for distributing tasks across agents."""
    
    batches: list[list[str]]  # Task IDs grouped for parallel execution
    agent_assignments: dict[str, str]  # task_id -> agent_id
    estimated_total_time: float
    estimated_total_tokens: int
    parallelization_factor: float  # Ratio of parallel to sequential work


class TaskDecomposer:
    """
    Decomposes tasks and analyzes complexity for swarm execution.
    
    The TaskDecomposer analyzes task complexity, detects dependencies,
    generates subtasks with proper granularity, and calculates load
    balancing for optimal swarm execution.
    
    Attributes:
        decomposer_id: Unique identifier for this decomposer
        model: Model used for analysis
        api_key: Anthropic API key
    
    Example:
        >>> decomposer = await TaskDecomposer.create()
        >>> complexity = await decomposer.analyze_complexity("Build a web app")
        >>> subtasks = await decomposer.decompose_task("Build a web app", complexity)
        >>> plan = await decomposer.get_execution_plan(subtasks, "swarm")
    """
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLD_SIMPLE = 0.3
    COMPLEXITY_THRESHOLD_MODERATE = 0.6
    COMPLEXITY_THRESHOLD_COMPLEX = 0.8
    
    def __init__(
        self,
        decomposer_id: str,
        model: ClaudeModel,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,  # Lower temperature for more consistent analysis
    ) -> None:
        """
        Initialize the task decomposer. Use `create()` for async initialization.
        
        Args:
            decomposer_id: Unique identifier for this decomposer
            model: Model to use for analysis
            api_key: Anthropic API key
            max_tokens: Maximum tokens for responses
            temperature: Sampling temperature
        """
        self._decomposer_id = decomposer_id
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        # State
        self._initialized = False
        self._analysis_agent: ClaudeAgent | None = None
        self._token_usage = TokenUsage()
        
        # Cache
        self._complexity_cache: dict[str, ComplexityScore] = {}
        self._decomposition_cache: dict[str, list[Subtask]] = {}
        
        logger.debug(f"TaskDecomposer {decomposer_id} initialized")
    
    @classmethod
    async def create(
        cls,
        model: ClaudeModel = "claude-3-7-sonnet-20250219",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> TaskDecomposer:
        """
        Async factory method to create and initialize a TaskDecomposer.
        
        Args:
            model: Model to use for analysis
            api_key: Anthropic API key
            max_tokens: Maximum tokens for responses
            temperature: Sampling temperature
        
        Returns:
            An initialized TaskDecomposer instance
        
        Raises:
            TaskDecomposerError: If initialization fails
        """
        decomposer_id = str(uuid4())
        
        try:
            instance = cls(
                decomposer_id=decomposer_id,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Create analysis agent
            instance._analysis_agent = await ClaudeAgent.create(
                model=model,
                system_prompt=instance._get_analysis_system_prompt(),
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            
            instance._initialized = True
            logger.info(f"TaskDecomposer {decomposer_id} created successfully")
            return instance
            
        except Exception as e:
            raise TaskDecomposerError(
                f"Failed to initialize task decomposer: {e}",
                error_code="INIT_ERROR"
            ) from e
    
    @property
    def decomposer_id(self) -> str:
        """Get the decomposer's unique identifier."""
        return self._decomposer_id
    
    @property
    def token_usage(self) -> TokenUsage:
        """Get the total token usage."""
        return self._token_usage
    
    def _get_analysis_system_prompt(self) -> str:
        """Get the system prompt for the analysis agent."""
        return """You are a task analysis expert. Your role is to:
1. Analyze task complexity across multiple dimensions
2. Decompose tasks into appropriate subtasks
3. Identify dependencies between subtasks
4. Estimate resource requirements

Provide your analysis in a structured format with clear reasoning."""
    
    async def analyze_complexity(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> ComplexityScore:
        """
        Analyze the complexity of a task.
        
        Args:
            task: The task description
            context: Additional context for analysis
            use_cache: Whether to use cached results
        
        Returns:
            ComplexityScore with detailed metrics
        
        Raises:
            ComplexityAnalysisError: If analysis fails
        """
        if not self._initialized or not self._analysis_agent:
            raise ComplexityAnalysisError(
                "TaskDecomposer not initialized",
                error_code="NOT_INITIALIZED"
            )
        
        # Check cache
        cache_key = f"{task}:{hash(str(context))}"
        if use_cache and cache_key in self._complexity_cache:
            logger.debug("Using cached complexity analysis")
            return self._complexity_cache[cache_key]
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Analyze the complexity of the following task:

Task: {task}{context_str}

Provide a detailed complexity analysis in JSON format:
{{
    "overall": <float 0-1>,
    "cognitive": <float 0-1>,
    "domain": <float 0-1>,
    "steps": <float 0-1>,
    "dependencies": <float 0-1>,
    "data_volume": <float 0-1>,
    "reasoning": "<explanation of scores>"
}}

Consider:
- Cognitive: How much reasoning and problem-solving is required?
- Domain: How much specialized knowledge is needed?
- Steps: How many distinct steps or phases are involved?
- Dependencies: How complex are the dependencies between steps?
- Data volume: How much data needs to be processed?"""
        
        try:
            response = await self._analysis_agent.execute(prompt)
            
            # Update token usage
            usage = response.get("usage", {})
            self._token_usage.input_tokens += usage.get("input_tokens", 0)
            self._token_usage.output_tokens += usage.get("output_tokens", 0)
            
            # Parse response
            content = response.get("content", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
            if json_match:
                # Try to find the complete JSON object
                try:
                    # Find matching braces
                    start = json_match.start()
                    brace_count = 0
                    end = start
                    for i, char in enumerate(content[start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = start + i + 1
                                break
                    
                    json_str = content[start:end]
                    analysis = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    # Fallback: use simple extraction
                    analysis = self._extract_complexity_scores(content)
            else:
                analysis = self._extract_complexity_scores(content)
            
            complexity = ComplexityScore(
                overall=float(analysis.get("overall", 0.5)),
                cognitive=float(analysis.get("cognitive", 0.5)),
                domain=float(analysis.get("domain", 0.5)),
                steps=float(analysis.get("steps", 0.5)),
                dependencies=float(analysis.get("dependencies", 0.5)),
                data_volume=float(analysis.get("data_volume", 0.5)),
            )
            
            # Cache result
            if use_cache:
                self._complexity_cache[cache_key] = complexity
            
            logger.debug(f"Complexity analysis: overall={complexity.overall:.2f}")
            return complexity
            
        except Exception as e:
            raise ComplexityAnalysisError(
                f"Complexity analysis failed: {e}",
                error_code="ANALYSIS_ERROR"
            ) from e
    
    def _extract_complexity_scores(self, content: str) -> dict[str, float]:
        """Extract complexity scores from unstructured text."""
        scores = {}
        
        # Look for score patterns
        patterns = {
            "overall": r'overall[:\s]+(\d+\.?\d*)',
            "cognitive": r'cognitive[:\s]+(\d+\.?\d*)',
            "domain": r'domain[:\s]+(\d+\.?\d*)',
            "steps": r'steps[:\s]+(\d+\.?\d*)',
            "dependencies": r'dependenc(?:y|ies)[:\s]+(\d+\.?\d*)',
            "data_volume": r'data[_\s]volume[:\s]+(\d+\.?\d*)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
            else:
                scores[key] = 0.5  # Default
        
        return scores
    
    async def decompose_task(
        self,
        task: str,
        complexity: ComplexityScore | None = None,
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Decompose a task into subtasks.
        
        Args:
            task: The task description
            complexity: Pre-computed complexity score (optional)
            context: Additional context
            use_cache: Whether to use cached results
        
        Returns:
            List of subtask dictionaries
        
        Raises:
            TaskDecomposerError: If decomposition fails
        """
        if not self._initialized or not self._analysis_agent:
            raise TaskDecomposerError(
                "TaskDecomposer not initialized",
                error_code="NOT_INITIALIZED"
            )
        
        # Check cache
        cache_key = f"decomp:{task}:{hash(str(context))}"
        if use_cache and cache_key in self._decomposition_cache:
            logger.debug("Using cached decomposition")
            return [
                {
                    "subtask_id": st.subtask_id,
                    "description": st.description,
                    "estimated_complexity": {
                        "overall": st.estimated_complexity.overall,
                        "cognitive": st.estimated_complexity.cognitive,
                        "domain": st.estimated_complexity.domain,
                        "steps": st.estimated_complexity.steps,
                        "dependencies": st.estimated_complexity.dependencies,
                        "data_volume": st.estimated_complexity.data_volume,
                    },
                    "dependencies": st.dependencies,
                    "estimated_tokens": st.estimated_tokens,
                    "estimated_time": st.estimated_time,
                    "required_specialization": st.required_specialization,
                    "priority": st.priority,
                }
                for st in self._decomposition_cache[cache_key]
            ]
        
        # Get complexity if not provided
        if complexity is None:
            complexity = await self.analyze_complexity(task, context, use_cache)
        
        # Simple tasks don't need decomposition
        if complexity.overall < self.COMPLEXITY_THRESHOLD_SIMPLE:
            subtask = Subtask(
                subtask_id=str(uuid4()),
                description=task,
                estimated_complexity=complexity,
                estimated_tokens=1000,
                estimated_time=30.0,
            )
            if use_cache:
                self._decomposition_cache[cache_key] = [subtask]
            return [{
                "subtask_id": subtask.subtask_id,
                "description": subtask.description,
                "estimated_complexity": {
                    "overall": complexity.overall,
                    "cognitive": complexity.cognitive,
                    "domain": complexity.domain,
                    "steps": complexity.steps,
                    "dependencies": complexity.dependencies,
                    "data_volume": complexity.data_volume,
                },
                "dependencies": [],
                "estimated_tokens": subtask.estimated_tokens,
                "estimated_time": subtask.estimated_time,
                "required_specialization": None,
                "priority": 0,
            }]
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Decompose the following task into subtasks:

Task: {task}
Complexity Score: {complexity.overall:.2f}{context_str}

Provide the decomposition in JSON format:
{{
    "subtasks": [
        {{
            "description": "<subtask description>",
            "dependencies": ["<dependency descriptions>"],
            "estimated_tokens": <estimated token count>,
            "estimated_time_seconds": <estimated time>,
            "required_specialization": "<specialization or null>",
            "priority": <priority number>
        }}
    ],
    "reasoning": "<explanation of decomposition>"
}}

Guidelines:
- Create {complexity.recommended_agents} subtasks based on complexity
- Each subtask should be independently executable
- Identify dependencies between subtasks
- Estimate token usage and time for each subtask"""
        
        try:
            response = await self._analysis_agent.execute(prompt)
            
            # Update token usage
            usage = response.get("usage", {})
            self._token_usage.input_tokens += usage.get("input_tokens", 0)
            self._token_usage.output_tokens += usage.get("output_tokens", 0)
            
            # Parse response
            content = response.get("content", "")
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    # Find matching braces
                    start = json_match.start()
                    brace_count = 0
                    end = start
                    for i, char in enumerate(content[start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = start + i + 1
                                break
                    
                    json_str = content[start:end]
                    decomposition = json.loads(json_str)
                    subtasks_data = decomposition.get("subtasks", [])
                except (json.JSONDecodeError, ValueError):
                    subtasks_data = self._extract_subtasks_fallback(content, task)
            else:
                subtasks_data = self._extract_subtasks_fallback(content, task)
            
            # Create Subtask objects
            subtasks: list[Subtask] = []
            for i, st_data in enumerate(subtasks_data):
                subtask = Subtask(
                    subtask_id=str(uuid4()),
                    description=st_data.get("description", f"Subtask {i+1}"),
                    estimated_complexity=ComplexityScore(
                        overall=st_data.get("complexity", 0.5),
                        cognitive=st_data.get("complexity", 0.5),
                        domain=st_data.get("complexity", 0.5),
                        steps=st_data.get("complexity", 0.5),
                        dependencies=0.5,
                        data_volume=0.5,
                    ),
                    dependencies=st_data.get("dependencies", []),
                    estimated_tokens=st_data.get("estimated_tokens", 1000),
                    estimated_time=st_data.get("estimated_time_seconds", 30.0),
                    required_specialization=st_data.get("required_specialization"),
                    priority=st_data.get("priority", i),
                )
                subtasks.append(subtask)
            
            # Cache result
            if use_cache:
                self._decomposition_cache[cache_key] = subtasks
            
            logger.debug(f"Decomposed task into {len(subtasks)} subtasks")
            
            return [
                {
                    "subtask_id": st.subtask_id,
                    "description": st.description,
                    "estimated_complexity": {
                        "overall": st.estimated_complexity.overall,
                        "cognitive": st.estimated_complexity.cognitive,
                        "domain": st.estimated_complexity.domain,
                        "steps": st.estimated_complexity.steps,
                        "dependencies": st.estimated_complexity.dependencies,
                        "data_volume": st.estimated_complexity.data_volume,
                    },
                    "dependencies": st.dependencies,
                    "estimated_tokens": st.estimated_tokens,
                    "estimated_time": st.estimated_time,
                    "required_specialization": st.required_specialization,
                    "priority": st.priority,
                }
                for st in subtasks
            ]
            
        except Exception as e:
            raise TaskDecomposerError(
                f"Task decomposition failed: {e}",
                error_code="DECOMPOSITION_ERROR"
            ) from e
    
    def _extract_subtasks_fallback(
        self,
        content: str,
        original_task: str,
    ) -> list[dict[str, Any]]:
        """Extract subtasks from unstructured text as fallback."""
        subtasks = []
        
        # Look for numbered or bulleted items
        lines = content.split('\n')
        current_subtask: dict[str, Any] | None = None
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered items (1. or 1) or bullet points
            if re.match(r'^(?:\d+[.\)]\s+|[-*]\s+)', line):
                if current_subtask:
                    subtasks.append(current_subtask)
                
                description = re.sub(r'^(?:\d+[.\)]\s+|[-*]\s+)', '', line)
                current_subtask = {
                    "description": description,
                    "dependencies": [],
                    "estimated_tokens": 1000,
                    "estimated_time_seconds": 30.0,
                    "required_specialization": None,
                    "priority": len(subtasks),
                }
            elif current_subtask and line:
                # Append to current description
                current_subtask["description"] += " " + line
        
        if current_subtask:
            subtasks.append(current_subtask)
        
        # If no subtasks found, return the original task
        if not subtasks:
            subtasks = [{
                "description": original_task,
                "dependencies": [],
                "estimated_tokens": 2000,
                "estimated_time_seconds": 60.0,
                "required_specialization": None,
                "priority": 0,
            }]
        
        return subtasks
    
    async def detect_dependencies(
        self,
        subtasks: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """
        Detect dependencies between subtasks.
        
        Args:
            subtasks: List of subtask dictionaries
        
        Returns:
            Dictionary mapping subtask IDs to their dependency IDs
        
        Raises:
            DependencyError: If dependency detection fails
        """
        if len(subtasks) <= 1:
            return {st["subtask_id"]: [] for st in subtasks}
        
        # Build dependency graph based on descriptions
        dependencies: dict[str, list[str]] = {}
        
        for i, subtask in enumerate(subtasks):
            subtask_id = subtask["subtask_id"]
            deps = subtask.get("dependencies", [])
            
            # Convert dependency descriptions to IDs
            dep_ids = []
            for dep_desc in deps:
                # Find matching subtask by description similarity
                for other in subtasks:
                    if other["subtask_id"] != subtask_id:
                        # Simple string matching (can be improved)
                        if dep_desc.lower() in other["description"].lower():
                            dep_ids.append(other["subtask_id"])
                            break
            
            dependencies[subtask_id] = dep_ids
        
        return dependencies
    
    async def get_execution_plan(
        self,
        subtasks: list[dict[str, Any]],
        pattern: str = "swarm",
    ) -> dict[str, Any]:
        """
        Generate an execution plan for the subtasks.
        
        Args:
            subtasks: List of subtask dictionaries
            pattern: Execution pattern ("swarm", "pipeline", "leader", "council")
        
        Returns:
            Execution plan dictionary
        """
        # Detect dependencies
        dependencies = await self.detect_dependencies(subtasks)
        
        # Build dependency graph
        graph = DependencyGraph()
        for st in subtasks:
            subtask_obj = Subtask(
                subtask_id=st["subtask_id"],
                description=st["description"],
                estimated_complexity=ComplexityScore(**st.get("estimated_complexity", {})),
                dependencies=st.get("dependencies", []),
                estimated_tokens=st.get("estimated_tokens", 1000),
                estimated_time=st.get("estimated_time", 30.0),
                required_specialization=st.get("required_specialization"),
                priority=st.get("priority", 0),
            )
            graph.nodes[st["subtask_id"]] = subtask_obj
            graph.edges[st["subtask_id"]] = dependencies.get(st["subtask_id"], [])
        
        # Get execution order
        try:
            batches = graph.get_execution_order()
        except DependencyError:
            # If circular dependency, execute sequentially
            batches = [[st["subtask_id"]] for st in subtasks]
        
        # Calculate estimates
        total_tokens = sum(
            st.get("estimated_tokens", 1000)
            for st in subtasks
        )
        
        # Calculate total time based on pattern
        if pattern == "pipeline":
            # Sequential execution
            total_time = sum(
                st.get("estimated_time", 30.0)
                for st in subtasks
            )
            parallelization = 0.0
        elif pattern == "swarm":
            # Parallel execution within batches
            total_time = sum(
                max(
                    graph.nodes[task_id].estimated_time
                    for task_id in batch
                )
                for batch in batches
            )
            sequential_time = sum(
                st.get("estimated_time", 30.0)
                for st in subtasks
            )
            parallelization = 1.0 - (total_time / sequential_time) if sequential_time > 0 else 0.0
        else:
            # Leader or council - roughly parallel
            total_time = max(
                st.get("estimated_time", 30.0)
                for st in subtasks
            ) * 1.5  # Overhead for coordination
            parallelization = 0.7
        
        # Create load balance plan
        load_plan = LoadBalancePlan(
            batches=batches,
            agent_assignments={},  # Will be filled by orchestrator
            estimated_total_time=total_time,
            estimated_total_tokens=total_tokens,
            parallelization_factor=parallelization,
        )
        
        return {
            "plan_id": str(uuid4()),
            "pattern": pattern,
            "subtasks": subtasks,
            "dependencies": dependencies,
            "execution_batches": batches,
            "critical_path": graph.get_critical_path(),
            "estimated_tokens": total_tokens,
            "estimated_time": total_time,
            "parallelization_factor": parallelization,
            "recommended_agents": min(len(subtasks), 10),
        }
    
    async def calculate_load_balance(
        self,
        subtasks: list[dict[str, Any]],
        num_agents: int,
    ) -> LoadBalancePlan:
        """
        Calculate optimal load balancing across agents.
        
        Args:
            subtasks: List of subtask dictionaries
            num_agents: Number of available agents
        
        Returns:
            LoadBalancePlan with agent assignments
        """
        if not subtasks:
            return LoadBalancePlan(
                batches=[],
                agent_assignments={},
                estimated_total_time=0.0,
                estimated_total_tokens=0,
                parallelization_factor=0.0,
            )
        
        # Detect dependencies
        dependencies = await self.detect_dependencies(subtasks)
        
        # Build dependency graph
        graph = DependencyGraph()
        for st in subtasks:
            graph.nodes[st["subtask_id"]] = Subtask(
                subtask_id=st["subtask_id"],
                description=st["description"],
                estimated_complexity=ComplexityScore(**st.get("estimated_complexity", {})),
                dependencies=st.get("dependencies", []),
                estimated_tokens=st.get("estimated_tokens", 1000),
                estimated_time=st.get("estimated_time", 30.0),
                required_specialization=st.get("required_specialization"),
                priority=st.get("priority", 0),
            )
            graph.edges[st["subtask_id"]] = dependencies.get(st["subtask_id"], [])
        
        # Get execution order
        batches = graph.get_execution_order()
        
        # Assign tasks to agents
        agent_assignments: dict[str, str] = {}
        agent_loads: dict[str, float] = {f"agent_{i}": 0.0 for i in range(num_agents)}
        
        for batch in batches:
            for task_id in batch:
                subtask = graph.nodes[task_id]
                
                # Find least loaded agent
                least_loaded = min(agent_loads, key=agent_loads.get)
                agent_assignments[task_id] = least_loaded
                agent_loads[least_loaded] += subtask.estimated_time
        
        # Calculate estimates
        total_tokens = sum(st.get("estimated_tokens", 1000) for st in subtasks)
        total_time = sum(
            max(
                graph.nodes[task_id].estimated_time
                for task_id in batch
            )
            for batch in batches
        )
        
        sequential_time = sum(st.get("estimated_time", 30.0) for st in subtasks)
        parallelization = 1.0 - (total_time / sequential_time) if sequential_time > 0 else 0.0
        
        return LoadBalancePlan(
            batches=batches,
            agent_assignments=agent_assignments,
            estimated_total_time=total_time,
            estimated_total_tokens=total_tokens,
            parallelization_factor=parallelization,
        )
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._analysis_agent:
            await self._analysis_agent.close()
            self._analysis_agent = None
        
        self._initialized = False
        logger.info(f"TaskDecomposer {self._decomposer_id} closed")
    
    async def __aenter__(self) -> TaskDecomposer:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
