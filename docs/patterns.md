# Orchestration Patterns Guide

This guide explains when and how to use each orchestration pattern in the Claude Agent Swarm framework.

## Pattern Selection Guide

| Pattern | Best For | Parallelism | Coordination | Complexity |
|---------|----------|-------------|--------------|------------|
| Leader | Complex, multi-domain tasks | Medium | Centralized | High |
| Swarm | Independent subtasks | Maximum | Minimal | Low |
| Pipeline | Sequential workflows | None | Linear | Medium |
| Council | Multi-perspective analysis | High | Consensus | High |

## Leader Pattern

### When to Use

- Tasks requiring different types of expertise
- Complex problems needing decomposition
- Workflows with clear specialist roles
- Scenarios requiring result synthesis

### Example Use Cases

1. **Software Development Project**
   - Architect: Design system
   - Backend Dev: Implement API
   - Frontend Dev: Build UI
   - DevOps: Setup deployment

2. **Research Report**
   - Researcher: Gather data
   - Analyst: Analyze findings
   - Writer: Create report
   - Editor: Review quality

### Implementation

```python
from claude_agent_swarm import SwarmOrchestrator
from claude_agent_swarm.patterns import LeaderPattern

orchestrator = SwarmOrchestrator()
leader = LeaderPattern(orchestrator)

# Register specialists
leader.register_specialist(
    "researcher",
    research_agent,
    capabilities=["research", "data_gathering"],
    priority=1
)

leader.register_specialist(
    "analyst",
    analysis_agent,
    capabilities=["analysis", "synthesis"],
    priority=2
)

# Execute with automatic specialist selection
result = await leader.delegate(
    "Research quantum computing advances",
    specialist_type="researcher"
)

# Or with fallback chain
result = await leader.delegate_with_fallback(
    "Analyze market trends",
    preferred_specialists=["senior_analyst", "analyst", "junior_analyst"]
)
```

### Configuration

```yaml
workflows:
  software_project:
    pattern: "leader"
    specialists:
      - name: "architect"
        template: "architect"
        capabilities: ["design", "architecture"]
      - name: "developer"
        template: "coder"
        capabilities: ["implementation", "coding"]
      - name: "tester"
        template: "tester"
        capabilities: ["testing", "qa"]
    synthesis_template: "synthesizer"
```

## Swarm Pattern

### When to Use

- Large number of independent tasks
- Embarrassingly parallel problems
- Data processing pipelines
- Search and aggregation tasks

### Example Use Cases

1. **Multi-Source Research**
   - Research 10 different topics in parallel
   - Aggregate results into comprehensive report

2. **Data Processing**
   - Process 1000 records concurrently
   - Collect and merge results

3. **Content Generation**
   - Generate 50 product descriptions
   - Review and select best ones

### Implementation

```python
from claude_agent_swarm.patterns import SwarmPattern

swarm = SwarmPattern(orchestrator, max_agents=20)

# Define tasks
tasks = [
    "Research topic A",
    "Research topic B",
    "Research topic C",
    # ... more tasks
]

# Execute in parallel
results = await swarm.execute_parallel(
    tasks,
    agent_template={
        "model": "claude-3-7-sonnet-20250219",
        "system_prompt": "You are a research specialist."
    },
    timeout=120
)

# Aggregate results
successful_results = [r for r in results if r.success]
```

### Dynamic Agent Allocation

```python
# Automatically determine optimal agent count
results = await swarm.execute_parallel(
    tasks,
    agent_allocation="dynamic",  # or "fixed:10"
    min_agents=3,
    max_agents=20
)
```

### Configuration

```yaml
workflows:
  multi_source_research:
    pattern: "swarm"
    config:
      max_agents: 20
      parallel_limit: 10
      agent_allocation: "dynamic"
      timeout: 120
      retry_failed: true
    aggregation:
      method: "synthesis"
      template: "synthesizer"
```

## Pipeline Pattern

### When to Use

- Multi-stage workflows
- Sequential processing requirements
- Stage dependencies
- Quality gates

### Example Use Cases

1. **Code Generation**
   - Stage 1: Design architecture
   - Stage 2: Implement code
   - Stage 3: Review code
   - Stage 4: Write tests

2. **Content Creation**
   - Stage 1: Research
   - Stage 2: Outline
   - Stage 3: Draft
   - Stage 4: Edit
   - Stage 5: Finalize

### Implementation

```python
from claude_agent_swarm.patterns import PipelinePattern

pipeline = PipelinePattern(orchestrator)

# Add stages
pipeline.add_stage(
    "design",
    design_agent,
    timeout=120
)

pipeline.add_stage(
    "implement",
    implementation_agent,
    timeout=180,
    condition=lambda prev_result: prev_result.success
)

pipeline.add_stage(
    "review",
    review_agent,
    timeout=120,
    condition=lambda prev_result: prev_result.data.get("complexity") == "high"
)

# Execute
result = await pipeline.execute_pipeline("Create a task queue system")

# Get pipeline status
status = pipeline.get_pipeline_status()
print(f"Progress: {status['completed_stages']}/{status['total_stages']}")
```

### Conditional Branching

```python
# Branch based on results
pipeline.add_conditional_branch(
    "evaluate",
    branches={
        "simple": (lambda r: r.data["complexity"] < 5, simple_agent),
        "complex": (lambda r: r.data["complexity"] >= 5, complex_agent)
    }
)
```

### Configuration

```yaml
workflows:
  code_generation:
    pattern: "pipeline"
    stages:
      - name: "design"
        template: "architect"
        timeout: 120
      - name: "implement"
        template: "coder"
        timeout: 180
        parallel: 3
      - name: "review"
        template: "reviewer"
        timeout: 120
        condition: "implementation.complete"
      - name: "test"
        template: "tester"
        timeout: 150
        condition: "review.passed"
```

## Council Pattern

### When to Use

- Complex decisions requiring multiple viewpoints
- Risk assessment
- Strategic planning
- Ethical considerations

### Example Use Cases

1. **Product Decision**
   - Technical feasibility
   - Business viability
   - User experience
   - Market fit

2. **Risk Assessment**
   - Technical risks
   - Business risks
   - Legal/compliance risks
   - Operational risks

### Implementation

```python
from claude_agent_swarm.patterns import CouncilPattern

council = CouncilPattern(orchestrator)

# Add perspectives
council.add_perspective(
    "technical",
    tech_agent,
    perspective_description="Technical feasibility and implementation"
)

council.add_perspective(
    "business",
    business_agent,
    perspective_description="Business value and market fit"
)

council.add_perspective(
    "user_experience",
    ux_agent,
    perspective_description="User experience and usability"
)

# Convene council
topic = "Should we build a mobile app?"
result = await council.convene_council(topic)

# Access individual perspectives
for perspective in result.perspectives:
    print(f"{perspective.name}: {perspective.data}")

# View synthesis
print(f"Consensus: {result.synthesis}")
```

### Voting Methods

```python
# Different voting strategies
result = await council.convene_council(
    topic,
    voting_method="consensus"  # or "majority", "supermajority", "weighted"
)

# Weighted voting
result = await council.convene_council(
    topic,
    voting_method="weighted",
    weights={
        "technical": 0.4,
        "business": 0.4,
        "user_experience": 0.2
    }
)
```

### Delphi Method

```python
# Iterative refinement
result = await council.convene_council(
    topic,
    method="delphi",
    rounds=3,
    convergence_threshold=0.8
)
```

### Configuration

```yaml
workflows:
  product_decision:
    pattern: "council"
    perspectives:
      - name: "technical"
        template: "tech_analyst"
        weight: 0.3
      - name: "business"
        template: "business_analyst"
        weight: 0.4
      - name: "ux"
        template: "ux_designer"
        weight: 0.3
    synthesis:
      template: "synthesizer"
      voting_method: "weighted"
```

## Hybrid Patterns

### Leader + Swarm

```python
# Leader delegates to multiple swarms
leader = LeaderPattern(orchestrator)

# Create specialized swarms
research_swarm = SwarmPattern(orchestrator, max_agents=10)
analysis_swarm = SwarmPattern(orchestrator, max_agents=5)

leader.register_specialist("research_team", research_swarm)
leader.register_specialist("analysis_team", analysis_swarm)
```

### Pipeline + Council

```python
# Review stage uses council
pipeline = PipelinePattern(orchestrator)

pipeline.add_stage("design", design_agent)
pipeline.add_stage("implement", implementation_agent)

# Multi-perspective review
review_council = CouncilPattern(orchestrator)
review_council.add_perspective("code_review", code_reviewer)
review_council.add_perspective("security", security_reviewer)
review_council.add_perspective("performance", perf_reviewer)

pipeline.add_stage("review", review_council)
```

### Swarm + Pipeline

```python
# Parallel processing followed by sequential refinement
swarm = SwarmPattern(orchestrator, max_agents=10)

# Generate multiple solutions
solutions = await swarm.execute_parallel([
    "Solution approach A",
    "Solution approach B",
    "Solution approach C"
])

# Pipeline to refine best solution
pipeline = PipelinePattern(orchestrator)
pipeline.add_stage("select", selection_agent)
pipeline.add_stage("refine", refinement_agent)
pipeline.add_stage("finalize", finalization_agent)

result = await pipeline.execute_pipeline(solutions)
```

## Best Practices

### 1. Pattern Selection

```python
# Use complexity analysis to select pattern
decomposer = TaskDecomposer()
score = decomposer.analyze_complexity(task)

if score.parallelizability_score > 0.8:
    pattern = SwarmPattern(orchestrator)
elif score.num_steps > 5:
    pattern = PipelinePattern(orchestrator)
elif score.domain_complexity > 0.7:
    pattern = CouncilPattern(orchestrator)
else:
    pattern = LeaderPattern(orchestrator)
```

### 2. Resource Management

```python
# Always set appropriate limits
swarm = SwarmPattern(
    orchestrator,
    max_agents=min(complexity_score / 10, 50),
    timeout=task_timeout
)
```

### 3. Error Handling

```python
# Handle partial failures
try:
    results = await swarm.execute_parallel(tasks)
    successful = [r for r in results if r.success]
    if len(successful) < len(results) * 0.8:
        logger.warning(f"High failure rate: {len(successful)}/{len(results)}")
except Exception as e:
    logger.error(f"Swarm execution failed: {e}")
```

### 4. Result Aggregation

```python
# Different aggregation strategies
def aggregate_results(results, strategy="concatenate"):
    if strategy == "concatenate":
        return "\n\n".join(r.data for r in results if r.success)
    elif strategy == "synthesis":
        return synthesizer.synthesize([r.data for r in results if r.success])
    elif strategy == "vote":
        return max(set(r.data for r in results if r.success), key=lambda x: sum(1 for r in results if r.data == x))
```

## Performance Comparison

| Task Type | Leader | Swarm | Pipeline | Council |
|-----------|--------|-------|----------|---------|
| 10 research queries | 120s | 25s | N/A | 35s |
| Code generation | 180s | N/A | 150s | N/A |
| Risk assessment | 90s | N/A | N/A | 60s |
| Multi-stage report | 300s | 200s | 240s | N/A |

*Times are approximate and depend on task complexity*

## Anti-Patterns

### 1. Over-Parallelization

```python
# Bad: Too many agents for simple task
swarm = SwarmPattern(orchestrator, max_agents=50)
results = await swarm.execute_parallel(["simple task"])

# Good: Match agent count to task complexity
swarm = SwarmPattern(orchestrator, max_agents=3)
results = await swarm.execute_parallel(complex_tasks)
```

### 2. Ignoring Dependencies

```python
# Bad: Running dependent tasks in parallel
swarm = SwarmPattern(orchestrator)
results = await swarm.execute_parallel([
    "Design database",
    "Implement API that uses database"  # Depends on first task
])

# Good: Use pipeline for dependent tasks
pipeline = PipelinePattern(orchestrator)
pipeline.add_stage("design", design_agent)
pipeline.add_stage("implement", implementation_agent)
```

### 3. Missing Error Handling

```python
# Bad: No error handling
results = await swarm.execute_parallel(tasks)
final = results[0].data  # May fail if results[0] failed

# Good: Check results
results = await swarm.execute_parallel(tasks)
successful = [r for r in results if r.success]
if not successful:
    raise Exception("All tasks failed")
final = successful[0].data
```
