# Configuration Reference

## Overview

The Claude Agent Swarm framework uses YAML-based configuration files for defining swarms, agents, and workflows.

## Configuration Structure

```yaml
version: 1
swarm:
  name: "My Swarm"
  description: "Description of the swarm"
  orchestration_mode: "auto"  # auto, leader, swarm, pipeline, council
  max_agents: 100
  parallel_limit: 10

  orchestrator:
    model: "claude-3-7-sonnet-20250219"
    description: "Main orchestrator"
    context_strategy: "summarize"  # summarize, full, minimal
    
  agent_templates:
    # Define reusable agent templates
    
  workflows:
    # Define workflow patterns
    
  settings:
    # Global settings
```

## Agent Templates

### Basic Template

```yaml
agent_templates:
  researcher:
    name: "researcher"
    description: "Research specialist"
    model: "claude-3-7-sonnet-20250219"
    system_prompt: |
      You are a research specialist. Your task is to:
      1. Search for and gather information
      2. Provide factual, well-sourced information
      3. Structure findings clearly
    tools:
      - web_search
      - read_file
    output_format: "structured_json"
    parameters:
      max_tokens: 2000
      temperature: 0.3
    timeout: 120
    retry_policy:
      max_retries: 3
      backoff_factor: 2
```

### Advanced Template

```yaml
agent_templates:
  coder:
    name: "coder"
    description: "Software developer"
    model: "claude-3-7-sonnet-20250219"
    system_prompt: |
      You are an expert developer...
    tools:
      - read_file
      - write_file
      - edit_file
      - code_execution
    tool_permissions:
      read_file: ["/home/user/project"]
      write_file: ["/home/user/project"]
    output_format: "code"
    parameters:
      max_tokens: 4000
      temperature: 0.2
      top_p: 0.95
    context_management:
      max_context_tokens: 80000
      summarization_threshold: 60000
      preserve_keywords: ["decision", "requirement", "constraint"]
    timeout: 180
    retry_policy:
      max_retries: 2
      backoff_factor: 1.5
      retry_on: ["timeout", "rate_limit"]
```

## Workflows

### Swarm Workflow

```yaml
workflows:
  multi_source_research:
    description: "Research from multiple sources"
    pattern: "swarm"
    steps:
      - name: "spawn_researchers"
        action: "spawn_agents"
        config:
          template: "researcher"
          count: "dynamic"
          count_formula: "min(max(3, complexity_score / 10), 10)"
          
      - name: "execute_research"
        action: "execute_parallel"
        config:
          timeout: 120
          retry_failed: true
          max_retries: 2
          
      - name: "collect_results"
        action: "collect_results"
        config:
          timeout: 300
          require_all: false
          min_success_rate: 0.8
          
      - name: "synthesize"
        action: "execute_agent"
        config:
          template: "synthesizer"
          timeout: 180
```

### Pipeline Workflow

```yaml
workflows:
  code_generation:
    description: "Generate code through multiple stages"
    pattern: "pipeline"
    stages:
      - name: "design"
        template: "architect"
        timeout: 120
        output_key: "design_doc"
        
      - name: "implement"
        template: "coder"
        timeout: 180
        parallel: 3
        condition: "design.success"
        input_from: "design_doc"
        
      - name: "review"
        template: "reviewer"
        timeout: 120
        condition: "implementation.complete"
        
      - name: "test"
        template: "tester"
        timeout: 150
        condition: "review.passed"
```

### Council Workflow

```yaml
workflows:
  strategic_decision:
    description: "Make decision from multiple perspectives"
    pattern: "council"
    perspectives:
      - name: "technical"
        template: "tech_analyst"
        weight: 0.3
        
      - name: "business"
        template: "business_analyst"
        weight: 0.4
        
      - name: "risk"
        template: "risk_analyst"
        weight: 0.3
        
    synthesis:
      template: "synthesizer"
      voting_method: "weighted"  # consensus, majority, supermajority
      consensus_threshold: 0.7
```

## Settings

### Global Settings

```yaml
settings:
  # Telemetry
  enable_telemetry: true
  telemetry_export_path: "./telemetry"
  
  # Dashboard
  enable_dashboard: true
  dashboard_refresh_rate: 1.0
  
  # State Management
  state_backend: "sqlite"  # sqlite, redis, memory
  sqlite_db_path: "./data/swarm_state.db"
  redis_url: "redis://localhost:6379/0"
  
  # Logging
  log_level: "INFO"
  log_format: "json"  # json, text
  log_file: "./logs/swarm.log"
  
  # Rate Limiting
  enable_rate_limiter: true
  anthropic_rate_limit: 60  # requests per minute
  
  # Security
  allowed_paths: ["/home/user/project"]
  block_symlinks: true
  max_file_size: 10485760  # 10MB
  
  # Performance
  checkpoint_interval: 60
  fail_fast: false
  max_consecutive_failures: 5
```

## MCP Server Configuration

```yaml
mcp_servers:
  brave-search:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
    transport: "stdio"
    timeout: 30
    auto_connect: true
    
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    transport: "stdio"
    
  postgres:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    transport: "stdio"
```

## Environment Variable Substitution

Configuration files support environment variable substitution:

```yaml
swarm:
  orchestrator:
    model: "${ORCHESTRATOR_MODEL:-claude-3-7-sonnet-20250219}"
    
  settings:
    log_level: "${LOG_LEVEL:-INFO}"
    anthropic_api_key: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
```

Syntax:
- `${VAR}` - Substitute variable
- `${VAR:-default}` - Default value if not set
- `${VAR:?error}` - Error if not set

## Loading Configuration

### From File

```python
from claude_agent_swarm import ConfigLoader

config = ConfigLoader.load("configs/my_swarm.yml")
```

### From Dictionary

```python
config_dict = {
    "version": 1,
    "swarm": {
        "name": "My Swarm",
        "max_agents": 50
    }
}

config = ConfigLoader.from_dict(config_dict)
```

### Validation

```python
# Validate configuration
errors = ConfigLoader.validate(config)
if errors:
    print(f"Validation errors: {errors}")
```

## Complete Example

```yaml
version: 1
swarm:
  name: "Research & Analysis Swarm"
  description: "Multi-source research with parallel agents"
  orchestration_mode: "hybrid"
  max_agents: 20
  parallel_limit: 10

  orchestrator:
    model: "claude-3-7-sonnet-20250219"
    description: "Research coordinator"
    context_strategy: "summarize"

  agent_templates:
    researcher:
      model: "claude-3-7-sonnet-20250219"
      system_prompt: |
        You are a research specialist...
      tools:
        - web_search
        - read_file
      timeout: 120

    synthesizer:
      model: "claude-3-7-sonnet-20250219"
      system_prompt: |
        You are a synthesis specialist...
      timeout: 180

  workflows:
    multi_source_research:
      pattern: "swarm"
      steps:
        - spawn_agents:
            template: "researcher"
            count: 5
        - collect_results:
            timeout: 300
        - synthesize:
            template: "synthesizer"

  settings:
    enable_telemetry: true
    enable_dashboard: true
    state_backend: "sqlite"
    log_level: "INFO"
```
