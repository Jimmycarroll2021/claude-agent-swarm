# API Reference

## SwarmOrchestrator

### Class: `SwarmOrchestrator`

Main orchestrator for managing agent swarms.

#### Constructor

```python
SwarmOrchestrator(
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    max_agents: int = 100,
    parallel_limit: int = 10,
    telemetry: Optional[TelemetryCollector] = None,
    state_manager: Optional[StateManager] = None
)
```

**Parameters:**
- `api_key` - Anthropic API key (or from env var)
- `config` - Configuration dictionary or loaded config
- `max_agents` - Maximum number of agents allowed
- `parallel_limit` - Maximum concurrent agents
- `telemetry` - Custom telemetry collector
- `state_manager` - Custom state manager

#### Methods

##### `create_agent`

```python
async def create_agent(
    agent_id: str,
    config: Dict[str, Any]
) -> ClaudeAgent
```

Create a new agent with the specified configuration.

**Parameters:**
- `agent_id` - Unique identifier for the agent
- `config` - Agent configuration dictionary

**Returns:** `ClaudeAgent` instance

**Raises:**
- `MaxAgentsExceededError` - If max agents limit reached
- `ConfigurationError` - If configuration is invalid

##### `execute_task`

```python
async def execute_task(
    task: str,
    orchestration_mode: str = "auto",
    **kwargs
) -> TaskResult
```

Execute a task using the specified orchestration mode.

**Parameters:**
- `task` - Task description
- `orchestration_mode` - Mode: "auto", "leader", "swarm", "pipeline", "council"
- `**kwargs` - Additional parameters for the orchestration pattern

**Returns:** `TaskResult` with execution results

##### `get_status`

```python
def get_status() -> Dict[str, Any]
```

Get current status of the orchestrator and all agents.

**Returns:** Status dictionary with agent counts, active tasks, etc.

##### `terminate_all`

```python
async def terminate_all() -> None
```

Terminate all agents and clean up resources.

## ClaudeAgent

### Class: `ClaudeAgent`

Individual agent wrapper for Claude API.

#### Constructor

```python
ClaudeAgent(
    agent_id: str,
    model: str = "claude-3-7-sonnet-20250219",
    system_prompt: Optional[str] = None,
    tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    context_manager: Optional[ContextManager] = None,
    telemetry: Optional[TelemetryCollector] = None
)
```

#### Methods

##### `execute`

```python
async def execute(
    task: str,
    context: Optional[List[Dict]] = None,
    tools: Optional[List[str]] = None,
    **kwargs
) -> AgentResult
```

Execute a task.

**Parameters:**
- `task` - Task description
- `context` - Additional context messages
- `tools` - Tools to make available
- `**kwargs` - Additional parameters for Claude API

**Returns:** `AgentResult` with response data

##### `send_message`

```python
async def send_message(
    message: str,
    role: str = "user"
) -> str
```

Send a message to the agent.

**Parameters:**
- `message` - Message content
- `role` - Message role ("user", "system", "assistant")

**Returns:** Agent response

##### `get_context`

```python
def get_context() -> List[Dict[str, Any]]
```

Get current conversation context.

**Returns:** List of context messages

##### `reset_context`

```python
def reset_context() -> None
```

Clear conversation context.

## SwarmManager

### Class: `SwarmManager`

Manages parallel execution of agents.

#### Constructor

```python
SwarmManager(
    orchestrator: SwarmOrchestrator,
    max_concurrent: int = 10,
    enable_load_balancing: bool = True
)
```

#### Methods

##### `spawn_agents`

```python
async def spawn_agents(
    count: int,
    template: str,
    **kwargs
) -> List[str]
```

Spawn multiple agents from a template.

**Parameters:**
- `count` - Number of agents to spawn
- `template` - Agent template name or configuration

**Returns:** List of agent IDs

##### `execute_parallel`

```python
async def execute_parallel(
    tasks: List[Union[str, Tuple]],
    agent_template: Optional[Dict] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> List[TaskResult]
```

Execute tasks in parallel.

**Parameters:**
- `tasks` - List of task descriptions or (task, args) tuples
- `agent_template` - Template for agent creation
- `timeout` - Timeout per task in seconds

**Returns:** List of task results

##### `scale_swarm`

```python
async def scale_swarm(
    target_count: int,
    template: Optional[str] = None
) -> None
```

Dynamically scale the swarm size.

**Parameters:**
- `target_count` - Target number of agents
- `template` - Agent template for new agents

## TaskDecomposer

### Class: `TaskDecomposer`

Analyzes and decomposes tasks.

#### Methods

##### `analyze_complexity`

```python
def analyze_complexity(
    task: str,
    use_cache: bool = True
) -> ComplexityScore
```

Analyze task complexity.

**Parameters:**
- `task` - Task description
- `use_cache` - Whether to use cached results

**Returns:** `ComplexityScore` object

##### `decompose_task`

```python
def decompose_task(
    task: str,
    max_subtasks: int = 10,
    **kwargs
) -> List[Subtask]
```

Decompose a task into subtasks.

**Parameters:**
- `task` - Task description
- `max_subtasks` - Maximum number of subtasks

**Returns:** List of `Subtask` objects

##### `get_execution_plan`

```python
def get_execution_plan(
    subtasks: List[Subtask]
) -> ExecutionPlan
```

Create an execution plan for subtasks.

**Parameters:**
- `subtasks` - List of subtasks

**Returns:** `ExecutionPlan` with execution order

## Patterns

### LeaderPattern

```python
class LeaderPattern:
    def __init__(self, orchestrator: SwarmOrchestrator)
    
    def register_specialist(
        self,
        specialist_id: str,
        agent: ClaudeAgent,
        capabilities: List[str],
        priority: int = 1
    ) -> None
    
    async def delegate(
        self,
        task: str,
        specialist_type: Optional[str] = None,
        **kwargs
    ) -> TaskResult
    
    async def synthesize(
        self,
        results: List[TaskResult],
        synthesis_task: Optional[str] = None
    ) -> TaskResult
```

### SwarmPattern

```python
class SwarmPattern:
    def __init__(
        self,
        orchestrator: SwarmOrchestrator,
        max_agents: int = 10
    )
    
    async def execute_parallel(
        self,
        tasks: List[str],
        agent_template: Optional[Dict] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[TaskResult]
    
    async def aggregate_results(
        self,
        results: List[TaskResult],
        method: str = "concatenate"
    ) -> Any
```

### PipelinePattern

```python
class PipelinePattern:
    def __init__(self, orchestrator: SwarmOrchestrator)
    
    def add_stage(
        self,
        name: str,
        agent: ClaudeAgent,
        condition: Optional[Callable] = None,
        **kwargs
    ) -> None
    
    async def execute_pipeline(
        self,
        initial_input: Any,
        **kwargs
    ) -> PipelineResult
    
    def get_pipeline_status(self) -> Dict[str, Any]
```

### CouncilPattern

```python
class CouncilPattern:
    def __init__(self, orchestrator: SwarmOrchestrator)
    
    def add_perspective(
        self,
        perspective_id: str,
        agent: ClaudeAgent,
        perspective_description: str,
        weight: float = 1.0
    ) -> None
    
    async def convene_council(
        self,
        topic: str,
        voting_method: str = "consensus",
        **kwargs
    ) -> CouncilResult
    
    async def synthesize_views(
        self,
        perspectives: List[PerspectiveResult]
    ) -> SynthesisResult
```

## StateManager

### Class: `StateManager`

Manages persistent state storage.

#### Constructor

```python
StateManager(
    backend: str = "sqlite",
    db_path: Optional[str] = None,
    redis_url: Optional[str] = None,
    **kwargs
)
```

#### Methods

##### `initialize`

```python
async def initialize() -> None
```

Initialize the state manager.

##### `get`

```python
async def get(
    key: str,
    namespace: str = "default",
    default: Any = None
) -> Any
```

Get a value.

##### `set`

```python
async def set(
    key: str,
    value: Any,
    namespace: str = "default",
    ttl: Optional[int] = None
) -> None
```

Set a value.

##### `delete`

```python
async def delete(
    key: str,
    namespace: str = "default"
) -> bool
```

Delete a value.

##### `increment`

```python
async def increment(
    key: str,
    amount: int = 1,
    namespace: str = "default"
) -> int
```

Increment a counter.

##### `checkpoint`

```python
async def checkpoint(
    namespace: str = "default"
) -> str
```

Create a checkpoint.

## MessageQueue

### Class: `MessageQueue`

Inter-agent message queue.

#### Methods

##### `send`

```python
async def send(
    recipient: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    sender: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> str
```

Send a message.

##### `receive`

```python
async def receive(
    recipient: str,
    timeout: Optional[float] = None,
    message_types: Optional[List[MessageType]] = None
) -> Optional[Dict]
```

Receive a message.

##### `broadcast`

```python
async def broadcast(
    channel: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    sender: Optional[str] = None
) -> int
```

Broadcast a message.

##### `subscribe`

```python
async def subscribe(
    agent_id: str,
    channel: str
) -> None
```

Subscribe to a channel.

## Tools

### BaseTool

Abstract base class for all tools.

```python
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult
    
    @abstractmethod
    def get_schema(self) -> ToolSchema
    
    def validate_args(self, args: Dict[str, Any]) -> List[str]
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: BaseTool) -> None
    def get(self, name: str) -> BaseTool
    def list_tools(self) -> List[str]
    def grant_access(self, agent_id: str, tool_names: List[str]) -> None
```

## Exceptions

```python
class SwarmError(Exception)
class ConfigurationError(SwarmError)
class AgentError(SwarmError)
class TaskExecutionError(SwarmError)
class MaxAgentsExceededError(SwarmError)
class PatternError(SwarmError)
class StateError(SwarmError)
class MessageQueueError(SwarmError)
class ToolError(SwarmError)
class MCPError(SwarmError)
```

## Type Definitions

```python
TaskResult = NamedTuple('TaskResult', [
    ('success', bool),
    ('data', Any),
    ('error', Optional[str]),
    ('execution_time_ms', float),
    ('agent_id', Optional[str])
])

ComplexityScore = NamedTuple('ComplexityScore', [
    ('overall_score', float),
    ('token_estimate', int),
    ('num_steps', int),
    ('parallelizability_score', float),
    ('token_tier', str)
])
```
