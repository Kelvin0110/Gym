(concepts-training-architecture)=
# Training Architecture

NeMo Gym separates training data generation from training frameworks through a decoupled service architecture. This design addresses challenges specific to generating training data for agentic tasks involving multi-turn interactions and tool calling.

:::{seealso}
For implementation details, see {doc}`../../training/index`. For general architecture, see {doc}`../architecture`.
:::

---

## Architecture Overview

**Separation of Concerns**:

NeMo Gym and your training framework (VeRL, NeMo-RL, OpenRLHF, TRL) handle different responsibilities:

```{list-table}
:header-rows: 1
:widths: 50 50

* - Training Framework
  - NeMo Gym
* - Sends tasks
  - Receives tasks
* - Receives trajectories + rewards
  - Generates trajectories + rewards
* - Handles backpropagation
  - Handles multi-turn coordination
* - Updates model weights
  - Executes tool calling loops
* - Manages training loop
  - Manages model-environment interaction
```

**Data Flow**:

```python
# Training framework → NeMo Gym
response = await server_client.post("/run", task)

# NeMo Gym processes task (multi-turn coordination, tool calling)

# NeMo Gym → Training framework  
result = await response.json()  # Contains trajectory + reward
```

---

## Why Decoupled Architecture

NeMo Gym's decoupled design provides six key capabilities for training data generation:

### 1. Multi-Turn Coordination

**Challenge**: Agentic tasks require multiple rounds of model generation and tool execution

**Solution**: Agent layer manages tool-calling loops independently from training

**Implementation**:

```python
while True:
    step += 1
    
    # Call model
    model_response = await self.server_client.post(
        server_name=self.config.model_server.name,
        url_path="/v1/responses",
        json=new_body
    )
    
    # Execute tool calls
    for tool_call in all_fn_calls:
        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{tool_call.name}",
            json=json.loads(tool_call.arguments)
        )
        new_outputs.append(tool_response)
    
    # Check completion
    if self.config.max_steps and step >= self.config.max_steps:
        break
```

**Benefit**: Training frameworks don't need custom logic for multi-turn flows

---

### 2. HTTP-Based Integration

**Challenge**: Integrating existing agent systems into training pipelines

**Solution**: All components (Models, Resources, Agents) are independent HTTP services

**Implementation**:

```python
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app
```

**Benefit**: Integrate existing agent frameworks (NAT, LangChain, LangGraph) without reimplementation

---

### 3. Interleaved Processing

**Challenge**: Synchronous batch processing creates bottlenecks for multi-turn tasks

**Solution**: Model and environment calls interleave within each rollout step

**Implementation**:

```python
while True:
    # 1. Model generates
    model_response = await self.server_client.post(
        server_name=self.config.model_server.name,
        url_path="/v1/responses"
    )
    
    # 2. Execute tools immediately (no batch waiting)
    for output_function_call in all_fn_calls:
        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{output_function_call.name}"
        )
        tool_response = NeMoGymFunctionCallOutput(
            type="function_call_output",
            call_id=output_function_call.call_id,
            output=(await api_response.content.read()).decode()
        )
        new_outputs.append(tool_response)
    
    # 3. Feed results back to model
    new_body = body.model_copy(update={"input": body.input + new_outputs})
```

**Benefit**: Process tasks as they arrive—no waiting for all models to finish, then all environments

---

### 4. Standardized Output Format

**Challenge**: Different models require model-specific parsing for tool calls and reasoning

**Solution**: All models return structured `NeMoGymResponse` objects following OpenAI Responses API

**Implementation**:

```python
# OpenAI model
return NeMoGymResponse(
    id=f"resp_{uuid4().hex}",
    created_at=int(time()),
    model=self.config.openai_model,
    object="response",
    output=response_output_dicts,
    ...
)

# vLLM model (different backend, same output format)
return NeMoGymResponse(
    id=f"resp_{uuid4().hex}",
    created_at=int(time()),
    model=body.model,
    object="response",
    output=response_output_dicts,
    ...
)
```

**Benefit**: Swap models via configuration—no parsing logic changes

---

### 5. Independent Testing

**Challenge**: Testing environments requires spinning up full training infrastructure

**Solution**: Resource servers run as standalone HTTP servers testable independently

**Implementation**:

CLI command:
```bash
ng_test +entrypoint=resources_servers/simple_weather
```

Test structure:
```python
class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        SimpleWeatherResourcesServer(
            config=config, 
            server_client=MagicMock(spec=ServerClient)
        )
```

**Benefit**: Validate and iterate on environments in seconds, integrate when ready

---

### 6. Async Parallelism

**Challenge**: Sequential processing doesn't utilize available compute for data generation

**Solution**: Configurable async parallelism with semaphore-based concurrency control

**Implementation**:

```python
# Configure concurrency limit
semaphore = nullcontext()
if config.num_samples_in_parallel:
    semaphore = Semaphore(config.num_samples_in_parallel)

# Async coroutine per task
async def _post_coroutine(row: dict) -> None:
    row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
    async with semaphore:
        response = await server_client.post(
            server_name=config.agent_name, 
            url_path="/run", 
            json=row
        )
        await raise_for_status(response)
        result = await response.json()
        f.write(json.dumps(result) + "\n")
        metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})

# Execute all tasks in parallel with progress tracking
await tqdm.gather(*map(_post_coroutine, rows), desc="Collecting rollouts", miniters=tqdm_miniters)
```

**Configuration**: Set `num_samples_in_parallel=10` to control concurrent rollouts

**Benefit**: Generate thousands of rollouts concurrently while respecting rate limits

---

## Component Communication

NeMo Gym's HTTP-based architecture enables flexible deployment:

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - Component
  - Type
  - Key Endpoints
  - Communication
* - **Model Server**
  - Stateless
  - `POST /v1/responses`
  - HTTP REST (OpenAI format)
* - **Resource Server**
  - Stateful (per session)
  - `POST /verify`, `POST /{tool_name}`
  - HTTP REST (OpenAI functions)
* - **Agent Server**
  - Stateless orchestrator
  - `POST /run`, `POST /v1/responses`
  - HTTP REST
* - **Training Framework**
  - External
  - N/A (calls NeMo Gym)
  - HTTP REST to agent `/run`
```

All components communicate via HTTP, enabling:
- Local development (everything on laptop)
- Distributed deployment (components on different machines)
- Containerization (each component in separate container)
- Independent scaling (scale resource servers separately from models)

---

## Implementation Patterns

### Pattern 1: Model-Only Training

**Use when**: Training directly against curated resource servers

```
Training Framework
    ↓ (sends tasks)
NeMo Gym Agent
    ↓ (coordinates)
Model Server ↔ Resource Server
    ↓ (returns trajectory + reward)
Training Framework
```

**Best for**: RL research, standard environments, curated datasets

---

### Pattern 2: Agent Workflow Integration

**Use when**: Training models using existing agent systems (NAT, LangChain, LangGraph)

```
Training Framework
    ↓ (sends tasks)
NeMo Gym
    ↓ (calls)
External Agent Workflow (HTTP endpoint)
    ↓ (returns trajectory + score)
NeMo Gym
    ↓ (returns trajectory + reward)
Training Framework
```

**Best for**: Leveraging existing agent infrastructure, production workflows

---

## Related Topics

- {doc}`../architecture` - General system architecture
- {doc}`core-abstractions` - Models, Resources, and Agents explained
- {doc}`rollout-collection-fundamentals` - How rollout collection works
- {doc}`../../training/index` - Training pipeline implementation guides

