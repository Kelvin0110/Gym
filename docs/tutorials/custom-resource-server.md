(tutorials-custom-resource-server)=

# Building Custom Resource Servers

Create custom resource servers for your domain-specific tasks, tools, and verification logic.

:::{card}

**Goal**: Build a custom resource server from scratch with tools, datasets, and verification.

^^^

**In this tutorial, you will**:

1. Set up the resource server directory structure
2. Implement custom tools and verification logic
3. Configure datasets and server settings
4. Test your resource server

:::

:::{tip}
**Already familiar with the built-in servers?** This tutorial builds on concepts from {doc}`../get-started/index`. Make sure you understand how resource servers work before building your own.
:::

---

## When to Build a Custom Resource Server

Build a custom resource server when you need:

- **Domain-specific tools**: APIs, databases, or services not covered by built-in servers
- **Custom verification**: Specialized scoring logic for your task domain
- **Proprietary datasets**: Internal data that requires custom processing
- **Novel task types**: Tasks that don't fit existing patterns (MCQA, math, coding, etc.)

:::{seealso}
**Choosing an existing server?** See {doc}`../training/resource-servers/index` for 13 built-in servers covering common tasks.
:::

---

## Setup: Manual Method

Create custom resource servers by copying the template structure:

```bash
# 1. Create directory structure
mkdir -p resources_servers/my_task/{configs,data,tests}

# 2. Copy and adapt from existing server
cp resources_servers/simple_weather/app.py resources_servers/my_task/
cp resources_servers/simple_weather/configs/simple_weather.yaml \
   resources_servers/my_task/configs/my_task.yaml

# 3. Update configuration with your server name
# Edit configs/my_task.yaml and replace names
```

### Required Directory Structure

```text
resources_servers/my_task/
├── app.py                          # Server implementation
├── configs/
│   └── my_task.yaml                # Server configuration
├── data/
│   ├── train.jsonl                 # Training dataset
│   ├── validation.jsonl            # Validation dataset
│   └── example.jsonl               # Example prompts
├── tests/
│   └── test_app.py                 # Server tests
└── requirements.txt                # Optional dependencies
```

---

## Step 1: Implement Server Logic

### Basic Server Structure

Start with the minimal server implementation in `app.py`:

```python
from pydantic import BaseModel
from nemo_gym.base_resources_server import (
    BaseResourcesServer,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

# Define your tool's input/output schemas
class MyToolRequest(BaseModel):
    input_text: str

class MyToolResponse(BaseModel):
    result: str

# Define your resource server
class MyTaskResourcesServer(BaseResourcesServer):
    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        """Your custom tool implementation."""
        # TODO: Implement your tool logic
        result = f"Processed: {body.input_text}"
        return MyToolResponse(result=result)
    
    async def verify(
        self, body: BaseVerifyRequest
    ) -> BaseVerifyResponse:
        """Verification logic that scores agent performance."""
        # TODO: Implement verification logic
        reward = 1.0  # Replace with actual scoring
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

# Required: Create server instance
server = MyTaskResourcesServer()
```

### Tool Implementation Patterns

::::{tab-set}

:::{tab-item} External API Call

```python
import httpx

async def fetch_data(self, body: FetchRequest) -> FetchResponse:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/data/{body.query}"
        )
        data = response.json()
    return FetchResponse(data=data)
```

:::

:::{tab-item} Database Query

```python
import asyncpg

async def query_database(self, body: QueryRequest) -> QueryResponse:
    conn = await asyncpg.connect('postgresql://...')
    result = await conn.fetch(
        'SELECT * FROM table WHERE id = $1',
        body.record_id
    )
    await conn.close()
    return QueryResponse(records=result)
```

:::

:::{tab-item} Code Execution

```python
import subprocess

async def execute_code(self, body: CodeRequest) -> CodeResponse:
    result = subprocess.run(
        ['python', '-c', body.code],
        capture_output=True,
        text=True,
        timeout=5
    )
    return CodeResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode
    )
```

:::

::::

### Verification Implementation

Your `verify()` function scores agent performance. Common patterns:

::::{tab-set}

:::{tab-item} Exact Match (Binary)

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    """Binary scoring: 1.0 if correct, 0.0 if incorrect."""
    
    # Extract agent's answer from the conversation
    agent_answer = self._extract_answer(body.input)
    
    # Get ground truth from metadata
    correct_answer = body.metadata.get("answer")
    
    # Binary scoring
    reward = 1.0 if agent_answer == correct_answer else 0.0
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: MCQA, instruction following, structured outputs

:::

:::{tab-item} Semantic Similarity (Continuous)

```python
from openai import AsyncOpenAI

async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    """Continuous scoring using LLM judge."""
    
    agent_answer = self._extract_answer(body.input)
    reference_answer = body.metadata.get("reference_answer")
    
    # Use LLM to judge semantic similarity
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Rate the semantic similarity of two answers from 0.0 to 1.0"
            },
            {
                "role": "user",
                "content": f"Reference: {reference_answer}\nAgent: {agent_answer}"
            }
        ]
    )
    
    # Parse score from LLM response
    reward = float(response.choices[0].message.content)
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Open-ended QA, creative tasks, multi-valid-answer scenarios

:::

:::{tab-item} Programmatic Check (Binary/Continuous)

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    """Verification using programmatic checks."""
    
    agent_output = self._extract_answer(body.input)
    
    # Run programmatic checks
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Valid JSON format
    try:
        import json
        parsed = json.loads(agent_output)
        checks_passed += 1
    except:
        parsed = {}
    
    # Check 2: Required fields present
    if all(k in parsed for k in ["name", "value"]):
        checks_passed += 1
    
    # Check 3: Value in valid range
    if 0 <= parsed.get("value", -1) <= 100:
        checks_passed += 1
    
    reward = checks_passed / total_checks
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Best for**: Structured outputs, constraint satisfaction, multi-criteria evaluation

:::

::::

---

## Step 2: Create Configuration

Your `configs/my_task.yaml` must follow this structure:

```yaml
# Resource server definition
my_task_resources_server:
  resources_servers:
    my_task:
      entrypoint: app.py
      host: 127.0.0.1
      # port: auto-assigned if omitted

# Agent definition
my_task_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_task_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
        - name: train
          type: train
          jsonl_fpath: resources_servers/my_task/data/train.jsonl
        - name: validation
          type: validation
          jsonl_fpath: resources_servers/my_task/data/validation.jsonl
```

**Key configuration elements:**

- **`entrypoint`**: Path to your `app.py` (relative to server directory)
- **`resources_server` reference**: Links agent to your resource server
- **`model_server` reference**: Links to policy model (defined in separate config)
- **`datasets`**: JSONL files containing your task data

---

## Step 3: Prepare Datasets

### Dataset Format

Create JSONL files with one task per line:

```jsonl
{"query": "What is the capital of France?", "answer": "Paris", "metadata": {"difficulty": "easy"}}
{"query": "Calculate 15 * 23", "answer": "345", "metadata": {"difficulty": "medium"}}
{"query": "Write a function to reverse a string", "answer": "def reverse(s): return s[::-1]", "metadata": {"difficulty": "hard"}}
```

**Required fields:**

- `query`: The task or question for the agent
- `answer`: Ground truth for verification
- `metadata`: Additional context (optional but recommended)

### Example Dataset

Create `data/example.jsonl` with 5+ examples for testing:

```bash
cat > resources_servers/my_task/data/example.jsonl << EOF
{"query": "Test query 1", "answer": "Expected answer 1", "metadata": {}}
{"query": "Test query 2", "answer": "Expected answer 2", "metadata": {}}
{"query": "Test query 3", "answer": "Expected answer 3", "metadata": {}}
{"query": "Test query 4", "answer": "Expected answer 4", "metadata": {}}
{"query": "Test query 5", "answer": "Expected answer 5", "metadata": {}}
EOF
```

:::{seealso}
**Dataset specifications:** See {doc}`../training/datasets/format-specification` for complete JSONL format requirements.
:::

---

## Step 4: Write Tests

Create `tests/test_app.py` to validate your server:

```python
import pytest
from app import MyTaskResourcesServer, MyToolRequest

@pytest.fixture
def server():
    return MyTaskResourcesServer()

@pytest.mark.asyncio
async def test_my_tool(server):
    """Test tool executes correctly."""
    request = MyToolRequest(input_text="test")
    response = await server.my_tool(request)
    
    assert response.result is not None
    assert "test" in response.result

@pytest.mark.asyncio
async def test_verify(server):
    """Test verification logic."""
    from nemo_gym.base_resources_server import BaseVerifyRequest
    
    verify_request = BaseVerifyRequest(
        input=[
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "correct answer"}
        ],
        metadata={"answer": "correct answer"}
    )
    
    response = await server.verify(verify_request)
    
    assert 0.0 <= response.reward <= 1.0
    assert response.reward == 1.0  # Correct answer should get full score
```

**Minimum test requirements:**

- At least one test (you own correctness)
- Tests should cover tool execution and verification
- Use `pytest` for test discovery

---

## Step 5: Test Your Server

### Test Server Functionality

```bash
# Test your resource server
ng_test +entrypoint=resources_servers/my_task

# Test with data validation
ng_test +entrypoint=resources_servers/my_task +should_validate_data=true
```

**What `ng_test` validates:**

- ✅ Server starts without errors
- ✅ Endpoints respond correctly
- ✅ Verification logic produces valid rewards
- ✅ Example data is well-formed (if `should_validate_data=true`)

### Test End-to-End

```bash
# Start your servers
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/my_task/configs/my_task.yaml"

ng_run "+config_paths=[${config_paths}]"

# In another terminal, collect rollouts
ng_collect_rollouts \
    +agent_name=my_task_simple_agent \
    +input_jsonl_fpath=resources_servers/my_task/data/example.jsonl \
    +output_jsonl_fpath=outputs/my_task_test.jsonl \
    +limit=5
```

**Validate results:**

```bash
# Inspect collected data
ng_viewer +jsonl_fpath=outputs/my_task_test.jsonl
```

Check for:

- ✅ Tools are called correctly
- ✅ Verification returns reasonable rewards
- ✅ Agent responses make sense for your task

---

## Advanced Patterns

### Multi-Tool Servers

```python
class MyTaskResourcesServer(BaseResourcesServer):
    async def tool_1(self, body: Tool1Request) -> Tool1Response:
        """First tool."""
        pass
    
    async def tool_2(self, body: Tool2Request) -> Tool2Response:
        """Second tool."""
        pass
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Verify based on which tools were used."""
        pass
```

### Multi-Metric Verification

```python
from typing import Dict

async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    """Return primary reward + additional metrics."""
    
    accuracy = self._compute_accuracy(body)
    completeness = self._compute_completeness(body)
    efficiency = self._compute_efficiency(body)
    
    # Primary reward (weighted combination)
    reward = 0.5 * accuracy + 0.3 * completeness + 0.2 * efficiency
    
    # Additional metrics for analysis
    metadata = {
        "accuracy": accuracy,
        "completeness": completeness,
        "efficiency": efficiency
    }
    
    return BaseVerifyResponse(
        **body.model_dump(),
        reward=reward,
        metadata=metadata
    )
```

:::{seealso}
**Multi-objective scoring patterns:** See {doc}`../training/verification/multi-objective-scoring` for advanced verification strategies.
:::

### External Dependencies

If your server requires additional packages, create `requirements.txt`:

```text
httpx>=0.24.0
asyncpg>=0.28.0
numpy>=1.24.0
```

`ng_test` will automatically install these dependencies in an isolated environment.

---

## Deployment Checklist

Before using your custom server for training:

- [ ] **Tests pass**: `ng_test +entrypoint=resources_servers/my_task`
- [ ] **Data validated**: At least 5 examples in `data/example.jsonl`
- [ ] **End-to-end works**: Successfully collected rollouts
- [ ] **Verification sensible**: Rewards in 0.0–1.0 range, distribution makes sense
- [ ] **Documentation**: Comments explain tool logic and verification strategy

---

## Common Patterns and Tips

### Tool Design

- **Keep tools focused**: One clear purpose per tool
- **Use async**: Always use `async def` for I/O operations
- **Handle errors**: Gracefully handle API failures, timeouts, invalid inputs
- **Add typing**: Use Pydantic models for request/response validation

### Verification Design

- **Match training goal**: Binary for SFT, continuous for DPO/PPO
- **Test edge cases**: Verify handles empty responses, errors, edge cases
- **Calibrate scores**: Test that reward distribution matches expectations
- **Document assumptions**: Comment on what constitutes a "correct" answer

### Performance

- **Async everything**: Use async for all I/O (API calls, database queries)
- **Connection pooling**: Reuse connections for external services
- **Caching**: Cache expensive computations when appropriate
- **Profiling**: Use `ng_run +profiling_enabled=true` to identify bottlenecks

:::{seealso}
**Performance optimization:** See {doc}`../setup-deployment/operations/index` for profiling and optimization techniques.
:::

---

## Next Steps

**After building your custom server:**

:::{button-ref} ../training/rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start Collecting Rollouts →
:::

**For production deployments:**

:::{button-ref} ../setup-deployment/deployment/index
:color: secondary
:outline:
:ref-type: doc

Deploy Your Server →
:::

---

## Troubleshooting

### Server Won't Start

**Issue**: Import errors or module not found

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/Gym

# Verify editable install
pip install -e ".[dev]"

# Check server path is correct
ls resources_servers/my_task/app.py
```

### Verification Returns Invalid Rewards

**Issue**: Rewards outside 0.0–1.0 range or NaN

**Solution**:
```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    reward = self._compute_reward(body)
    
    # Clamp to valid range
    reward = max(0.0, min(1.0, reward))
    
    # Handle NaN
    if reward != reward:  # NaN check
        reward = 0.0
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

### Tools Not Accessible to Agent

**Issue**: Agent doesn't use your tools

**Solution**: Ensure tools are registered in your server and visible to the agent via the responses API. Check that tool descriptions are clear and match the task.

---

## Resources

- {doc}`../training/resource-servers/index` - Built-in resource servers
- {doc}`../training/verification/index` - Verification patterns
- {doc}`../training/datasets/index` - Dataset preparation
- {doc}`../setup-deployment/deployment/local-development` - Development workflows

