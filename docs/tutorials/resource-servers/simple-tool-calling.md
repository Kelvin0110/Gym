(tutorial-simple-tool-calling)=

# Simple Tool Calling

In {doc}`Creating a Resource Server </tutorials/creating-resource-server>`, you learned the basics of resource server structure. Now you'll implement a complete single-step tool calling pattern with deterministic verification.

:::{card}

**Goal**: Build a weather tool server that agents can query and verify.

^^^

**In this tutorial, you will**:

1. Define a tool schema for weather queries
2. Implement the tool handler
3. Write deterministic verification logic
4. Test end-to-end with an agent

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`/tutorials/creating-resource-server`
- ✅ NeMo Gym installed and working
- ✅ Basic Python and FastAPI knowledge

**What you'll build**: A weather information server where agents query weather data and you verify they correctly extract and report the information.

:::{tip}
**Reference implementation**: `resources_servers/example_simple_weather/`
:::

---

## 1. Understand the Pattern

<!-- SME: Explain when to use simple tool calling vs multi-step -->

Simple tool calling is appropriate when:

- The task requires a single tool invocation
- Verification is deterministic (correct/incorrect)
- No state needs to persist between calls

```{mermaid}
sequenceDiagram
    participant Agent
    participant ResourceServer
    Agent->>ResourceServer: Call tool (get_weather)
    ResourceServer-->>Agent: Return result
    Agent->>ResourceServer: Submit final answer
    ResourceServer-->>Agent: Verification score
```

**✅ Success Check**: You understand when to use this pattern vs. multi-step or stateful patterns.

---

## 2. Define the Tool Schema

<!-- SME: Show the tool definition from example_simple_weather/app.py -->

```python
# TODO: Tool definition from example_simple_weather/app.py
```

```{list-table} Tool Schema Fields
:header-rows: 1
:widths: 25 15 60

* - Field
  - Type
  - Description
* - `name`
  - `str`
  - Tool identifier used by agents
* - `description`
  - `str`
  - What the tool does (shown to agent)
* - `parameters`
  - `dict`
  - JSON Schema for input parameters
```

**✅ Success Check**: <!-- SME: What should they verify? -->

---

## 3. Implement the Tool Handler

<!-- SME: Show the handler implementation -->

```python
# TODO: Handler implementation from example_simple_weather/app.py
```

**✅ Success Check**: <!-- SME: How to verify handler works -->

---

## 4. Write Verification Logic

<!-- SME: Show the verify function -->

```python
# TODO: Verification logic from example_simple_weather/app.py
```

```{list-table} Verification Parameters
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - <!-- SME: Add parameters -->
  - 
  - 
```

**✅ Success Check**: <!-- SME: Verification criteria -->

---

## 5. Create Test Data

<!-- SME: Show example.jsonl format -->

```json
// TODO: Example from example_simple_weather/data/example.jsonl
```

**✅ Success Check**: You have at least 3–5 test examples in `data/example.jsonl`.

---

## 6. Run and Test

<!-- SME: Show commands to run and test -->

```bash
# TODO: Commands to run and test the server
```

**✅ Success Check**: <!-- SME: Expected output -->

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Tool not appearing in agent's available tools
<!-- SME: Solution -->
:::

:::{dropdown} Verification always returns 0
<!-- SME: Solution -->
:::

:::{dropdown} Agent not calling the tool
<!-- SME: Solution -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Interactions
:link: multi-step-interactions
:link-type: doc

Orchestrate multiple tool calls for complex tasks.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Sessions
:link: stateful-sessions
:link-type: doc

Manage state across multiple interactions.
:::

::::
