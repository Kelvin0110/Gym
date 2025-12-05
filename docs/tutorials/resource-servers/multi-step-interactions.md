(tutorial-multi-step-interactions)=

# Multi-Step Interactions

In {doc}`Simple Tool Calling <simple-tool-calling>`, you built a server for single-step tool use. Now you'll orchestrate multiple sequential tool calls to complete complex extraction tasks.

:::{card}

**Goal**: Build a multi-step extraction server where agents query multiple data sources.

^^^

**In this tutorial, you will**:

1. Define multiple related tools
2. Design data that requires sequential queries
3. Implement aggregation verification
4. Test multi-step rollouts

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

- ✅ Completed {doc}`simple-tool-calling`
- ✅ Understanding of tool sequencing concepts
- ✅ NeMo Gym installed and working

**What you'll build**: An extraction server where agents must query multiple data sources sequentially and aggregate results into a final answer.

:::{tip}
**Reference implementation**: `resources_servers/example_multi_step/`
:::

---

## 1. Understand the Pattern

<!-- SME: Explain multi-step vs single-step -->

Multi-step interactions are needed when:

- Tasks require information from multiple sources
- Results from one tool inform the next query
- The agent must synthesize multiple outputs

```{mermaid}
sequenceDiagram
    participant Agent
    participant ResourceServer
    Agent->>ResourceServer: Call tool 1
    ResourceServer-->>Agent: Result 1
    Agent->>ResourceServer: Call tool 2
    ResourceServer-->>Agent: Result 2
    Agent->>ResourceServer: Call tool 3
    ResourceServer-->>Agent: Result 3
    Agent->>ResourceServer: Submit aggregated answer
    ResourceServer-->>Agent: Verification score
```

**✅ Success Check**: You understand when multi-step is necessary vs. single-step.

---

## 2. Define Multiple Tools

<!-- SME: Show tool definitions from example_multi_step/app.py -->

```python
# TODO: Tool definitions from example_multi_step/app.py
```

```{list-table} Tool Definitions
:header-rows: 1
:widths: 25 75

* - Tool
  - Purpose
* - <!-- SME: Add tools -->
  - 
```

**✅ Success Check**: <!-- SME: Verification -->

---

## 3. Design Sequential Data

<!-- SME: Show how to structure data that requires multiple steps -->

```json
// TODO: Example from example_multi_step/data/example.jsonl
```

**✅ Success Check**: Each example requires 2+ tool calls to solve correctly.

---

## 4. Implement Aggregation Verification

<!-- SME: Show verification that checks all steps -->

```python
# TODO: Verification logic from example_multi_step/app.py
```

```{list-table} Verification Logic
:header-rows: 1
:widths: 30 70

* - Check
  - Description
* - <!-- SME: Add checks -->
  - 
```

**✅ Success Check**: <!-- SME: Verification criteria -->

---

## 5. Run and Test

<!-- SME: Commands to run and test -->

```bash
# TODO: Commands to run and test
```

**✅ Success Check**: Agent successfully completes multi-step examples with correct aggregation.

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Agent stops after first tool call
<!-- SME: Solution -->
:::

:::{dropdown} Agent calls tools in wrong order
<!-- SME: Solution -->
:::

:::{dropdown} Verification doesn't account for all steps
<!-- SME: Solution -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Sessions
:link: stateful-sessions
:link-type: doc

Add state management when tools modify shared data.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: llm-as-judge
:link-type: doc

When deterministic verification isn't possible.
:::

::::
