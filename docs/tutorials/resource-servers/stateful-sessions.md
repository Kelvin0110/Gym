(tutorial-stateful-sessions)=

# Stateful Sessions

In {doc}`Multi-Step Interactions <multi-step-interactions>`, you orchestrated sequential tool calls. Now you'll manage session state that persists and changes across interactions.

:::{card}

**Goal**: Build a stateful counter server where tools modify shared state.

^^^

**In this tutorial, you will**:

1. Implement session state management
2. Define tools that read and modify state
3. Verify based on final state
4. Handle state isolation between rollouts

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

- ✅ Completed {doc}`multi-step-interactions`
- ✅ Understanding of session management concepts
- ✅ NeMo Gym installed and working

**What you'll build**: A counter server where agents increment/decrement a value to reach a target, with verification based on final state.

:::{tip}
**Reference implementation**: `resources_servers/example_stateful_counter/`
:::

---

## 1. Understand the Pattern

<!-- SME: Explain stateful vs stateless -->

Stateful sessions are needed when:

- Tool calls modify shared state
- The agent must track cumulative changes
- Verification depends on the final state

```{mermaid}
sequenceDiagram
    participant Agent
    participant ResourceServer
    Note over ResourceServer: Initialize state
    Agent->>ResourceServer: increment()
    Note over ResourceServer: state = 1
    ResourceServer-->>Agent: Current: 1
    Agent->>ResourceServer: increment()
    Note over ResourceServer: state = 2
    ResourceServer-->>Agent: Current: 2
    Agent->>ResourceServer: Submit answer
    ResourceServer-->>Agent: Verify state == target
```

**✅ Success Check**: You understand when stateful sessions are necessary.

---

## 2. Implement Session State

<!-- SME: Show state management from example_stateful_counter/app.py -->

```python
# TODO: State management from example_stateful_counter/app.py
```

```{list-table} State Management
:header-rows: 1
:widths: 30 70

* - Concept
  - Implementation
* - Session initialization
  - <!-- SME: How state is created -->
* - State isolation
  - <!-- SME: How rollouts are isolated -->
* - State cleanup
  - <!-- SME: When/how state is cleared -->
```

**✅ Success Check**: <!-- SME: How to verify state management -->

---

## 3. Define Stateful Tools

<!-- SME: Show tool definitions that modify state -->

```python
# TODO: Tool definitions from example_stateful_counter/app.py
```

**✅ Success Check**: Tools correctly read and modify session state.

---

## 4. Verify Final State

<!-- SME: Show verification logic -->

```python
# TODO: Verification logic from example_stateful_counter/app.py
```

**✅ Success Check**: Verification correctly compares final state to target.

---

## 5. Create Test Data

<!-- SME: Show example.jsonl with target states -->

```json
// TODO: Example from example_stateful_counter/data/example.jsonl
```

**✅ Success Check**: Each example has a clear target state to reach.

---

## 6. Run and Test

<!-- SME: Commands to run and test -->

```bash
# TODO: Commands to run and test
```

**✅ Success Check**: Agent reaches target states and receives correct rewards.

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} State leaking between rollouts
<!-- SME: Solution -->
:::

:::{dropdown} State not initializing correctly
<!-- SME: Solution -->
:::

:::{dropdown} Concurrent rollouts corrupting state
<!-- SME: Solution -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: llm-as-judge
:link-type: doc

When verification needs semantic understanding.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Execution
:link: code-execution
:link-type: doc

Execute and verify generated code.
:::

::::
