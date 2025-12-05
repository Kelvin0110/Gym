(tutorial-code-execution)=

# Code Execution

In previous tutorials, you verified text outputs. Now you'll execute generated code in a sandbox and verify correctness against test cases.

:::{card}

**Goal**: Build a code execution server that runs and verifies generated code safely.

^^^

**In this tutorial, you will**:

1. Set up a sandboxed execution environment
2. Define a code execution tool
3. Implement test-based verification
4. Handle security and resource limits

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
- ✅ Understanding of code sandboxing concepts
- ✅ Docker or similar isolation mechanism (recommended)

**What you'll build**: A code generation server where agents write code, you execute it safely, and verify against test cases.

:::{tip}
**Reference implementations**:
- `resources_servers/code_gen/` — Code generation with LiveCodeBench
- `resources_servers/math_with_code/` — Math via code execution
:::

---

## 1. Understand the Pattern

<!-- SME: Explain code execution verification -->

Code execution verification is used when:

- The task is to generate working code
- Correctness is defined by test cases
- Multiple valid implementations exist

```{mermaid}
flowchart LR
    A["Generated Code"] --> B["Sandbox"]
    B --> C["Execute"]
    C --> D{"Tests Pass?"}
    D -->|"All pass"| E["✅ Score: 1.0"]
    D -->|"Some pass"| F["⚠️ Partial"]
    D -->|"None pass"| G["❌ Score: 0.0"]
```

**✅ Success Check**: You understand when code execution verification is appropriate.

---

## 2. Set Up the Sandbox

<!-- SME: Show sandbox configuration from code_gen -->

```python
# TODO: Sandbox setup from code_gen/app.py
```

```{list-table} Sandbox Configuration
:header-rows: 1
:widths: 30 70

* - Setting
  - Purpose
* - Resource limits
  - Prevent runaway processes
* - Network isolation
  - Prevent data exfiltration
* - Filesystem restrictions
  - Limit file access
```

**✅ Success Check**: Sandbox prevents malicious code from affecting the host system.

---

## 3. Define the Code Execution Tool

<!-- SME: Show tool that executes code -->

```python
# TODO: Tool definition from code_gen/app.py
```

**✅ Success Check**: Tool accepts code and returns execution results.

---

## 4. Implement Test-Based Verification

<!-- SME: Show verification against test cases -->

```python
# TODO: Verification logic from code_gen/app.py
```

```{list-table} Verification Metrics
:header-rows: 1
:widths: 30 70

* - Metric
  - Calculation
* - Pass rate
  - `passed_tests / total_tests`
* - <!-- SME: Add other metrics -->
  - 
```

**✅ Success Check**: Verification correctly scores based on test results.

---

## 5. Create Test Data

<!-- SME: Show example.jsonl with test cases -->

```json
// TODO: Example from code_gen/data/example.jsonl
```

**✅ Success Check**: Each example includes problem description and test cases.

---

## 6. Run and Test

<!-- SME: Commands to run and test -->

```bash
# TODO: Commands to run and test
```

**✅ Success Check**: Agent generates code that passes test cases.

---

## Security Considerations

:::{warning}
Code execution introduces security risks. Always:

- Use sandboxed environments (Docker, VMs, or similar)
- Set resource limits (CPU, memory, time)
- Restrict filesystem and network access
- Never execute code with elevated privileges
:::

```{list-table} Security Checklist
:header-rows: 1
:widths: 30 70

* - Risk
  - Mitigation
* - Infinite loops
  - Execution timeout
* - Memory exhaustion
  - Memory limits
* - File system access
  - Restrict to temp directory
* - Network access
  - Disable or proxy
```

**✅ Success Check**: Security measures are in place and tested.

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Sandbox setup failing
<!-- SME: Solution -->
:::

:::{dropdown} Code timing out on valid solutions
<!-- SME: Solution -->
:::

:::{dropdown} Test cases not matching expected format
<!-- SME: Solution -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: llm-as-judge
:link-type: doc

Combine with LLM verification for partial credit.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Multi-Verifier Patterns
:link: /training/verification/index
:link-type: doc

Combine code execution with other verification methods.
:::

::::
