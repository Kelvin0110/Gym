(tutorial-llm-as-judge)=

# LLM-as-Judge Verification

In previous tutorials, you used deterministic verification. Now you'll use language models to verify open-ended outputs where exact matching isn't possible.

:::{card}

**Goal**: Build a verification system using LLMs to evaluate answer correctness.

^^^

**In this tutorial, you will**:

1. Configure a judge model
2. Implement hybrid verification (rule-based + LLM)
3. Handle edge cases and ambiguity
4. Optimize for cost and latency

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
- ✅ Access to a judge model (OpenAI API or local model)
- ✅ Understanding of semantic equivalence challenges

**What you'll build**: A math verification system that combines rule-based checking with LLM evaluation for ambiguous cases.

:::{tip}
**Reference implementations**:
- `resources_servers/math_with_judge/` — Math verification
- `resources_servers/equivalence_llm_judge/` — General equivalence
:::

---

## 1. Understand the Pattern

<!-- SME: Explain when to use LLM-as-judge -->

LLM-as-judge is appropriate when:

- Answers have multiple valid phrasings
- Semantic equivalence matters more than exact match
- Rule-based verification is too brittle

```{mermaid}
flowchart LR
    A["Agent Response"] --> B{"Rule-based<br/>Check"}
    B -->|"Clear match"| C["✅ Correct"]
    B -->|"Clear mismatch"| D["❌ Incorrect"]
    B -->|"Uncertain"| E["LLM Judge"]
    E --> F["Score 0.0-1.0"]
```

**✅ Success Check**: You understand when LLM verification is necessary vs. deterministic.

---

## 2. Configure the Judge Model

<!-- SME: Show judge model configuration from math_with_judge -->

```yaml
# TODO: Configuration from math_with_judge/configs/
```

```{list-table} Judge Configuration
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - `judge_model_server`
  - <!-- SME: Describe -->
* - <!-- SME: Add other params -->
  - 
```

**✅ Success Check**: Judge model responds to test queries.

---

## 3. Implement Hybrid Verification

<!-- SME: Show verification that combines rule-based and LLM -->

```python
# TODO: Verification logic from math_with_judge/app.py
```

**✅ Success Check**: Obvious cases resolve without LLM calls; ambiguous cases go to judge.

---

## 4. Handle Edge Cases

<!-- SME: Show how to handle ambiguous cases -->

```python
# TODO: Edge case handling
```

```{list-table} Edge Case Handling
:header-rows: 1
:widths: 30 70

* - Case
  - Handling
* - Equivalent but different format
  - <!-- SME: How handled -->
* - Partially correct
  - <!-- SME: How handled -->
* - Judge uncertain
  - <!-- SME: How handled -->
```

**✅ Success Check**: Edge cases are handled gracefully with appropriate scores.

---

## 5. Run and Test

<!-- SME: Commands to test verification -->

```bash
# TODO: Commands to test verification
```

**✅ Success Check**: Verification correctly scores a mix of correct, incorrect, and ambiguous answers.

---

## Cost and Latency Optimization

<!-- SME: Add guidance on optimizing judge calls -->

```{list-table} Optimization Strategies
:header-rows: 1
:widths: 30 70

* - Strategy
  - Impact
* - Batch judge calls
  - Reduces API overhead
* - Cache results
  - Avoids duplicate evaluations
* - Use smaller models for obvious cases
  - Reduces cost
```

**✅ Success Check**: Judge calls are minimized without sacrificing accuracy.

---

## Troubleshooting

<!-- SME: Add common issues and solutions -->

:::{dropdown} Judge model not accessible
<!-- SME: Solution -->
:::

:::{dropdown} Inconsistent judge responses
<!-- SME: Solution -->
:::

:::{dropdown} High latency from judge calls
<!-- SME: Solution -->
:::

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Execution
:link: code-execution
:link-type: doc

Verify code by executing it.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Multi-Verifier Patterns
:link: /training/verification/index
:link-type: doc

Combine multiple verification methods.
:::

::::
