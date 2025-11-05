(training-verification-patterns)=

# Verification Patterns

Reference catalog of verification approaches with built-in examples, tradeoffs, and selection guidance.

:::{card}

**Goal**: Choose the right verification pattern for your task.

^^^

**You'll find**:

1. Catalog of verification patterns with examples
2. Tradeoffs and when to use each approach
3. Built-in resource servers demonstrating patterns
4. Quick decision guide

:::

**Use this reference when**: Designing verification logic or comparing verification approaches for your domain.

---

## How to Use This Reference

### Structure

Each pattern includes:

* **Overview** - What the pattern verifies
* **When to use** - Task characteristics that fit this pattern
* **Tradeoffs** - Accuracy, speed, complexity
* **Built-in examples** - NeMo Gym resource servers using this pattern
* **Configuration** - Key parameters and variations

### Finding Your Pattern

1. **Browse by category** - Correctness, execution, quality, hybrid
2. **Check decision guide** - Quick lookup table at bottom
3. **Compare tradeoffs** - Understand cost vs accuracy

---

## Correctness Patterns

Verify agent output matches expected answer or demonstrates correct understanding.

### Exact Match

**Overview**: Compare extracted answer directly to expected value.

**When to use**:

* Answers have canonical form (letters, numbers, exact strings)
* No ambiguity in correct answer
* High-stakes correctness requirements

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Deterministic, no false positives
* - **Speed**
  - Very fast (simple string comparison)
* - **Flexibility**
  - Brittle to format variations
* - **Cost**
  - Minimal computational cost
```

**Built-in example**: `resources_servers/mcqa/`

```python
# From mcqa/app.py:259-261
gold = (expected_answer or "").strip().upper()
is_correct = (pred == gold) if (pred is not None and gold) else False
reward = 1.0 if is_correct else 0.0
```

**Source**: `resources_servers/mcqa/app.py:259-261`

**Variations**:

* **Strict boxed matching** - Look for `\boxed{answer}` format
* **Regex extraction** - Use per-record regex patterns to extract answers
* **Lenient mode** - Try multiple extraction strategies (boxed, then option text, then answer colon)

**Configuration options** (MCQA example):

* `grading_mode`: `strict_single_letter_boxed`, `lenient_boxed`, `lenient_answer_colon`
* `template_metadata.output_regex`: Per-record custom extraction patterns

---

### Semantic Equivalence (LLM Judge)

**Overview**: Use LLM to judge if generated answer is equivalent to expected answer, accounting for paraphrasing and format differences.

**When to use**:

* Multiple valid phrasings of correct answer
* Open-ended questions
* Semantic correctness matters more than exact wording

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Handles paraphrasing, but may have judge biases
* - **Speed**
  - Slower (requires LLM call per verification)
* - **Flexibility**
  - Very flexible, handles many answer formats
* - **Cost**
  - Higher inference cost per rollout
```

**Built-in example**: `resources_servers/equivalence_llm_judge/`

**Approach**:

1. Format prompt comparing expected vs generated answers
2. Call judge model with structured prompt
3. Parse judge verdict (`[[A=B]]` or `[[A!=B]]`)
4. Optional: Swap check to reduce position bias

```python
# Judge prompt structure
<|Problem|>
{question}

<|Start of Assistant A's Answer|>
{expected_answer}
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
{generated_answer}
<|End of Assistant B's Answer|>
```

**Source**: Pattern from `resources_servers/equivalence_llm_judge/app.py:379`

**Configuration options**:

* `judge_system_message`: Custom system prompt for judge
* `judge_prompt_template`: Template with placeholders for question/answers
* `judge_equal_label` / `judge_not_equal_label`: Expected output labels
* `check_twice_swap`: Enable/disable swap check for bias reduction
* `use_per_record_regex`: Allow per-record answer extraction patterns

**Bias mitigation**: Swap check repeats judgment with swapped positions—if verdict changes, flag as uncertain.

---

### Mathematical Equivalence

**Overview**: Combine symbolic math library for equivalence checking with LLM judge fallback for edge cases.

**When to use**:

* Math problems with multiple valid forms (fractions, decimals, algebraic expressions)
* Need deterministic verification when possible
* Okay with LLM fallback for complex expressions

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - High for symbolic math, LLM handles edge cases
* - **Speed**
  - Fast for library check, slower for LLM fallback
* - **Flexibility**
  - Handles multiple mathematical notations
* - **Cost**
  - Low when library succeeds, higher for LLM fallback
```

**Built-in example**: `resources_servers/library_judge_math/`

**Approach**:

1. Extract answer from response using LaTeX/expression parsers
2. Attempt symbolic equivalence check (math-verify library)
3. If library check inconclusive, call LLM judge
4. Combine signals into final reward

**Source**: `resources_servers/library_judge_math/app.py:118`

**Library used**: `math-verify` package with `ExprExtractionConfig` and `LatexExtractionConfig`.

**Configuration options**:

* `judge_model_server`: Which model to use for judge fallback
* `should_use_judge`: Enable/disable LLM judge fallback
* `judge_responses_create_params`: Judge model parameters

**Hybrid scoring**: Prioritize library verification (fast, deterministic), fall back to LLM judge when needed.

---

## Execution Patterns

Verify by executing code or running tests.

### Test-Based Execution

**Overview**: Execute agent-generated code and check if it passes test cases.

**When to use**:

* Code generation tasks
* Objective pass/fail criteria from tests
* Computational correctness requirements

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Deterministic (passes tests or doesn't)
* - **Speed**
  - Depends on test execution time
* - **Safety**
  - Requires sandboxing for untrusted code
* - **Cost**
  - Moderate (execution overhead)
```

**Built-in example**: `resources_servers/comp_coding/`

**Approach**:

1. Extract code from agent response (using code fence detection)
2. Execute code with test inputs in sandboxed environment
3. Compare outputs against expected outputs
4. Reward based on pass rate

```python
# From comp_coding/app.py:143
reward=1.0 if all(r == True for r in result) else 0.0
```

**Source**: `resources_servers/comp_coding/app.py:79-149`

**Execution environment**: Ray workers with process pool for parallel execution.

**Configuration options**:

* `num_processes`: Parallelism for test execution
* `unit_test_timeout_secs`: Maximum time per test
* `debug`: Enable debug output

**Reward variants**:

* **Binary**: All tests pass (1.0) or any fail (0.0)
* **Partial credit**: Pass rate (e.g., 3/5 tests = 0.6)

---

### Python Execution with Result Checking

**Overview**: Execute Python code and verify final computed value matches expected result.

**When to use**:

* Math problems solved via code
* Agent uses code interpreter or execution tools
* Final value is checkable (not multi-step verification)

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Deterministic for numerical results
* - **Speed**
  - Fast for simple calculations
* - **Flexibility**
  - Works for any computable answer
* - **Safety**
  - Requires sandboxed execution
```

**Built-in example**: `resources_servers/python_math_exec/`

**Approach**:

1. Agent writes and executes Python code
2. Extract final answer from agent's response (often in `\boxed{}` format)
3. Compare extracted answer to expected result
4. Binary reward: correct (1.0) or incorrect (0.0)

```python
# From python_math_exec/app.py:232-233
accuracy = str(actual) == str(expected)
reward = 1.0 if accuracy else 0.0
```

**Source**: `resources_servers/python_math_exec/app.py:214-240`

**Session management**: Maintains Python execution sessions per agent interaction.

**Configuration options**:

* Python version and available libraries
* Session timeout and cleanup

---

## Quality Patterns

Verify adherence to constraints or stylistic requirements.

### Constraint Checking (Instruction Following)

**Overview**: Enumerate specific constraints and verify each is satisfied.

**When to use**:

* Tasks with explicit requirements (format, length, content)
* All-or-nothing compliance needed
* Objective, checkable constraints

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Deterministic if constraints are clear
* - **Speed**
  - Fast (simple checks)
* - **Strictness**
  - Binary pass/fail (no partial credit)
* - **Cost**
  - Minimal
```

**Built-in example**: `resources_servers/instruction_following/`

**Approach**:

1. List all constraints the agent must satisfy
2. Check each constraint independently
3. Reward 1.0 if ALL constraints met, 0.0 if ANY violated

**Common constraint types**:

* Length requirements ("Use exactly three sentences")
* Forbidden words ("Do not use the word 'however'")
* Required keywords ("Include the word 'sustainability'")
* Format requirements ("End with a question")

**Source**: `resources_servers/instruction_following/app.py:75`

**Design consideration**: Instruction following is typically all-or-nothing to emphasize precise compliance.

---

### Structured Output Validation

**Overview**: Verify response conforms to JSON schema or structured format.

**When to use**:

* Agent must produce structured data (JSON, XML)
* Schema compliance is critical
* Downstream systems require specific format

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Accuracy**
  - Deterministic schema validation
* - **Speed**
  - Fast (JSON parsing + validation)
* - **Strictness**
  - Binary (valid schema or not)
* - **Cost**
  - Minimal
```

**Built-in example**: `resources_servers/structured_outputs/`

**Approach**:

1. Define expected JSON schema
2. Parse agent response as JSON
3. Validate against schema
4. Check required fields and types

**Source**: `resources_servers/structured_outputs/app.py:50`

**Validation aspects**:

* Schema conformance
* Required fields present
* Type correctness
* Value constraints (enums, ranges)

---

## Hybrid Patterns

Combine multiple verification dimensions.

### Multi-Criteria Scoring

**Overview**: Evaluate multiple aspects independently and combine into composite reward.

**When to use**:

* Task has multiple success dimensions (correctness + efficiency + style)
* Some criteria more important than others
* Want to track individual metrics separately

**Tradeoffs**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Aspect
  - Consideration
* - **Richness**
  - Captures multiple quality dimensions
* - **Complexity**
  - Requires weight tuning
* - **Interpretability**
  - Can track each metric separately
* - **Cost**
  - Sum of individual verification costs
```

**Built-in example**: `resources_servers/multineedle/`

**Approach**:

1. Compute each metric independently
2. Primary metric determines main reward
3. Additional metrics returned as separate fields

```python
# From multineedle/app.py:96-105
accuracy = expected == actual
set_overlap = len(set(actual) & set(expected)) / len(expected)

return MultiNeedleVerifyResponse(
    reward=float(accuracy),  # Primary metric
    accuracy=accuracy,        # Track separately
    set_overlap=set_overlap,  # Additional metric
    # ... other metrics
)
```

**Source**: `resources_servers/multineedle/app.py:86-105`

**NeMo Gym feature**: All numeric fields automatically aggregated across rollouts.

**Source**: Automatic aggregation documented in `docs/about/concepts/rollout-collection-fundamentals.md:355-380`

**Multiple metrics returned**:

* `reward` - Primary signal for training
* `accuracy` - Exact match metric
* `set_overlap` - Partial credit metric
* Custom metrics - Any numeric field

See {doc}`multi-objective-scoring` for combining strategies.

---

### Correctness + Efficiency

**Overview**: Balance accuracy with resource usage (tool calls, response length).

**When to use**:

* Correct answers are necessary but not sufficient
* Want to train for efficiency alongside correctness
* Need to penalize wasteful behavior

**Example pattern**:

```python
# Compute correctness
correctness_score = 1.0 if is_correct else 0.0

# Compute efficiency
optimal_tool_calls = 2
actual_tool_calls = count_tool_calls(response)
efficiency_score = max(0.0, 1.0 - (actual_tool_calls - optimal_tool_calls) * 0.2)

# Weighted combination
reward = 0.7 * correctness_score + 0.3 * efficiency_score
```

**Weight guidelines**:

* **Correctness should dominate** (0.6–0.8 weight)
* **Efficiency as secondary** (0.2–0.4 weight)
* Incorrect answer should score low even if efficient

---

## Decision Guide

Quick reference for selecting verification pattern:

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Task Characteristics
  - Recommended Pattern
  - Built-in Example
* - **Exact answer, canonical form**
  - Exact match
  - mcqa
* - **Multiple valid phrasings**
  - LLM judge
  - equivalence_llm_judge
* - **Math with notation flexibility**
  - Mathematical equivalence
  - library_judge_math
* - **Code generation with tests**
  - Test-based execution
  - comp_coding
* - **Python code for computation**
  - Python execution
  - python_math_exec
* - **Specific constraints**
  - Constraint checking
  - instruction_following
* - **JSON/structured output**
  - Schema validation
  - structured_outputs
* - **Multiple quality dimensions**
  - Multi-criteria
  - multineedle
* - **Correctness + efficiency**
  - Hybrid scoring
  - (custom pattern)
```

---

## Pattern Selection Criteria

### By Training Algorithm

```{list-table}
:header-rows: 1
:widths: 25 75

* - Algorithm
  - Pattern Recommendation
* - **SFT**
  - Binary patterns (exact match, test-based)
* - **DPO**
  - Continuous patterns with clear score separation
* - **PPO/RL**
  - Shaped rewards with partial credit
```

### By Domain

```{list-table}
:header-rows: 1
:widths: 25 75

* - Domain
  - Pattern Recommendation
* - **QA (closed)**
  - Exact match or LLM judge
* - **QA (open)**
  - LLM judge with semantic equivalence
* - **Math**
  - Mathematical equivalence (symbolic + judge)
* - **Code**
  - Test-based execution
* - **Writing**
  - Multi-criteria (quality + constraints)
* - **Tool use**
  - Hybrid (correctness + efficiency)
```

### By Accuracy Requirements

```{list-table}
:header-rows: 1
:widths: 25 75

* - Accuracy Need
  - Pattern Recommendation
* - **High-stakes**
  - Deterministic patterns (exact match, test-based)
* - **Moderate**
  - LLM judge with swap check
* - **Exploratory**
  - Flexible patterns, continuous rewards
```

---

## Combining Patterns

You can combine multiple patterns:

**Sequential verification**:

1. Try fast deterministic check first
2. Fall back to expensive LLM judge if inconclusive

**Parallel verification**:

1. Run multiple verifiers independently
2. Combine scores (see {doc}`multi-objective-scoring`)

**Hierarchical verification**:

1. Primary verifier determines main reward
2. Secondary verifiers provide additional signals

---

## Implementation Notes

### BaseVerifyRequest Structure

All verification receives:

```python
class BaseVerifyRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
```

**Contains**:

* `responses_create_params` - Original task (question, expected answer, tools)
* `response` - Complete agent interaction trajectory

**Source**: `nemo_gym/base_resources_server.py:38-39`

### BaseVerifyResponse Structure

All verification returns:

```python
class BaseVerifyResponse(BaseVerifyRequest):
    reward: float  # Primary training signal (0.0 to 1.0)
```

**Additional fields** (optional):

* `accuracy` - Boolean or numeric accuracy metric
* `extracted_answer` - What verifier extracted from response
* Custom metrics - Any numeric field for automatic aggregation

**Source**: `nemo_gym/base_resources_server.py:42-43`

### Performance Considerations

**Target latency**: < 100ms per verification for production scale

**Optimization strategies**:

* Use deterministic checks before expensive LLM calls
* Batch LLM judge calls when possible
* Cache verification results for repeated checks
* Parallelize test execution

**Bottleneck identification**: See {doc}`../rollout-collection/optimize-for-training/identify-bottleneck`

---

## Exploring Built-in Resource Servers

All resource servers located in `resources_servers/` directory:

```bash
# View MCQA verification
cat resources_servers/mcqa/app.py

# View comp_coding verification
cat resources_servers/comp_coding/app.py

# View equivalence_llm_judge verification
cat resources_servers/equivalence_llm_judge/app.py
```

Each `app.py` contains:

* `verify()` method implementation
* Custom request/response types
* Configuration options
* Domain-specific logic

Study these for patterns to adapt to your domain.

---

## Related Topics

### Reward Design

* {doc}`reward-shaping` - Designing effective reward signals
* {doc}`../../about/concepts/verifying-agent-results` - Verification theory

### Multi-Objective

* {doc}`multi-objective-scoring` - Combining multiple verification signals
* {doc}`../data-quality/index` - Validating verification effectiveness

### Optimization

* {doc}`../rollout-collection/optimize-for-training/index` - Collection throughput
* {doc}`../rollout-collection/optimize-for-training/production-scale` - Scaling verification

---

## Next Steps

:::{button-ref} multi-objective-scoring
:color: primary
:outline:
:ref-type: doc

Learn Multi-Objective Scoring →
:::

Or return to {doc}`index` for verification overview.
