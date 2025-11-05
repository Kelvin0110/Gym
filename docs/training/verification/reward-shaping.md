(training-verification-reward-shaping)=

# Reward Shaping

Design effective reward signals that drive reinforcement learning by shaping how your resource server's `verify()` method scores agent performance.

:::{card}

**Goal**: Create reward signals aligned with training objectives.

^^^

**You'll learn how to**:

1. Choose between binary and continuous rewards
2. Design sparse and dense reward patterns
3. Shape rewards for exploration and learning
4. Validate reward effectiveness

:::

**Prerequisites**: Basic verification understanding from {doc}`../../get-started/verifying-agent-results`.

---

## When You Need This

Use reward shaping when:

* **Training struggles** - Agent shows little improvement despite many rollouts
* **Binary isn't enough** - Need to reward partial progress or quality gradations
* **RL training** - Moving beyond SFT/DPO to on-policy algorithms
* **Multi-phase tasks** - Want to reward intermediate milestones

:::{tip}
**Already have reward logic?** Use {doc}`verification-patterns` to compare approaches or {doc}`multi-objective-scoring` to balance multiple objectives.
:::

---

## Binary Rewards

Binary rewards provide clear success/failure signals—the simplest and most interpretable reward pattern.

### When to Use Binary

**Best for**:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Training Type
  - Why Binary Works
* - **SFT (Supervised Fine-Tuning)**
  - Filter for correct examples only
* - **DPO pairs**
  - Need clear preference ranking (1.0 vs 0.0)
* - **Clear success criteria**
  - Pass/fail tasks (code tests, exact match)
```

**Examples from built-in resource servers**:

- **mcqa**: Multiple choice—either correct (1.0) or incorrect (0.0)
- **comp_coding**: All tests pass (1.0) or any fail (0.0)
- **instruction_following**: All constraints met (1.0) or any violated (0.0)

### Implementing Binary Rewards

In your resource server's `verify()` method, return `reward=1.0` or `reward=0.0`:

```python
# In resources_servers/your_server/app.py
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    # Extract and check answer
    is_correct = (extracted_answer == expected_answer)
    
    return BaseVerifyResponse(
        **body.model_dump(),
        reward=1.0 if is_correct else 0.0
    )
```

**Source**: Pattern from `resources_servers/mcqa/app.py:261`

### Binary Reward Limitations

Binary rewards work poorly when:

* **Progress matters** - No signal for "almost correct" attempts
* **Multiple quality levels** - Can't distinguish good from excellent
* **Exploration needed** - RL agents get no guidance from failures

For these cases, consider continuous rewards.

---

## Continuous Rewards

Continuous rewards (values between 0.0 and 1.0) provide nuanced feedback for partial correctness or quality levels.

### When to Use Continuous

**Best for**:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Scenario
  - Why Continuous Helps
* - **Partial correctness**
  - Reward progress toward full solution
* - **Quality gradations**
  - Distinguish good/better/best responses
* - **RL training**
  - Provide richer learning signal
* - **Multi-criteria tasks**
  - Combine multiple scoring dimensions
```

**Examples from built-in resource servers**:

- **multineedle**: Set overlap score (0.0–1.0) based on how many items matched
- **library_judge_math**: Hybrid library + LLM judge scoring
- **python_math_exec**: Continuous score for numerical correctness

### Implementing Continuous Rewards

Calculate a score between 0.0 and 1.0:

```python
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    # Calculate partial credit
    expected_items = set(body.expected_values)
    actual_items = set(extracted_values)
    
    overlap = len(actual_items & expected_items) / len(expected_items)
    
    return BaseVerifyResponse(
        **body.model_dump(),
        reward=float(overlap)  # 0.0 to 1.0
    )
```

**Source**: Pattern from `resources_servers/multineedle/app.py:96`

### Designing Continuous Scales

**Key principles**:

1. **Meaningful differences** - Ensure 0.6 vs 0.7 represents real quality gap
2. **Avoid clustering** - Don't let all scores fall in narrow range (0.4–0.6)
3. **Bounded range** - Keep rewards in [0.0, 1.0] for consistency
4. **Interpretability** - Document what each score level means

**Common patterns**:

```python
# Percentage correct
reward = num_correct / total_items

# Distance-based (invert so closer = better)
reward = 1.0 - (abs(actual - expected) / max_possible_distance)

# Threshold-based levels
if quality >= 0.9:
    reward = 1.0
elif quality >= 0.7:
    reward = 0.8
elif quality >= 0.5:
    reward = 0.6
else:
    reward = quality * 0.5  # Scale lower scores
```

---

## Sparse vs Dense Rewards

Reward frequency impacts how quickly agents learn.

### Sparse Rewards

**Definition**: Reward signal only at task completion.

**Characteristics**:

* Single reward per episode (at the end)
* Common in goal-based tasks
* Simpler to implement

**When to use**:

* Task success is well-defined endpoint
* Intermediate steps don't have clear quality
* Training with DPO or SFT (not on-policy RL)

**Example**: Math problem—reward only when final answer checked, not during calculation steps.

### Dense Rewards

**Definition**: Reward signals throughout task execution.

**Characteristics**:

* Multiple rewards per episode (after each action)
* Rewards for intermediate progress
* Guides exploration more effectively

**When to use**:

* On-policy RL training (PPO, TRPO)
* Long-horizon tasks where agents get lost
* Want to reward process, not just outcome

**Example**: Multi-step reasoning—reward each valid reasoning step, not just final answer.

**NeMo Gym Pattern**: Most built-in resource servers use sparse rewards (single score per rollout) because verification happens after agent completes interaction. Dense rewards require custom agent modifications.

---

## Shaped Rewards for RL

Reward shaping provides guidance that helps RL agents learn faster by giving credit for progress.

### What Is Reward Shaping?

Transform sparse binary rewards into richer signals:

```python
# Before shaping: Binary
reward = 1.0 if correct else 0.0

# After shaping: Continuous with partial credit
if correct:
    reward = 1.0
elif close_to_correct:
    reward = 0.7  # Good attempt
elif on_right_track:
    reward = 0.4  # Some progress
else:
    reward = 0.0  # No progress
```

### Shaping Patterns

**1. Distance-based shaping**

Reward proximity to goal:

```python
# Example: Extracting values from text
expected_count = len(expected_values)
actual_count = len(extracted_values)

if extracted_values == expected_values:
    reward = 1.0  # Perfect
else:
    # Partial credit for getting some
    overlap = len(set(extracted_values) & set(expected_values))
    reward = overlap / expected_count
```

**Source**: Similar to `resources_servers/multineedle/app.py:96`

**2. Milestone rewards**

Credit for reaching intermediate goals:

```python
reward = 0.0

# Progressive milestones
if called_required_tool:
    reward += 0.3
if extracted_data_from_tool:
    reward += 0.3
if used_data_in_response:
    reward += 0.4

# Total: 0.0 (nothing), 0.3 (called), 0.6 (extracted), 1.0 (complete)
```

**3. Quality tiers**

Different levels of solution quality:

```python
if solution_optimal:
    reward = 1.0
elif solution_correct_but_inefficient:
    reward = 0.8
elif solution_has_minor_errors:
    reward = 0.5
elif attempted_solution:
    reward = 0.2
else:
    reward = 0.0
```

### Shaping Pitfalls

**Avoid**:

* **Reward hacking** - Agent finds shortcuts to maximize reward without solving task
* **Local optima** - Shaped rewards lead away from true goal
* **Over-shaping** - Too much guidance prevents exploration

**Safe approach**: Start with sparse rewards, add shaping only if training stalls.

---

## Testing Your Rewards

Validate reward design before large-scale training.

### 1. Check Distribution

After collecting rollouts, examine reward distribution:

```bash
ng_collect_rollouts +input_jsonl_fpath=tasks.jsonl +output_jsonl_fpath=rollouts.jsonl

# Automatic aggregation shows:
# {
#   "reward": 0.73,
#   "accuracy": 0.68
# }
```

**Source**: Automatic aggregation from `ng_collect_rollouts` (documented in `docs/about/concepts/rollout-collection-fundamentals.md:355`)

**Red flags**:

* All rewards identical → Binary verification may be too strict or broken
* All rewards near 0.0 → Task too hard or verification too harsh  
* All rewards near 1.0 → Task too easy or verification too lenient
* Narrow range (0.45–0.55) → Rewards not discriminative enough

### 2. Manual Inspection

Spot-check rewards match quality:

```python
import json

# Load rollouts
rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Find high and low rewards
high_reward = max(rollouts, key=lambda r: r['reward'])
low_reward = min(rollouts, key=lambda r: r['reward'])

print(f"High reward ({high_reward['reward']}):")
print(high_reward['response']['output'][-1])  # Final response

print(f"\nLow reward ({low_reward['reward']}):")
print(low_reward['response']['output'][-1])
```

**Validate**: Does high-reward response genuinely demonstrate better performance?

### 3. Compare Variations

Test edge cases:

```python
test_cases = [
    {"input": "...", "expected": "perfect", "expected_reward": 1.0},
    {"input": "...", "expected": "partial", "expected_reward": 0.6},
    {"input": "...", "expected": "wrong", "expected_reward": 0.0},
]
```

Ensure rewards match intentions.

### 4. Training Signal Check

Good rewards should:

* **Discriminate** - Different quality responses get different scores
* **Correlate** - Higher scores correspond to better human judgments
* **Stable** - Similar responses get similar scores (not random)

**Tool**: Use `python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl` for detailed metrics.

**Source**: Built-in utility at `scripts/print_aggregate_results.py`

---

## Reward Design Checklist

Before training, verify:

- [ ] **Reward range is [0.0, 1.0]** - Stays within expected bounds
- [ ] **Distribution is reasonable** - Not all same value, not too narrow
- [ ] **High rewards = high quality** - Manual inspection confirms alignment
- [ ] **Handles edge cases** - Unexpected inputs don't break verification
- [ ] **Computation is fast** - Verification takes < 100ms per rollout (for production scale)
- [ ] **Documented** - Team understands what each score level means

---

## Examples by Training Type

### SFT (Supervised Fine-Tuning)

**Goal**: Filter for correct examples only.

**Reward pattern**: Binary (1.0 = include, 0.0 = exclude)

```python
# Keep only perfect responses
reward = 1.0 if is_correct else 0.0
```

**Filter during dataset prep**:

```python
correct_only = [r for r in rollouts if r['reward'] == 1.0]
```

### DPO (Direct Preference Optimization)

**Goal**: Create preference pairs (chosen vs rejected).

**Reward pattern**: Continuous with clear gaps

```python
# Ensure meaningful difference between pairs
reward = calculate_quality(response)  # 0.0 to 1.0

# Later, pair high-reward with low-reward
# e.g., (0.9, 0.3) not (0.6, 0.5)
```

**Recommended minimum gap**: 0.2 between chosen and rejected.

### PPO (Proximal Policy Optimization)

**Goal**: On-policy RL with shaped rewards.

**Reward pattern**: Continuous with partial credit

```python
# Reward progress, not just success
if fully_correct:
    reward = 1.0
elif mostly_correct:
    reward = 0.7
elif some_progress:
    reward = 0.4
else:
    reward = 0.1  # Small credit for attempting
```

**Key**: Never all-zero rewards—RL needs gradient.

---

## Related Topics

### Verification Approaches

* {doc}`verification-patterns` - Catalog of verification patterns with tradeoffs
* {doc}`../../about/concepts/verifying-agent-results` - Verification theory and design

### Multi-Objective

* {doc}`multi-objective-scoring` - Combining multiple reward signals
* {doc}`../data-quality/index` - Validating reward distributions

### Collection Pipeline

* {doc}`../rollout-collection/sampling-strategies/index` - Sampling strategies by training algorithm
* {doc}`../rollout-collection/optimize-for-training/index` - Optimizing collection throughput

---

## Next Steps

:::{button-ref} verification-patterns
:color: primary
:outline:
:ref-type: doc

Explore Verification Patterns →
:::

Or continue to {doc}`multi-objective-scoring` for combining multiple signals.
