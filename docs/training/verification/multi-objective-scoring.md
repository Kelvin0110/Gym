(training-verification-multi-objective)=

# Multi-Objective Scoring

Balance multiple objectives or combine several reward signals using NeMo Gym's automatic metric aggregation and composite scoring patterns.

:::{card}

**Goal**: Effectively combine multiple verification signals.

^^^

**You'll learn how to**:

1. Use NeMo Gym's automatic metric aggregation
2. Combine metrics in `verify()` method
3. Design weighted composite rewards
4. Monitor multiple objectives during training

:::

**Prerequisites**: Understanding of basic verification from {doc}`../../get-started/verifying-agent-results` and {doc}`reward-shaping`.

---

## When You Need Multi-Objective Scoring

Use multi-objective scoring when:

* **Multiple success criteria** - Task has several independent quality dimensions
* **Conflicting objectives** - Improving one metric may hurt another (accuracy vs speed)
* **Hierarchical goals** - Primary objective must be met, then optimize secondary goals
* **Rich training signal** - Want to track multiple metrics beyond single reward

**Examples**:

* **Code generation** - Correctness + efficiency + style
* **Search tasks** - Relevance + coverage + conciseness
* **Tool use** - Accuracy + minimal tool calls + response quality

---

## NeMo Gym's Automatic Aggregation

NeMo Gym automatically aggregates any numeric field returned from `verify()` across all rollouts—no additional code required.

### How It Works

**In your resource server**:

```python
async def verify(self, body: YourVerifyRequest) -> YourVerifyResponse:
    # Calculate multiple metrics
    correctness = 1.0 if is_correct else 0.0
    efficiency = calculate_efficiency_score(response)
    
    return YourVerifyResponse(
        **body.model_dump(),
        reward=correctness * 0.7 + efficiency * 0.3,  # Composite
        correctness=correctness,  # ← Automatically aggregated
        efficiency=efficiency      # ← Automatically aggregated
    )
```

**After collection**:

```bash
ng_collect_rollouts +input_jsonl_fpath=tasks.jsonl +output_jsonl_fpath=rollouts.jsonl

# Automatic output:
# {
#   "reward": 0.73,
#   "correctness": 0.85,
#   "efficiency": 0.64
# }
```

**Source**: Automatic aggregation from `nemo_gym/train_data_utils.py:224-238`

### What Gets Aggregated

**Automatically averaged**:

* `reward` - Primary training signal
* Any numeric field (int, float, bool converted to int)
* Custom metrics you define

**Not aggregated**:

* String fields
* `responses_create_params` and `response` (full rollout data)

**Source**: Logic in `nemo_gym/train_data_utils.py:224-238`

### Built-in Analysis

View aggregated metrics:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

Shows averages, min, max for all numeric fields.

**Source**: Utility at `scripts/print_aggregate_results.py`

---

## Weighted Combination

Combine multiple objectives into single composite reward.

### Basic Pattern

Weight objectives by importance:

```python
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    # Calculate individual scores (0.0 to 1.0)
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    style = check_style(response)
    
    # Weighted combination
    reward = (
        0.6 * correctness +  # Primary: 60%
        0.3 * efficiency +   # Secondary: 30%
        0.1 * style          # Tertiary: 10%
    )
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Key principle**: Weights should sum to 1.0 for interpretability.

### Choosing Weights

**Guidelines**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Objective Type
  - Weight Range
* - **Primary (must-have)**
  - 0.5–0.7
* - **Secondary (important)**
  - 0.2–0.3
* - **Tertiary (nice-to-have)**
  - 0.1–0.2
```

**Example weight distributions**:

* **Correctness-dominant**: 0.7 correctness + 0.2 efficiency + 0.1 style
* **Balanced**: 0.5 accuracy + 0.3 relevance + 0.2 completeness
* **Quality-focused**: 0.6 correctness + 0.4 quality (binary efficiency)

### Testing Weight Combinations

Experiment with different weights:

```python
# Test different weight configurations
configs = [
    {"correctness": 0.8, "efficiency": 0.2},  # Correctness-heavy
    {"correctness": 0.6, "efficiency": 0.4},  # Balanced
    {"correctness": 0.5, "efficiency": 0.5},  # Equal weight
]

for config in configs:
    reward = (config["correctness"] * correctness_score + 
              config["efficiency"] * efficiency_score)
    # Evaluate if reward distribution is reasonable
```

**Goal**: Ensure high-reward rollouts align with your quality definition.

---

## Multiple Independent Metrics

Track multiple objectives without combining into single reward.

### Pattern: Return Multiple Fields

```python
async def verify(self, body: YourVerifyRequest) -> YourVerifyResponse:
    # Calculate metrics independently
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    coverage = check_coverage(response)
    
    # Primary reward (choose most important)
    reward = correctness
    
    # Return all metrics
    return YourVerifyResponse(
        **body.model_dump(),
        reward=reward,         # Primary training signal
        correctness=correctness,  # Track separately
        efficiency=efficiency,    # Track separately
        coverage=coverage         # Track separately
    )
```

**Built-in example**: `resources_servers/multineedle/`

```python
# From multineedle/app.py:97-105
return MultiNeedleVerifyResponse(
    reward=float(accuracy),  # Primary metric
    accuracy=accuracy,        # Boolean metric
    set_overlap=set_overlap,  # Continuous metric
    original_term_minefield_hit=...,  # Additional tracking
    order_instruction_following_failure=...,  # Additional tracking
)
```

**Source**: `resources_servers/multineedle/app.py:86-105`

### When to Use Multiple Metrics

**Best for**:

* Exploratory analysis - Not sure which metric matters most yet
* Monitoring - Track multiple dimensions during training
* Post-hoc filtering - Filter rollouts by different criteria
* A/B testing - Compare different reward formulations

**Downstream usage**:

```python
import json

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

# Filter by different criteria
high_correctness = [r for r in rollouts if r['correctness'] > 0.9]
high_efficiency = [r for r in rollouts if r['efficiency'] > 0.8]
both_high = [r for r in rollouts if r['correctness'] > 0.9 and r['efficiency'] > 0.8]
```

---

## Hierarchical Objectives

Enforce must-have requirements before optimizing secondary goals.

### Pattern: Threshold-Based Weighting

```python
async def verify(self, body: YourVerifyRequest) -> BaseVerifyResponse:
    correctness = check_correctness(response)
    efficiency = check_efficiency(response)
    
    # Hierarchical: correctness is gating
    if correctness < 0.5:
        # Failed primary objective - low reward regardless of efficiency
        reward = correctness * 0.3
    else:
        # Met primary objective - now optimize efficiency too
        reward = 0.6 * correctness + 0.4 * efficiency
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Effect**: Agent must meet primary objective before secondary objective contributes.

### Pattern: Multiplicative Gating

```python
# Correctness gates efficiency reward
if is_correct:
    reward = 0.8 + 0.2 * efficiency  # 0.8 to 1.0 based on efficiency
else:
    reward = 0.3 * partial_correctness  # Max 0.3 if incorrect
```

**Effect**: Incorrect answers can never score high, regardless of efficiency.

---

## Common Multi-Objective Patterns

### Correctness + Efficiency

**Goal**: Balance accuracy with resource usage.

```python
# Measure efficiency
optimal_tool_calls = 2
actual_tool_calls = count_tool_calls(response.output)
tool_call_penalty = max(0, actual_tool_calls - optimal_tool_calls) * 0.1

# Combine
correctness = 1.0 if is_correct else 0.0
efficiency = max(0.0, 1.0 - tool_call_penalty)

reward = 0.7 * correctness + 0.3 * efficiency
```

**Weight guideline**: Correctness should dominate (0.6–0.8).

### Accuracy + Completeness

**Goal**: Reward both correct and comprehensive responses.

```python
# Accuracy: how much is correct
accuracy = correct_items / total_items_attempted

# Completeness: how much was attempted
completeness = total_items_attempted / total_items_required

# Combine
reward = 0.6 * accuracy + 0.4 * completeness
```

**Use case**: Information extraction, multi-part questions.

### Precision + Recall

**Goal**: Balance false positives vs false negatives.

```python
# Calculate metrics
true_positives = len(set(predicted) & set(actual))
precision = true_positives / len(predicted) if predicted else 0.0
recall = true_positives / len(actual) if actual else 0.0

# Combine (F1-like)
if precision + recall > 0:
    reward = 2 * (precision * recall) / (precision + recall)
else:
    reward = 0.0
```

**Use case**: Classification, extraction tasks.

**Alternative**: F-beta score with custom beta to weight precision vs recall differently.

### Speed + Quality

**Goal**: Reward fast responses without sacrificing quality.

```python
# Quality check (binary)
quality = 1.0 if meets_quality_threshold(response) else 0.0

# Speed score (normalized)
max_acceptable_time = 10.0  # seconds
time_score = max(0.0, 1.0 - (response_time / max_acceptable_time))

# Hierarchical: quality gates speed reward
if quality > 0:
    reward = 0.8 * quality + 0.2 * time_score
else:
    reward = 0.0  # No credit for fast but wrong answers
```

---

## Monitoring Multi-Objective Training

### Track Metrics Over Time

After each collection run, monitor all metrics:

```bash
# Run 1
ng_collect_rollouts ... +output_jsonl_fpath=rollouts_iter1.jsonl
# Output: {"reward": 0.65, "correctness": 0.75, "efficiency": 0.60}

# Run 2 (after training)
ng_collect_rollouts ... +output_jsonl_fpath=rollouts_iter2.jsonl
# Output: {"reward": 0.71, "correctness": 0.82, "efficiency": 0.65}
```

**Analyze**:

* Did all metrics improve?
* Did optimizing one hurt another? (trade-off)
* Is improvement aligned with weights?

### Analyze Trade-Offs

```python
import json
import matplotlib.pyplot as plt

rollouts = [json.loads(line) for line in open('rollouts.jsonl')]

correctness = [r['correctness'] for r in rollouts]
efficiency = [r['efficiency'] for r in rollouts]

# Scatter plot to visualize trade-off
plt.scatter(correctness, efficiency)
plt.xlabel('Correctness')
plt.ylabel('Efficiency')
plt.title('Trade-off Analysis')
plt.show()
```

**Look for**:

* Negative correlation? (trade-off exists)
* Pareto frontier? (best achievable balance)
* Clusters? (different strategies)

### Adjusting Weights

If training optimizes wrong objective:

1. **Increase weight** on underperforming objective
2. **Re-collect rollouts** with adjusted reward
3. **Compare distributions** before and after
4. **Iterate** until balance matches goals

**Example**:

```yaml
iteration_1:
  weights: {correctness: 0.6, efficiency: 0.4}
  results: {correctness: 0.85, efficiency: 0.50}
  # Efficiency too low!

iteration_2:
  weights: {correctness: 0.5, efficiency: 0.5}  # Increase efficiency weight
  results: {correctness: 0.82, efficiency: 0.65}
  # Better balance
```

---

## Design Checklist

Before deploying multi-objective scoring:

- [ ] **Primary objective is clear** - One metric must be most important
- [ ] **Weights sum to 1.0** - For interpretability
- [ ] **Metrics are normalized** - All in [0.0, 1.0] range
- [ ] **Trade-offs are understood** - Know which objectives conflict
- [ ] **Hierarchical gating if needed** - Must-have vs nice-to-have
- [ ] **Tested on sample data** - Reward distribution makes sense
- [ ] **Monitoring plan** - Track all metrics during training

---

## Examples by Training Algorithm

### SFT (Supervised Fine-Tuning)

**Goal**: Filter for high-quality examples.

**Pattern**: Use multiple metrics to define quality threshold.

```python
# Return multiple metrics
reward = correctness  # Primary for filtering
correctness = ...
efficiency = ...

# Later, filter rollouts:
sft_data = [r for r in rollouts 
            if r['correctness'] == 1.0 and r['efficiency'] > 0.7]
```

### DPO (Direct Preference Optimization)

**Goal**: Create preference pairs with clear winner.

**Pattern**: Composite reward should separate chosen/rejected clearly.

```python
# Composite reward
reward = 0.6 * correctness + 0.4 * quality

# When creating pairs, ensure gap:
# chosen: reward=0.85, rejected: reward=0.45
# Gap = 0.40 (good separation)
```

**Recommendation**: Aim for minimum 0.2 reward gap between chosen and rejected.

### PPO (Proximal Policy Optimization)

**Goal**: On-policy RL with rich signal.

**Pattern**: Shaped composite rewards with partial credit.

```python
# All objectives contribute to gradient
reward = 0.5 * partial_correctness + 0.3 * progress + 0.2 * efficiency

# Even failed attempts get some reward for progress
```

---

## Built-in Multi-Objective Examples

### MultiNeedle Resource Server

**Task**: Extract multiple values from long context.

**Metrics**:

```python
# From multineedle/app.py
accuracy = expected == actual  # Binary: all correct?
set_overlap = len(set(actual) & set(expected)) / len(expected)  # Partial credit
original_term_minefield_hit = ...  # Did agent avoid trap?
order_instruction_following_failure = ...  # Order matters?

reward = float(accuracy)  # Primary metric is binary accuracy
```

**Source**: `resources_servers/multineedle/app.py:86-105`

**Design choice**: Primary reward is binary, but track partial credit metrics separately.

### Library Judge Math Resource Server

**Task**: Math problem with symbolic + judge verification.

**Metrics**:

```python
# Hybrid approach
library_reward = symbolic_equivalence_check()  # Fast, deterministic
judge_reward = llm_judge_equivalence() if library_inconclusive else None

# Combine with preference for library
if library_reward is not None:
    reward = library_reward
elif judge_reward is not None:
    reward = judge_reward
else:
    reward = 0.0
```

**Source**: `resources_servers/library_judge_math/app.py:118`

**Design choice**: Sequential fallback (library first, judge if needed) rather than weighted average.

---

## Common Pitfalls

### Pitfall 1: Conflicting Objectives

**Problem**: Optimizing one metric hurts another.

**Example**:

```python
# Reward both speed and thoroughness
reward = 0.5 * speed + 0.5 * thoroughness

# But: being thorough requires being slow!
```

**Solution**: Use hierarchical gating or adjust weights to prioritize.

### Pitfall 2: Unbalanced Scales

**Problem**: Metrics on different scales (0-100 vs 0.0-1.0).

**Example**:

```python
# Bug: metrics on different scales
reward = 0.5 * accuracy + 0.5 * response_length  # Wrong!
# accuracy ∈ [0, 1], response_length ∈ [0, 500]
```

**Solution**: Normalize all metrics to [0.0, 1.0]:

```python
max_length = 500
normalized_length = min(response_length / max_length, 1.0)
reward = 0.5 * accuracy + 0.5 * normalized_length
```

### Pitfall 3: Correlation Masking

**Problem**: Two metrics are highly correlated—combined weight is too high.

**Example**:

```python
# accuracy and correctness are essentially the same metric
reward = 0.5 * accuracy + 0.5 * correctness
# Effectively: 1.0 * correctness (doubled weight!)
```

**Solution**: Use orthogonal metrics (independent dimensions).

### Pitfall 4: Ignoring Primary Objective

**Problem**: Secondary objectives dominate because primary is too hard.

**Example**:

```python
reward = 0.7 * correctness + 0.3 * style

# If correctness always near 0.0, agent optimizes style only
```

**Solution**: Use hierarchical gating or adjust task difficulty.

---

## Related Topics

### Reward Design

* {doc}`reward-shaping` - Design effective single-objective rewards first
* {doc}`verification-patterns` - Choose verification approach for each objective

### Validation

* {doc}`../data-quality/index` - Validate multi-objective reward distributions
* {doc}`../rollout-collection/optimize-for-training/production-scale` - Monitor metrics during collection

### Training Integration

* {doc}`../datasets/index` - Multi-metric rollouts in training formats
* {doc}`../rollout-collection/sampling-strategies/index` - Sampling by training algorithm

---

## Next Steps

:::{button-ref} ../data-quality/index
:color: primary
:outline:
:ref-type: doc

Validate Data Quality →
:::

Or return to {doc}`index` for verification overview.
