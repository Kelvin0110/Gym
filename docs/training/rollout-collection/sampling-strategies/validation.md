(training-rollout-sampling-validation)=

# Measuring Success

Validate that your sampling strategy produces the intended data characteristics.

:::{card}

**Task**: Verify that collected rollouts match your training objectives by measuring distribution, diversity, and quality.

^^^

**This guide shows you how to**:

1. Analyze reward distributions for your strategy
2. Measure response diversity and uniqueness
3. Validate quality signals and filter thresholds
4. Compare results against expected patterns

:::

---

## Before You Start

Ensure you have these prerequisites before validating rollouts:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Rollouts collected**
  - Output JSONL file from `ng_collect_rollouts` with reward/metrics
* - **Strategy selected**
  - Know which strategy you used (SFT, DPO, RL, etc.)
* - **Python environment**
  - Python with `jq` for metrics extraction and analysis
* - **Expected patterns**
  - Understanding of what distributions your strategy should produce
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Key Validation Metrics

### Reward Distribution

The shape of your reward distribution should match your strategy.

**Check distribution**:
```bash
jq '.reward' rollouts.jsonl | python -c "
import sys
import statistics

rewards = [float(x) for x in sys.stdin]

# Basic stats
print(f'Count:  {len(rewards)}')
print(f'Mean:   {statistics.mean(rewards):.3f}')
print(f'Median: {statistics.median(rewards):.3f}')
print(f'Stdev:  {statistics.stdev(rewards):.3f}')
print(f'Min:    {min(rewards):.3f}')
print(f'Max:    {max(rewards):.3f}')

# Percentiles
sorted_rewards = sorted(rewards)
p25 = sorted_rewards[len(sorted_rewards) // 4]
p75 = sorted_rewards[3 * len(sorted_rewards) // 4]
print(f'P25:    {p25:.3f}')
print(f'P75:    {p75:.3f}')
"
```

**Expected patterns by strategy**:

| Strategy | Expected Distribution |
|----------|----------------------|
| **SFT** | Peaked at high values (0.7-1.0), long tail of failures |
| **DPO** | Bimodal or broad (0.3-0.9), good variance |
| **RL** | Initially broad, narrows over iterations |
| **Evaluation** | Tight around model's performance level |
| **Research** | Very wide (0.0-1.0), high variance |

**Visualize**:
```python
import matplotlib.pyplot as plt
import json

with open('rollouts.jsonl') as f:
    rewards = [json.loads(line)['reward'] for line in f]

plt.figure(figsize=(10, 6))
plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('reward_distribution.png')
```

---

## Response Diversity

Measure uniqueness of completions.

**Count unique responses**:
```bash
jq -r '.output[] | select(.type=="message") | .content' rollouts.jsonl | \
  sort | uniq | wc -l

# Compare to total
total=$(wc -l < rollouts.jsonl)
unique=$(jq -r '.output[] | select(.type=="message") | .content' rollouts.jsonl | sort | uniq | wc -l)

echo "Diversity: $unique / $total = $(python -c "print(f'{$unique/$total:.1%}')")"
```

**Expected diversity by temperature**:

| Temperature | Unique Responses |
|-------------|------------------|
| **0.1-0.3** | 30-50% |
| **0.4-0.6** | 60-80% |
| **0.7-0.9** | 85-95% |

**If diversity too low**:
- ❌ Temperature too low for goal
- ❌ Seed accidentally fixed
- ✅ Increase temperature

**If diversity too high but quality low**:
- ❌ Temperature too high
- ✅ Reduce temperature

---

## Success Rate

Percentage of rollouts meeting quality threshold.

```bash
# Success rate at threshold 0.8
success=$(jq 'select(.reward >= 0.8)' rollouts.jsonl | wc -l)
total=$(wc -l < rollouts.jsonl)

echo "Success rate: $success / $total = $(python -c "print(f'{$success/$total:.1%}')")"
```

**Expected by strategy**:

| Strategy | Success Rate (reward ≥ 0.8) |
|----------|------------------------------|
| **SFT** | 60-80% (after filtering) |
| **DPO** | 40-60% (want variance) |
| **RL** | Starts low (30%), improves to 70% |
| **Evaluation** | Depends on model capability |
| **Research** | 20-40% (high temp) |

---

## Red Flags

### All Rewards Identical

```bash
# Check if all rewards the same
jq '.reward' rollouts.jsonl | sort | uniq
```

**If output is single value**:
- ❌ Temperature set to 0.0 (fully deterministic)
- ❌ Verification function returns constant
- ❌ Seed fixed unintentionally

**Fix**: Check temperature > 0.1, verify verification logic.

### Extremely Low Success Rate (<20%)

```bash
failure_rate=$(jq 'select(.reward < 0.3)' rollouts.jsonl | wc -l)
echo "Failure rate: $(python -c "print(f'{$failure_rate/$total:.1%}')")"
```

**If >80% failures**:
- ❌ Temperature too high (try ≤0.8)
- ❌ Task too difficult for model
- ❌ Verification too strict

### No Diversity in Repeated Samples

For `num_repeats > 1`, check variance within groups:

```python
import json
import statistics

# Check first task's repeats (assuming num_repeats=5)
with open('rollouts.jsonl') as f:
    first_group = [json.loads(next(f))['reward'] for _ in range(5)]

print(f"First task variance: σ={statistics.stdev(first_group):.3f}")

if statistics.stdev(first_group) < 0.01:
    print("WARNING: No diversity in repeats!")
```

**If no variance**:
- ❌ Temperature too low
- ❌ Seed being reused across repeats
- ✅ Increase temperature

### Nonsensical Outputs

Sample and manually review:

```bash
# Random sample of 5 rollouts
jq '.output[] | select(.type=="message") | .content' rollouts.jsonl | shuf -n 5
```

**If responses are gibberish**:
- ❌ Temperature too high (reduce to ≤0.9)
- ❌ Top_p too high (reduce to 0.9)
- ❌ Model issue (check model health)

---

## Strategy-Specific Validation

### For SFT

**Check**: High-reward peak, reasonable filtering rate

```python
high_reward = sum(1 for r in rewards if r >= 0.8)
print(f"High-reward rate: {high_reward / len(rewards):.1%}")
# Target: 60-80%
```

### For DPO

**Check**: Pair creation rate, reward gaps

```python
# After creating pairs (see dpo.md)
pair_rate = len(pairs) / len(groups)
print(f"Pair creation rate: {pair_rate:.1%}")
# Target: 60-80%

gaps = [p['reward_gap'] for p in pairs]
print(f"Average gap: {statistics.mean(gaps):.3f}")
# Target: 0.2-0.4
```

### For RL

**Check**: Reward progression over iterations

```bash
for f in outputs/rl_iter_*.jsonl; do
    echo -n "$(basename $f): "
    jq -s 'map(.reward) | add/length' < $f
done
```

**Target**: Monotonic increase in average reward.

### For Evaluation

**Check**: Low variance across runs

```bash
# Run twice with same seed
run1_avg=$(jq -s 'map(.reward) | add/length' eval_run1.jsonl)
run2_avg=$(jq -s 'map(.reward) | add/length' eval_run2.jsonl)

diff=$(python -c "print(abs($run1_avg - $run2_avg))")
echo "Variance between runs: $diff"
# Target: <0.02
```

---

## Automated Validation Script

```python
import json
import statistics

def validate_rollouts(jsonl_path, strategy, expected_config):
    """Automated validation of rollout collection."""
    
    # Load rollouts
    with open(jsonl_path) as f:
        rollouts = [json.loads(line) for line in f]
    
    rewards = [r['reward'] for r in rollouts]
    
    # Compute metrics
    metrics = {
        'count': len(rollouts),
        'mean_reward': statistics.mean(rewards),
        'std_reward': statistics.stdev(rewards),
        'success_rate': sum(1 for r in rewards if r >= 0.8) / len(rewards),
        'failure_rate': sum(1 for r in rewards if r < 0.3) / len(rewards)
    }
    
    # Validate against strategy expectations
    issues = []
    
    if strategy == 'sft':
        if metrics['mean_reward'] < 0.6:
            issues.append("Mean reward too low for SFT (expected ≥0.6)")
        if metrics['success_rate'] < 0.5:
            issues.append("Success rate too low for SFT (expected ≥50%)")
    
    elif strategy == 'dpo':
        if metrics['std_reward'] < 0.15:
            issues.append("Variance too low for DPO (expected ≥0.15)")
    
    elif strategy == 'rl':
        # Check against previous iteration (if available)
        pass  # Implement iteration comparison
    
    # Report
    print(f"=== Validation Report: {strategy.upper()} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    if issues:
        print("\n⚠️  Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Validation passed")
    
    return metrics, issues

# Usage
metrics, issues = validate_rollouts('rollouts.jsonl', strategy='sft', expected_config={})
```
