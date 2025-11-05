(training-rollout-patterns-development)=

# Development Patterns

Proven patterns for rapid development, debugging, parameter tuning, and verification testing.

---

## Quick Debug (5 Samples)

**Use Case**: Rapid iteration when developing agents  
**Goal**: <30 second feedback loop

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=debug_tasks.jsonl \
    +output_jsonl_fpath=/tmp/debug_rollouts.jsonl \
    +limit=5 \
    +num_samples_in_parallel=1
```

**Why parallelism=1**: Sequential execution easier to debug, clear ordering.

### Quick Inspection

```bash
# View first rollout
head -1 /tmp/debug_rollouts.jsonl | jq '.'

# Check all rewards
jq '.reward' /tmp/debug_rollouts.jsonl

# Interactive viewer
ng_viewer +input_jsonl_fpath=/tmp/debug_rollouts.jsonl
```

---

## Parameter Sweep

**Use Case**: Find optimal temperature or other parameters  
**Goal**: Compare multiple configurations

### Sweep Script

```bash
#!/bin/bash
# Temperature sweep to find optimal setting

TEMPS=(0.1 0.3 0.5 0.7 0.9)

for temp in "${TEMPS[@]}"; do
    echo "Testing temperature $temp..."
    
    ng_collect_rollouts \
        +agent_name=my_agent \
        +input_jsonl_fpath=sweep_tasks.jsonl \
        +output_jsonl_fpath=sweep_temp_${temp}.jsonl \
        +responses_create_params.temperature=$temp \
        +limit=100
    
    # Compute metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' sweep_temp_${temp}.jsonl)
    echo "Temperature $temp: avg_reward=$avg_reward"
done

echo "---"
echo "Summary:"
for temp in "${TEMPS[@]}"; do
    avg_reward=$(jq -s 'map(.reward) | add/length' sweep_temp_${temp}.jsonl)
    echo "temp=$temp: $avg_reward"
done
```

### Expected Output

```
temp=0.1: 0.634
temp=0.3: 0.672
temp=0.5: 0.689
temp=0.7: 0.671
temp=0.9: 0.623
```

Pick `temp=0.5` (highest reward) for production.

---

## Behavioral Exploration

**Use Case**: Understand agent capabilities and failure modes  
**Goal**: Surface diverse strategies and edge cases

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=research_tasks.jsonl \
    +output_jsonl_fpath=exploration_rollouts.jsonl \
    +responses_create_params.temperature=0.9 \
    +num_repeats=5 \
    +limit=50 \
    +num_samples_in_parallel=10
```

**Total rollouts**: 50 tasks × 5 repeats = 250 rollouts

### Analysis Workflow

**1. Interactive inspection**:
```bash
ng_viewer +input_jsonl_fpath=exploration_rollouts.jsonl
```

**2. Find failures**:
```bash
jq 'select(.reward < 0.3)' exploration_rollouts.jsonl > failures.jsonl

echo "Failure rate: $(wc -l < failures.jsonl) / $(wc -l < exploration_rollouts.jsonl)"

# Inspect first failure
head -1 failures.jsonl | jq '.output'
```

**3. Measure per-task variance**:
```python
import json
from collections import defaultdict
import statistics

# Group by task (every 5 rollouts)
groups = defaultdict(list)
with open('exploration_rollouts.jsonl') as f:
    for i, line in enumerate(f):
        rollout = json.loads(line)
        task_id = i // 5
        groups[task_id].append(rollout['reward'])

# Compute statistics
for task_id, rewards in groups.items():
    print(f"Task {task_id}: "
          f"μ={statistics.mean(rewards):.2f}, "
          f"σ={statistics.stdev(rewards):.2f}, "
          f"range=[{min(rewards):.2f}, {max(rewards):.2f}]")
```

---

## Verification Testing

**Use Case**: Iterate on verification function  
**Goal**: Quick feedback on reward signal changes

### Command

```bash
ng_collect_rollouts \
    +agent_name=test_verifier \
    +input_jsonl_fpath=verification_test.jsonl \
    +output_jsonl_fpath=/tmp/verify_test.jsonl \
    +limit=20 \
    +num_samples_in_parallel=5
```

### Check Reward Distribution

```bash
# Count reward values
jq '.reward' /tmp/verify_test.jsonl | sort -n | uniq -c

# Example output:
#   3 0.0
#   7 0.5
#  10 1.0
```

**Iterate**:
1. Update `verify()` function in resource server
2. Restart server
3. Re-run collection command
4. Compare reward distributions

---

## Next Steps

**Scale patterns**: {ref}`training-rollout-patterns-scale`  
Handle large-scale and long-running collection jobs.

**Training preparation**: {ref}`training-rollout-patterns-training-prep`  
Apply these development patterns to production training data generation.

**Quick reference**: {ref}`training-rollout-patterns-quick-reference`  
Essential commands at a glance.

