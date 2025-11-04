(training-rollout-patterns-training-prep)=

# Training Preparation Patterns

Proven patterns for generating training data for supervised fine-tuning (SFT), preference optimization (DPO), reinforcement learning (RL), and evaluation benchmarks.

---

## Pattern 3.1: SFT Dataset Generation

**Use Case**: Generate supervised fine-tuning demonstrations  
**Scale**: 10K-1M rollouts  
**Infrastructure**: Any (optimize for throughput)

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=sft_train.jsonl \
    +output_jsonl_fpath=sft_rollouts.jsonl \
    +responses_create_params.temperature=0.2 \
    +responses_create_params.top_p=0.95 \
    +num_samples_in_parallel=20 \
    +limit=null
```

### What Happens

- Processes entire input dataset (no limit)
- Low temperature (0.2) ensures consistent, high-quality demonstrations
- High parallelism (20) maximizes throughput
- Single sample per task (default `num_repeats=1`)
- Prints average metrics at completion

### Expected Output

```
Found 10000 rows!
Querying with 20 concurrent requests
Collecting rollouts: 100%|████████████| 10000/10000 [08:32<00:00, 19.52it/s]
{
    "reward": 0.734,
    "accuracy": 0.689
}
```

**Throughput**: 15-20 samples/sec (typical for hosted API or local 70B model)

### Post-Processing

**Filter for high-quality rollouts**:
```bash
# Keep only high-reward demonstrations (adjust threshold based on your task)
jq 'select(.reward >= 0.8)' sft_rollouts.jsonl > sft_filtered.jsonl

echo "Filtered: $(wc -l < sft_filtered.jsonl) / $(wc -l < sft_rollouts.jsonl)"
```

**Shuffle before training**:
```bash
shuf sft_filtered.jsonl > sft_train_ready.jsonl
```

**Check task distribution** (if tasks have categories):
```bash
jq -r '.responses_create_params.input[0].content | match("Category: ([A-Z]+)").captures[0].string' sft_train_ready.jsonl | sort | uniq -c
```

---

## Pattern 3.2: DPO Preference Pairs

**Use Case**: Generate chosen/rejected pairs for Direct Preference Optimization  
**Scale**: 5K-100K pairs (10K-200K total rollouts with `num_repeats=2-4`)  
**Infrastructure**: Medium GPU or hosted API

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=dpo_prompts.jsonl \
    +output_jsonl_fpath=dpo_rollouts.jsonl \
    +responses_create_params.temperature=0.7 \
    +num_repeats=3 \
    +num_samples_in_parallel=10
```

### What Happens

- Generates 3 rollouts per input prompt
- Repeats processed in `aabbcc` pattern (3 consecutive rollouts share same input)
- Moderate temperature (0.7) creates behavioral diversity
- Each group of 3 has varying rewards for preference ranking

### Expected Output

```
Found 5000 rows!
Repeating rows (in a pattern of abc to aabbcc) from 5000 to 15000!
Querying with 10 concurrent requests
Collecting rollouts: 100%|████████████| 15000/15000 [15:23<00:00, 16.24it/s]
{
    "reward": 0.612,
    "accuracy": 0.558
}
```

**Note**: Average reward is moderate (not peaked) due to diversity.

### Post-Processing: Create Preference Pairs

**Step 1: Group by original task**

```python
import json

def group_rollouts(input_fpath, num_repeats=3):
    """Group consecutive rollouts that share same input."""
    groups = []
    current_group = []
    
    with open(input_fpath) as f:
        for i, line in enumerate(f, 1):
            rollout = json.loads(line)
            current_group.append(rollout)
            
            if i % num_repeats == 0:
                groups.append(current_group)
                current_group = []
    
    return groups

groups = group_rollouts('dpo_rollouts.jsonl', num_repeats=3)
print(f"Created {len(groups)} prompt groups (3 rollouts each)")
```

**Step 2: Select chosen and rejected**

```python
def create_dpo_pairs(groups, min_reward_gap=0.15):
    """Create preference pairs with sufficient quality difference."""
    pairs = []
    
    for group in groups:
        # Sort by reward descending
        sorted_rollouts = sorted(group, key=lambda x: x['reward'], reverse=True)
        
        # Require minimum gap for clear preference signal
        if sorted_rollouts[0]['reward'] - sorted_rollouts[1]['reward'] >= min_reward_gap:
            pairs.append({
                'prompt': sorted_rollouts[0]['responses_create_params']['input'],
                'chosen': sorted_rollouts[0]['output'],
                'rejected': sorted_rollouts[1]['output'],
                'reward_chosen': sorted_rollouts[0]['reward'],
                'reward_rejected': sorted_rollouts[1]['reward'],
                'reward_gap': sorted_rollouts[0]['reward'] - sorted_rollouts[1]['reward']
            })
    
    return pairs

pairs = create_dpo_pairs(groups, min_reward_gap=0.15)

print(f"Created {len(pairs)} preference pairs from {len(groups)} groups")
print(f"Pair creation rate: {len(pairs)/len(groups)*100:.1f}%")

# Save pairs
with open('dpo_pairs.jsonl', 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')
```

**Step 3: Analyze quality**

```python
import statistics

chosen_rewards = [p['reward_chosen'] for p in pairs]
rejected_rewards = [p['reward_rejected'] for p in pairs]
gaps = [p['reward_gap'] for p in pairs]

print(f"Chosen:   μ={statistics.mean(chosen_rewards):.3f}, σ={statistics.stdev(chosen_rewards):.3f}")
print(f"Rejected: μ={statistics.mean(rejected_rewards):.3f}, σ={statistics.stdev(rejected_rewards):.3f}")
print(f"Gap:      μ={statistics.mean(gaps):.3f}, σ={statistics.stdev(gaps):.3f}")
```

**Expected**: 60-80% pair creation rate, 0.2-0.4 average reward gap.

---

## Pattern 3.3: RL Training Buffer

**Use Case**: Collect exploration data for online reinforcement learning  
**Scale**: 1K-10K rollouts per iteration  
**Infrastructure**: Fast inference required (local GPU recommended)

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_rl_agent \
    +input_jsonl_fpath=rl_tasks.jsonl \
    +output_jsonl_fpath=outputs/rl_iter_001.jsonl \
    +responses_create_params.temperature=0.5 \
    +num_samples_in_parallel=15
```

### Iterative RL Loop

```bash
#!/bin/bash
# RL training loop with policy updates

NUM_ITERATIONS=10

for iter in $(seq -f "%03g" 1 $NUM_ITERATIONS); do
    echo "=== RL Iteration $iter ==="
    
    # Step 1: Collect rollouts with current policy
    ng_collect_rollouts \
        +agent_name=my_rl_agent \
        +input_jsonl_fpath=rl_tasks.jsonl \
        +output_jsonl_fpath=outputs/rl_iter_${iter}.jsonl \
        +responses_create_params.temperature=0.5 \
        +num_samples_in_parallel=15
    
    # Step 2: Compute iteration metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' outputs/rl_iter_${iter}.jsonl)
    echo "Average reward: $avg_reward"
    
    # Step 3: Train RL for N steps on collected data
    python train_rl.py \
        --input outputs/rl_iter_${iter}.jsonl \
        --checkpoint models/rl_checkpoint_${iter}.pt \
        --steps 1000
    
    # Step 4: Update agent configuration to use new checkpoint
    # (Agent server auto-loads latest checkpoint or update config here)
    
    echo "Completed iteration $iter"
    echo "---"
done

# Analyze reward progression
echo "Reward progression over iterations:"
for f in outputs/rl_iter_*.jsonl; do
    echo -n "$(basename $f): "
    jq -s 'map(.reward) | add/length' < $f
done
```

### Expected Progression

```
Iteration 001: reward=0.423
Iteration 002: reward=0.451
Iteration 003: reward=0.489
...
Iteration 010: reward=0.687
```

---

## Pattern 3.4: Evaluation Benchmark

**Use Case**: Reproducible model evaluation on test set  
**Scale**: 100-5K test examples  
**Infrastructure**: Any (throughput less critical)

### Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=eval_benchmark.jsonl \
    +output_jsonl_fpath=eval_model_v1_results.jsonl \
    +responses_create_params.temperature=0.1 \
    +responses_create_params.seed=42 \
    +num_samples_in_parallel=5 \
    +limit=null
```

### Model Comparison

```bash
#!/bin/bash
# Compare multiple model versions on same benchmark

BENCHMARK="eval_benchmark.jsonl"
MODELS=("model_v1" "model_v2" "model_v3")

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    
    ng_collect_rollouts \
        +agent_name=$model \
        +input_jsonl_fpath=$BENCHMARK \
        +output_jsonl_fpath=eval_${model}_results.jsonl \
        +responses_create_params.temperature=0.1 \
        +responses_create_params.seed=42 \
        +num_samples_in_parallel=5
    
    # Extract metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' eval_${model}_results.jsonl)
    accuracy=$(jq -s 'map(.accuracy) | add/length' eval_${model}_results.jsonl)
    
    echo "$model: reward=$avg_reward, accuracy=$accuracy"
done
```

### Expected Output

```
model_v1: reward=0.634, accuracy=0.589
model_v2: reward=0.701, accuracy=0.652
model_v3: reward=0.723, accuracy=0.678
```

---

## Next Steps

**Infrastructure patterns**: {ref}`training-rollout-patterns-infrastructure`  
Choose the right infrastructure for your training data generation.

**Development patterns**: {ref}`training-rollout-patterns-development`  
Rapid iteration and debugging workflows.

**Scale patterns**: {ref}`training-rollout-patterns-scale`  
Handle large-scale and long-running collection jobs.

