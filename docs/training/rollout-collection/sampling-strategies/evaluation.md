(training-rollout-sampling-evaluation)=

# Evaluation Sampling Strategy

Configure for benchmarking: reproducible, deterministic evaluation with minimal variance.

:::{card}

**Task**: Generate reproducible evaluation measurements for comparing models and tracking progress over time.

^^^

**This guide shows you how to**:

1. Configure for deterministic, low-variance evaluation
2. Set up reproducible benchmarking with fixed seeds
3. Compare results across models and checkpoints
4. Detect statistically significant improvements

:::

---

## Before You Start

Ensure you have these prerequisites before running evaluation:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers with deterministic inference support
* - **Evaluation objective**
  - Understanding of benchmark metrics and comparison methodology
* - **Benchmark dataset**
  - Fixed evaluation set in JSONL format (100-5K samples)
* - **Baseline results**
  - Previous model's scores for comparison (optional)
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Run Evaluation

Generate reproducible, deterministic evaluation results for benchmarking.

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=eval_benchmark.jsonl \
    +output_jsonl_fpath=eval_model_v1.jsonl \
    +responses_create_params.temperature=<temperature> \
    +responses_create_params.seed=<seed> \
    +responses_create_params.top_p=<top_p> \
    +num_samples_in_parallel=<parallelism> \
    +limit=null
```

**Configuration**: For evaluation, use very low temperature and fixed seed for reproducibility, single samples per task, and low parallelism to minimize variance. Refer to {doc}`parameters` for parameter explanations.

---

## Model Comparison Workflow

### Step 1: Evaluate Multiple Models

```bash
#!/bin/bash
# Compare 3 model versions on same benchmark

BENCHMARK="eval_benchmark.jsonl"
MODELS=("model_v1" "model_v2" "model_v3")
SEED=42

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    
    ng_collect_rollouts \
        +agent_name=$model \
        +input_jsonl_fpath=$BENCHMARK \
        +output_jsonl_fpath=eval_${model}.jsonl \
        +responses_create_params.temperature=0.1 \
        +responses_create_params.seed=$SEED \
        +num_samples_in_parallel=5
    
    # Extract metrics
    avg_reward=$(jq -s 'map(.reward) | add/length' eval_${model}.jsonl)
    accuracy=$(jq -s 'map(select(.accuracy != null)) | map(.accuracy) | add/length' eval_${model}.jsonl)
    
    echo "$model: reward=$avg_reward, accuracy=$accuracy"
done
```

### Step 2: Statistical Comparison

```python
import json
import statistics

def load_eval_results(jsonl_path):
    """Load evaluation rollouts and extract rewards."""
    with open(jsonl_path) as f:
        return [json.loads(line)['reward'] for line in f]

# Load results
v1_rewards = load_eval_results('eval_model_v1.jsonl')
v2_rewards = load_eval_results('eval_model_v2.jsonl')
v3_rewards = load_eval_results('eval_model_v3.jsonl')

# Compare
results = {
    'model_v1': {
        'mean': statistics.mean(v1_rewards),
        'median': statistics.median(v1_rewards),
        'stdev': statistics.stdev(v1_rewards)
    },
    'model_v2': {
        'mean': statistics.mean(v2_rewards),
        'median': statistics.median(v2_rewards),
        'stdev': statistics.stdev(v2_rewards)
    },
    'model_v3': {
        'mean': statistics.mean(v3_rewards),
        'median': statistics.median(v3_rewards),
        'stdev': statistics.stdev(v3_rewards)
    }
}

# Print comparison
for model, metrics in results.items():
    print(f"{model}: μ={metrics['mean']:.3f}, median={metrics['median']:.3f}, σ={metrics['stdev']:.3f}")
```

### Step 3: Significance Testing

```python
from scipy import stats

# Paired t-test (since same test set)
t_stat, p_value = stats.ttest_rel(v2_rewards, v1_rewards)

print(f"Model v2 vs v1: t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
    if statistics.mean(v2_rewards) > statistics.mean(v1_rewards):
        print("Model v2 is significantly better")
else:
    print("No significant difference")
```

---

## Best Practices

### Separate Eval from Training

**Keep evaluation data pristine**:
```bash
# Never train on eval set
eval_set="eval_benchmark.jsonl"  # Hold-out test set
train_set="train_data.jsonl"     # Training data only
```

### Version Control Results

```bash
# Save with model version and timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=eval_benchmark.jsonl \
    +output_jsonl_fpath=eval_results/model_v2_${timestamp}.jsonl \
    +responses_create_params.temperature=0.1 \
    +responses_create_params.seed=42
```

### Run Multiple Times for Confidence

Even with low temperature, measure variance:

```bash
#!/bin/bash
# Run evaluation 3 times with different seeds

for seed in 42 123 456; do
    ng_collect_rollouts \
        +agent_name=my_agent \
        +input_jsonl_fpath=benchmark.jsonl \
        +output_jsonl_fpath=eval_seed${seed}.jsonl \
        +responses_create_params.temperature=0.1 \
        +responses_create_params.seed=$seed \
        +num_samples_in_parallel=5
done

# Compute mean and std across runs
python -c "
import json
import statistics

runs = []
for seed in [42, 123, 456]:
    with open(f'eval_seed{seed}.jsonl') as f:
        rewards = [json.loads(line)['reward'] for line in f]
        runs.append(statistics.mean(rewards))

print(f'Mean: {statistics.mean(runs):.3f} ± {statistics.stdev(runs):.3f}')
"
```

---

## Validation

Verify evaluation reproducibility and consistency.

```{dropdown} Expected Variance
:icon: graph
:color: info

**Target**: <0.02 difference across runs with same seed

~~~bash
# Compare two runs with same seed
diff_rate=$(python -c "
import json
with open('eval_run1.jsonl') as f1, open('eval_run2.jsonl') as f2:
    r1 = [json.loads(line)['reward'] for line in f1]
    r2 = [json.loads(line)['reward'] for line in f2]
    diff = sum(abs(a - b) for a, b in zip(r1, r2)) / len(r1)
    print(f'{diff:.4f}')
")
echo "Average difference: $diff_rate"
~~~

**If variance high**:
- Check if seed actually being used
- Verify no other sources of randomness
- Check for resource contention

```

```{dropdown} Reproducibility Check
:icon: sync
:color: info

~~~python
def check_reproducibility(run1_path, run2_path, tolerance=0.01):
    """Verify two evaluation runs are nearly identical."""
    with open(run1_path) as f1, open(run2_path) as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2), 1):
            r1 = json.loads(line1)
            r2 = json.loads(line2)
            
            if abs(r1['reward'] - r2['reward']) > tolerance:
                print(f"Mismatch at line {i}: {r1['reward']} vs {r2['reward']}")
                return False
    
    print("Runs are reproducible within tolerance")
    return True

check_reproducibility('eval_seed42_run1.jsonl', 'eval_seed42_run2.jsonl')
~~~

```

---

## Evaluation Metrics

### Beyond Average Reward

```python
def compute_comprehensive_metrics(eval_jsonl):
    """Extract multiple evaluation metrics."""
    with open(eval_jsonl) as f:
        rollouts = [json.loads(line) for line in f]
    
    rewards = [r['reward'] for r in rollouts]
    
    return {
        # Central tendency
        'mean_reward': statistics.mean(rewards),
        'median_reward': statistics.median(rewards),
        
        # Spread
        'std_reward': statistics.stdev(rewards),
        'min_reward': min(rewards),
        'max_reward': max(rewards),
        
        # Success rates at different thresholds
        'perfect_rate': sum(1 for r in rewards if r == 1.0) / len(rewards),
        'success_rate_0.8': sum(1 for r in rewards if r >= 0.8) / len(rewards),
        'failure_rate': sum(1 for r in rewards if r < 0.3) / len(rewards),
        
        # Task coverage
        'num_tasks': len(rollouts)
    }

metrics = compute_comprehensive_metrics('eval_model_v2.jsonl')
print(json.dumps(metrics, indent=2))
```
