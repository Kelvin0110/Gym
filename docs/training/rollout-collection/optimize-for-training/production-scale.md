(training-rollout-optimize-production)=

# Production Scale

Scale rollout generation to millions of samples using NeMo Gym's built-in features and production-ready operational patterns.

:::{card}

**Task**: Generate large-scale training datasets (100K-1M+ rollouts) with monitoring, resilience, and distribution strategies.

^^^

**This guide shows you how to**:

1. Monitor progress with NeMo Gym's built-in metrics
2. Resume interrupted runs using append mode
3. Optimize verification for high throughput
4. Distribute workloads across machines

:::

---

## NeMo Gym Features

Built-in capabilities for production-scale collection.

### Progress Monitoring

Watch real-time throughput during collection:

```
Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
```

**Key metric**: `it/s` = samples per second

### Automatic Metrics

After collection, NeMo Gym displays aggregated metrics:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1
}
```

```{dropdown} How This Works
:icon: gear

```python
metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})
avg_metrics = {k: v / len(rows) for k, v in metrics.items()}
```

Any numeric field from verification is automatically averaged.
```

### Resume Interrupted Runs

NeMo Gym opens output files in **append mode**—you can safely resume:

```bash
# Check how many completed
wc -l rollouts.jsonl  # Output: 5432

# Process remaining tasks
tail -n +5433 input.jsonl > remaining.jsonl
ng_collect_rollouts \
    +input_jsonl_fpath=remaining.jsonl \
    +output_jsonl_fpath=rollouts.jsonl  # Appends automatically
    +limit=4568
```

```{dropdown} Why This Works
:icon: code

```python
with open(config.output_jsonl_fpath, "a") as f:
```

Output file opens in append mode by default.
```

### Parameter Overrides

Override model parameters globally via CLI:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=512
```

```{dropdown} How Overrides Work
:icon: code

```python
row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
```

CLI overrides merge with per-task parameters. CLI takes precedence.
```

---

## Verification Optimization

If verification is slow, it bottlenecks collection.

### Detecting the Bottleneck

```{list-table}
:header-rows: 1
:widths: 50 50

* - Symptom
  - Likely Cause
* - Model responds fast, but `it/s` is slow
  - Verification taking too long
* - High CPU usage during collection
  - Compute-heavy verification
* - Progress bar stalls between samples
  - External API calls in verification
```

### Optimization Patterns

::::{tab-set}

:::{tab-item} Cache Lookups
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_lookup(key):
    return result

def verify(self, task, response):
    data = expensive_lookup(task['key'])
    return compute_reward(data, response)
```
:::

:::{tab-item} Fast Mode
```python
def verify(self, task, response):
    if self.config.get('fast_mode', False):
        return quick_heuristic(response)
    else:
        return precise_verification(response)
```

Use fast mode for training, precise for evaluation.
:::

:::{tab-item} Defer Verification
1. Modify resource server to return placeholder reward
2. Collect at full speed
3. Run separate verification pass later

**Trade-off**: No real-time quality feedback
:::

::::

---

## Operational Patterns

Strategies for large-scale, distributed generation.

### Distribute Across Machines

Split large datasets and process in parallel:

```bash
# Split 100K dataset into 10 chunks
split -l 10000 -d --additional-suffix=.jsonl dataset.jsonl chunk_

# Machine 1:
ng_collect_rollouts \
    +input_jsonl_fpath=chunk_00.jsonl \
    +output_jsonl_fpath=rollouts_00.jsonl

# Machine 2:
ng_collect_rollouts \
    +input_jsonl_fpath=chunk_01.jsonl \
    +output_jsonl_fpath=rollouts_01.jsonl

# Merge results
cat rollouts_*.jsonl > final_rollouts.jsonl
```

### Long-Running Jobs

Use tmux for resilience:

```bash
# Start detachable session
tmux new -s rollout_collection

# Run collection
ng_collect_rollouts \
    +input_jsonl_fpath=million_tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t rollout_collection

# Monitor progress
watch -n 60 'wc -l rollouts.jsonl'
```

### Track Throughput

Compute samples/sec or tokens/sec:

```bash
# Samples per second
time ng_collect_rollouts +limit=1000 ...
# Calculate: 1000 / total_seconds
```

```python
# Tokens per second (more precise)
import json
total_tokens = sum(
    json.loads(line).get('usage', {}).get('total_tokens', 0)
    for line in open('rollouts.jsonl')
)
tokens_per_sec = total_tokens / elapsed_seconds
```

---

## Next Steps

**Tune sampling** → {doc}`../sampling-strategies/index` for temperature and diversity  
**Reference patterns** → {doc}`../collection-patterns/index` for copy-paste commands
