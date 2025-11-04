(training-rollout-patterns-scale)=

# Scale Patterns

Proven patterns for large-scale rollout generation: resume interrupted runs, million-scale datasets, continuous collection, and multi-dataset workflows.

---

## Pattern 6.1: Incremental Collection with Resume

**Use Case**: Resume interrupted long-running collection  
**Goal**: Avoid losing progress

### Initial Run (Interrupted)

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=large_dataset.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=10000

# ... process crashes or is interrupted at 5247 samples ...
```

### Check Progress

```bash
completed=$(wc -l < rollouts.jsonl)
echo "Completed $completed rollouts"
# Output: Completed 5247 rollouts
```

### Resume from Where You Left Off

```bash
# Skip completed samples
tail -n +5248 large_dataset.jsonl > remaining.jsonl

# Resume collection (appends to existing file)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=remaining.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=$((10000 - 5247))

# Verify total
wc -l rollouts.jsonl
# Output: 10000 rollouts.jsonl
```

**Note**: Output file opens in append mode by default (`rollout_collection.py:94`).

---

## Pattern 6.2: Million-Scale Generation

**Use Case**: Generate 1M+ rollouts for large-scale training  
**Strategy**: Chunk, parallelize, track progress

### Setup

```bash
mkdir -p outputs/million_scale
cd outputs/million_scale
```

### Chunk Dataset

```bash
# Split 1M dataset into 100 chunks of 10K each
split -l 10000 -d --additional-suffix=.jsonl \
    ../../datasets/million_tasks.jsonl \
    chunk_
```

### Parallel Processing Script

```bash
#!/bin/bash
# Process chunks in parallel on single machine

MAX_PARALLEL=4  # Number of collection jobs to run simultaneously

for chunk_file in chunk_*.jsonl; do
    # Wait if too many jobs running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
    done
    
    # Start collection for this chunk
    chunk_id=$(basename $chunk_file .jsonl)
    (
        echo "Starting $chunk_id..."
        ng_collect_rollouts \
            +agent_name=my_agent \
            +input_jsonl_fpath=$chunk_file \
            +output_jsonl_fpath=rollouts_${chunk_id}.jsonl \
            +num_samples_in_parallel=20
        echo "Completed $chunk_id"
    ) &
done

# Wait for all jobs to complete
wait

echo "All chunks complete!"
```

### Monitor Progress

```bash
# Count completed rollouts
watch -n 30 'wc -l rollouts_*.jsonl | tail -1'

# Check which chunks are done
ls -lh rollouts_chunk_*.jsonl | wc -l
```

### Merge Results

```bash
# Concatenate all rollouts
cat rollouts_chunk_*.jsonl > final_million_rollouts.jsonl

# Verify count
echo "Total rollouts: $(wc -l < final_million_rollouts.jsonl)"

# Compute aggregate metrics
jq -s 'map(.reward) | add/length' final_million_rollouts.jsonl
```

---

## Pattern 6.3: Continuous Collection

**Use Case**: Long-running collection job (hours/days)  
**Goal**: Resilience and monitoring

### Run in Detachable Session

```bash
# Start tmux session
tmux new -session -s rollout_gen

# Inside tmux, run collection
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=huge_dataset.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +num_samples_in_parallel=20

# Detach: Ctrl+b, then d
```

### Monitor from Another Terminal

```bash
# Reattach to session
tmux attach -session rollout_gen

# Or monitor without attaching
watch -n 60 'wc -l rollouts.jsonl'

# GPU monitoring
watch -n 5 nvidia-smi

# System resources
htop
```

### Logging

```bash
# Capture stdout/stderr
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    2>&1 | tee collection.log

# Later, analyze log
grep "Collecting rollouts" collection.log
```

---

## Pattern 6.4: Multi-Dataset Workflow

**Use Case**: Different sampling strategies for different data sources  
**Goal**: Combined training dataset with varied characteristics

### Collect High-Quality Demonstrations

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=human_curated.jsonl \
    +output_jsonl_fpath=train_high_quality.jsonl \
    +responses_create_params.temperature=0.2 \
    +num_samples_in_parallel=20
```

### Collect Exploration Data

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=synthetic_tasks.jsonl \
    +output_jsonl_fpath=train_exploration.jsonl \
    +responses_create_params.temperature=0.6 \
    +num_samples_in_parallel=20
```

### Collect Edge Cases

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=difficult_cases.jsonl \
    +output_jsonl_fpath=train_hard.jsonl \
    +responses_create_params.temperature=0.4 \
    +num_repeats=3 \
    +num_samples_in_parallel=15
```

### Merge with Balanced Sampling

```python
import json
import random

# Load all datasets
datasets = {
    'high_quality': 'train_high_quality.jsonl',
    'exploration': 'train_exploration.jsonl',
    'hard': 'train_hard.jsonl'
}

# Define target distribution
target_mix = {
    'high_quality': 0.6,  # 60% high-quality
    'exploration': 0.3,   # 30% exploration
    'hard': 0.1          # 10% hard cases
}

# Load and sample
all_rollouts = []
for name, fpath in datasets.items():
    with open(fpath) as f:
        rollouts = [json.loads(line) for line in f]
    
    # Sample according to target mix
    sample_size = int(10000 * target_mix[name])
    sampled = random.sample(rollouts, min(sample_size, len(rollouts)))
    all_rollouts.extend(sampled)
    
    print(f"{name}: sampled {len(sampled)}/{len(rollouts)}")

# Shuffle
random.shuffle(all_rollouts)

# Write combined dataset
with open('train_mixed.jsonl', 'w') as f:
    for rollout in all_rollouts:
        f.write(json.dumps(rollout) + '\n')

print(f"Final dataset: {len(all_rollouts)} rollouts")
```

---

## Next Steps

**Training preparation**: {ref}`training-rollout-patterns-training-prep`  
Apply these scale patterns to different training objectives.

**Infrastructure patterns**: {ref}`training-rollout-patterns-infrastructure`  
Choose infrastructure that supports your scale requirements.

**Quick reference**: {ref}`training-rollout-patterns-quick-reference`  
Essential commands at a glance.

