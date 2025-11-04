(training-rollout-optimize-production)=

# Production Scale

Monitor metrics, handle interruptions, and deploy production patterns for million-scale generation.

---

## Monitor and Measure

Track metrics to validate improvements and detect regressions.

### Real-Time Monitoring

During collection, watch the progress bar:

```
Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
```

**Key metric**: `it/s` (iterations per second) = samples per second

### Computing Throughput

**Samples per second**:
```bash
# Time your collection
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=benchmark.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=1000

# Calculate: 1000 samples / [total seconds] = samples/sec
```

**Tokens per second** (more precise):
```python
import json

total_tokens = 0
with open('rollouts.jsonl') as f:
    for line in f:
        rollout = json.loads(line)
        # Sum input + output tokens if available
        total_tokens += rollout.get('usage', {}).get('total_tokens', 0)

elapsed_seconds = 300  # From `time` command
print(f"Tokens/sec: {total_tokens / elapsed_seconds:.2f}")
```

### Aggregate Metrics

After collection completes, review automatic metrics:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1
}
```

These aggregate any numeric fields returned by verification.

### Tracking Over Time

Create a simple log for trend analysis:

```bash
# Append metrics to log file
echo "$(date),1000,$elapsed_sec,$samples_per_sec,$avg_reward" >> throughput_log.csv

# Later, analyze trends
column -t -s, throughput_log.csv
```

---

## Production Patterns

Strategies for large-scale generation in production environments.

### Incremental Collection

Append to existing files to resume interrupted jobs:

```bash
# Initial run (may be interrupted)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=large_dataset.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +limit=10000

# If interrupted, check how many completed
wc -l rollouts.jsonl
# Output: 5432 rollouts.jsonl

# Resume from where you left off
tail -n +5433 large_dataset.jsonl > remaining.jsonl
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=remaining.jsonl \
    +output_jsonl_fpath=rollouts.jsonl  # Appends due to 'a' mode
    +limit=4568  # 10000 - 5432
```

**Note**: Output file opens in append mode by default.

### Chunked Processing

Split large datasets for parallel processing across machines:

```bash
# Split 100K dataset into 10 chunks
split -l 10000 -d --additional-suffix=.jsonl huge_dataset.jsonl chunk_

# Distribute chunks to different machines/GPUs
# Machine 1:
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_00.jsonl \
    +output_jsonl_fpath=rollouts_00.jsonl

# Machine 2:
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_01.jsonl \
    +output_jsonl_fpath=rollouts_01.jsonl

# ... continue for all chunks ...

# Merge results
cat rollouts_*.jsonl > final_rollouts.jsonl
```

### Continuous Generation

For long-running jobs, use tmux or screen for resilience:

```bash
# Start detachable session
tmux new -s rollout_collection

# Inside tmux, run collection
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=million_tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Detach: Ctrl+b, then d
# Reattach later: tmux attach -t rollout_collection
```

Monitor with:
```bash
# Check progress
watch -n 60 'wc -l rollouts.jsonl'

# Monitor GPU usage
watch -n 5 nvidia-smi
```

---

## Troubleshooting

Common issues and solutions when optimizing throughput.

### Out of Memory Errors

**Symptoms**: CUDA OOM, process killed, server crashes

**Solutions**:
1. **Reduce parallelism**: Lower `num_samples_in_parallel`
2. **Reduce context length**: Lower `max_model_len` in vLLM config
3. **Reduce GPU memory**: Lower `gpu_memory_utilization` to 0.85-0.90
4. **Use smaller model**: Switch to 8B instead of 70B if acceptable

### Timeouts

**Symptoms**: Requests timing out, partial responses

**Solutions**:
1. **Increase client timeout**: Update `GlobalAIOHTTPAsyncClientConfig` in config
2. **Reduce output length**: Lower `max_output_tokens`
3. **Check verification time**: Profile verification function
4. **Increase model server timeout**: Update vLLM or API timeout settings

### Inconsistent Speeds

**Symptoms**: Throughput varies significantly between runs

**Solutions**:
1. **Check resource contention**: Other processes using GPU/CPU
2. **Monitor thermal throttling**: GPU temperatures causing slowdowns
3. **Check network stability**: For hosted APIs
4. **Use dedicated resources**: Avoid shared infrastructure for benchmarking

### Low GPU Utilization

**Symptoms**: GPU usage <50%, but collection is slow

**Solutions**:
1. **Increase parallelism**: More concurrent requests to fill batches
2. **Check verification bottleneck**: CPU-bound verification limiting throughput
3. **Reduce batch waiting time**: Adjust vLLM batching parameters
4. **Pre-warm model**: Send dummy requests before starting

---

## Next Steps

Now that you are generating rollouts efficiently:

**Tune data characteristics**: {doc}`../sampling-strategies/index`  
Learn how to configure temperature and sampling for different training objectives.

**See complete patterns**: {doc}`../collection-patterns/index`  
Reference guide with copy-paste commands for common scenarios.

**Filter and curate**: {doc}`../../data-quality/index`  
Improve training data quality through systematic filtering.

