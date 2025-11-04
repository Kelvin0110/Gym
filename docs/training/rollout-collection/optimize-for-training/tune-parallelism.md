(training-rollout-optimize-parallelism)=

# Tune Parallelism

The `num_samples_in_parallel` parameter controls concurrent requests to your agent server.

---

## How It Works

NeMo Gym uses asyncio with semaphores to manage concurrency:

```python
# Conceptual implementation
semaphore = Semaphore(num_samples_in_parallel)
async with semaphore:
    response = await agent_server.post("/run", task)
    save_rollout(response)
```

This limits in-flight requests while maximizing throughput.

---

## Finding Your Sweet Spot

Use a systematic approach to find optimal parallelism:

**Step 1: Establish Baseline**
```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tuning_dataset.jsonl \
    +output_jsonl_fpath=/tmp/baseline.jsonl \
    +limit=200 \
    +num_samples_in_parallel=5
# Note the time: e.g., 120 seconds = 1.67 samples/sec
```

**Step 2: Double and Measure**
```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tuning_dataset.jsonl \
    +output_jsonl_fpath=/tmp/test_p10.jsonl \
    +limit=200 \
    +num_samples_in_parallel=10
# Note improvement: e.g., 75 seconds = 2.67 samples/sec
```

**Step 3: Continue Until Plateau**
- Keep doubling: 20, 40, 80
- Stop when throughput plateaus or degrades
- Back off 20% from peak for stability

**Step 4: Long-Term Stability**
```bash
# Test with larger sample to ensure stability
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=larger_dataset.jsonl \
    +output_jsonl_fpath=/tmp/stability_test.jsonl \
    +limit=1000 \
    +num_samples_in_parallel=[your_optimal_value]
```

---

## Guidelines by Infrastructure

Starting points for different setups:

```{list-table}
:header-rows: 1
:widths: 40 20 40

* - Infrastructure
  - Starting Value
  - Notes
* - **Local vLLM (single GPU)**
  - 10-15
  - Monitor GPU memory
* - **Local vLLM (multi-GPU)**
  - 20-40
  - Scale with GPU count
* - **Hosted API (OpenAI, Azure)**
  - 5-10
  - Check rate limit tier
* - **NVIDIA NIM**
  - 15-30
  - Depends on instance size
* - **Self-hosted cluster**
  - 30-100
  - Tune based on cluster size
```

**Default behavior**: When `num_samples_in_parallel` is not specified, no limit is applied (processes all samples concurrently). This can overwhelm resources.

---

## Next Step

After tuning parallelism:

**Optimize infrastructure** → {doc}`optimize-infrastructure` for model server and verification tuning  
**Production patterns** → {doc}`production-scale` for monitoring and scale strategies

