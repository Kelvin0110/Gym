(training-rollout-optimize-bottleneck)=

# Identify Your Bottleneck

Before optimizing, determine what limits your throughput.

---

## The Three Common Bottlenecks

::::{tab-set}

:::{tab-item} Model Inference (Most Common)
**Symptoms**:
- GPU utilization near 100%
- CPU idle during collection
- Increasing parallelism barely helps

**When this happens**:
- Local models (vLLM, TensorRT-LLM)
- Large models (70B+)
- Long output sequences

**Solution path**: {doc}`optimize-infrastructure` (Optimize Model Server)
:::

:::{tab-item} Verification Complexity
**Symptoms**:
- Model inference fast, but overall throughput slow
- CPU usage high during collection
- Verification logic is compute-intensive

**When this happens**:
- Code execution verification
- Complex LLM-as-judge verification
- External API calls in verification

**Solution path**: {doc}`optimize-infrastructure` (Reduce Verification Overhead)
:::

:::{tab-item} Network/API Rate Limits
**Symptoms**:
- 429 rate limit errors
- Slow responses from hosted API
- Parallelism capped at low numbers

**When this happens**:
- Hosted APIs (OpenAI, Azure, etc.)
- Self-hosted with rate limiting
- Network bandwidth constraints

**Solution path**: {doc}`tune-parallelism` (respect rate limits)
:::

::::

---

## Quick Diagnostic

Run a small test to identify your bottleneck:

```bash
# Test with low parallelism
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=test_100.jsonl \
    +output_jsonl_fpath=/tmp/test_p5.jsonl \
    +limit=100 \
    +num_samples_in_parallel=5

# Test with high parallelism
time ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=test_100.jsonl \
    +output_jsonl_fpath=/tmp/test_p20.jsonl \
    +limit=100 \
    +num_samples_in_parallel=20
```

**Interpret results**:
- **Time reduced significantly**: Model was bottleneck → {doc}`tune-parallelism`
- **Time similar**: Verification or network is bottleneck → {doc}`optimize-infrastructure`
- **Errors or crashes**: Infrastructure overloaded → {doc}`tune-parallelism` (reduce parallelism)

---

## Next Step

Based on your diagnostic results:

**Model bottleneck** → {doc}`tune-parallelism` to increase concurrency  
**Verification bottleneck** → {doc}`optimize-infrastructure` to optimize verification  
**Network bottleneck** → {doc}`tune-parallelism` to respect rate limits

