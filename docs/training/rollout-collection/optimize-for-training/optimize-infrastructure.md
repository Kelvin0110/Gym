(training-rollout-optimize-infrastructure)=

# Optimize Infrastructure

Configure model servers and verification logic for maximum performance.

---

## Optimize Model Server

Make your model inference as fast as possible.

### vLLM Configuration

For local vLLM deployments, tune these parameters in your model configuration file:

**Example: `responses_api_models/vllm_model/configs/my_vllm.yaml`**

```yaml
vllm_model:
  model_name: meta-llama/Llama-3.1-70B-Instruct
  
  # GPU Memory
  gpu_memory_utilization: 0.95  # Increase from default 0.9 for training workloads
  
  # Context Length
  max_model_len: 8192  # Reduce if your tasks are shorter (e.g., 4096)
  
  # Batching
  max_num_batched_tokens: 8192  # Match or exceed max_model_len
  
  # Multi-GPU
  tensor_parallel_size: 1  # Set to GPU count for large models
```

**Key optimizations**:

**1. Reduce `max_model_len`**:
- Default: Model's full context (often 32K-128K)
- Training data: Often needs only 4K-8K
- **Impact**: More GPU memory for batching → higher throughput

**2. Increase `gpu_memory_utilization`**:
- Default: 0.9 (conservative)
- Training: 0.95 (aggressive)
- **Impact**: Larger KV cache → more concurrent requests

**3. Tune `max_num_batched_tokens`**:
- Set equal to or higher than `max_model_len`
- **Impact**: Better batching efficiency

**Trade-off**: Higher memory utilization risks OOM errors. Test with your workload.

**Reference**: {doc}`../../../models/vllm/configuration`

### Hosted API Optimization

For OpenAI, Azure OpenAI, or similar hosted services:

**1. Choose the Right Tier**:
```yaml
# Azure OpenAI example
responses_create_params:
  service_tier: "scale"  # Options: auto, default, flex, scale
```

- **Scale tier**: Higher throughput, higher cost
- **Flex tier**: Lower cost, variable latency
- Use scale tier for large training data generation

**2. Reduce Token Usage**:
```bash
ng_collect_rollouts \
    +agent_name=openai_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=512  # Set minimum needed
```

**Impact**: Lower latency, lower cost, higher throughput

**3. Respect Rate Limits**:
- Check your tier's requests-per-minute (RPM) limit
- Set `num_samples_in_parallel` below RPM / 60
- Example: 500 RPM tier → use parallelism ≤ 8

### Quick Wins

Apply these regardless of infrastructure:

**Reduce Output Length**:
```bash
+responses_create_params.max_output_tokens=512  # Down from 2048 default
```

**Use Faster Models**:
- For simple tasks: GPT-4o-mini instead of GPT-4o
- For local: Llama 3.1 8B instead of 70B (if quality acceptable)

**Pre-warm Model Server**:
```bash
# Send a few dummy requests before large collection
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"responses_create_params": {"input": [{"role": "user", "content": "test"}]}}'
```

Ensures model is loaded and caches are warm.

---

## Reduce Verification Overhead

Verification runs synchronously during collection—optimize it to avoid bottlenecks.

### When Verification Becomes a Bottleneck

Watch for these signs:
- Model inference completes quickly, but overall collection is slow
- High CPU usage during collection (for compute-heavy verification)
- Verification calls external APIs or executes code

### Optimization Strategies

**1. Cache Expensive Computations**

If verification involves repeated lookups or calculations:

```python
# In your resource server's verify() function
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_lookup(key):
    # Expensive operation here
    return result

def verify(self, task, response):
    # Use cached function
    data = expensive_lookup(task['some_key'])
    # ... rest of verification
```

**2. Use Approximations for Training Data**

For RL training, perfect verification isn't always necessary:

```python
def verify(self, task, response):
    if self.config.get('fast_mode', False):
        # Quick heuristic check
        return simple_reward_heuristic(response)
    else:
        # Exact verification (for evaluation)
        return precise_but_slow_verification(response)
```

Run collection with `fast_mode=True`, use exact verification for final evaluation.

**3. Defer Verification**

Collect first, verify in batch later:

```bash
# Phase 1: Collect without verification (modify resource server to skip)
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=unverified_rollouts.jsonl

# Phase 2: Batch verification (custom script)
python batch_verify.py \
    --input unverified_rollouts.jsonl \
    --output verified_rollouts.jsonl \
    --parallelism 50  # Can be much higher without model inference
```

**Trade-off**: Lose real-time feedback on agent performance.

---

## Next Step

After optimizing infrastructure:

**Production scale** → {doc}`production-scale` for monitoring, production patterns, and troubleshooting

