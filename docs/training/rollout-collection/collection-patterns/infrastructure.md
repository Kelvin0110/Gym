(training-rollout-patterns-infrastructure)=

# Infrastructure Patterns

Proven patterns for different infrastructure configurations: local GPUs, hosted APIs, distributed setups, and cost-optimized cloud deployments.

---

## Pattern 4.1: Local vLLM (Single GPU)

**Use Case**: Maximum throughput from local model server  
**Hardware**: 1x A100/H100 80GB  
**Model**: Llama 3.1 70B or similar

### Server Configuration

`responses_api_models/vllm_model/configs/my_vllm.yaml`:

```yaml
vllm_model:
  model_name: meta-llama/Llama-3.1-70B-Instruct
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_model_len: 8192
  max_num_batched_tokens: 8192
  dtype: auto
```

Start server:
```bash
config_paths="resources_servers/MY_RESOURCE/configs/my_resource.yaml,responses_api_models/vllm_model/configs/my_vllm.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collection Command

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=train.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=1024 \
    +num_samples_in_parallel=20 \
    +limit=null
```

### Expected Throughput

- **70B model**: 15-25 samples/sec
- **8B model**: 40-60 samples/sec
- **GPU utilization**: 95-100%

**Tuning**: Increase `num_samples_in_parallel` to 25-30 if GPU utilization <90%.

---

## Pattern 4.2: Hosted OpenAI API

**Use Case**: Quick prototyping without infrastructure  
**Service**: OpenAI GPT-4o, GPT-4o-mini  
**Considerations**: Rate limits, API costs

### Environment Setup

`env.yaml`:
```yaml
OPENAI_API_KEY: "sk-proj-..."
```

### Collection Command

```bash
ng_collect_rollouts \
    +agent_name=openai_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.model=gpt-4o-mini \
    +responses_create_params.temperature=0.3 \
    +responses_create_params.max_output_tokens=512 \
    +num_samples_in_parallel=8
```

### Rate Limit Management

**Check your tier**:
- Tier 1: 500 RPM (use parallelism ≤ 8)
- Tier 2: 5000 RPM (use parallelism ≤ 80)
- Tier 3+: Higher limits

**Start conservative**:
```bash
# Start with low parallelism
+num_samples_in_parallel=5

# Monitor for 429 errors
# If no errors after 100 samples, increase to 10
# Continue increasing until rate limits hit
```

### Cost Estimation

```python
# Rough cost calculation
num_tasks = 10000
avg_input_tokens = 500
avg_output_tokens = 512

# GPT-4o-mini pricing (as of 2025)
cost_per_1m_input = 0.150
cost_per_1m_output = 0.600

total_input_tokens = num_tasks * avg_input_tokens
total_output_tokens = num_tasks * avg_output_tokens

estimated_cost = (
    (total_input_tokens / 1_000_000 * cost_per_1m_input) +
    (total_output_tokens / 1_000_000 * cost_per_1m_output)
)

print(f"Estimated cost for {num_tasks} rollouts: ${estimated_cost:.2f}")
```

---

## Pattern 4.3: Distributed Generation (Multi-Machine)

**Use Case**: Million+ rollout generation  
**Infrastructure**: N machines with independent model servers  
**Strategy**: Split input, process in parallel, merge outputs

### Step 1: Split Input Dataset

```bash
# Split 100K tasks into 10 chunks of 10K each
split -l 10000 -d --additional-suffix=.jsonl large_dataset.jsonl chunk_
```

Results in: `chunk_00.jsonl`, `chunk_01.jsonl`, ..., `chunk_09.jsonl`

### Step 2: Distribute Chunks

Copy chunks to different machines:
```bash
# From coordination machine
for i in {0..9}; do
    scp chunk_$(printf "%02d" $i).jsonl user@machine$i:/path/to/workspace/
done
```

### Step 3: Run Collection on Each Machine

SSH to each machine and run:

```bash
# On machine 0
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_00.jsonl \
    +output_jsonl_fpath=rollouts_00.jsonl

# On machine 1
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=chunk_01.jsonl \
    +output_jsonl_fpath=rollouts_01.jsonl

# ... and so on for all machines
```

**Or automated**:
```bash
# Parallel SSH execution
for i in {0..9}; do
    ssh user@machine$i "cd /path/to/workspace && \
        ng_collect_rollouts \
        +agent_name=my_agent \
        +input_jsonl_fpath=chunk_$(printf '%02d' $i).jsonl \
        +output_jsonl_fpath=rollouts_$(printf '%02d' $i).jsonl" &
done
wait
```

### Step 4: Collect and Merge Results

```bash
# Download results from all machines
for i in {0..9}; do
    scp user@machine$i:/path/to/workspace/rollouts_$(printf "%02d" $i).jsonl .
done

# Merge into single file
cat rollouts_*.jsonl > final_rollouts.jsonl

# Verify count
echo "Total rollouts: $(wc -l < final_rollouts.jsonl)"
```

---

## Pattern 4.4: Cost-Optimized Cloud API

**Use Case**: Balance cost and speed for cloud APIs  
**Service**: Azure OpenAI, hosted NIM, or similar  
**Goal**: Minimize API costs

### Command

```bash
ng_collect_rollouts \
    +agent_name=azure_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +responses_create_params.max_output_tokens=512 \
    +responses_create_params.temperature=0.2 \
    +num_samples_in_parallel=5 \
    +limit=1000
```

### Cost Optimization Strategies

**1. Reduce output tokens**:
```bash
# Set strict limit based on task needs
+responses_create_params.max_output_tokens=256  # Instead of default 2048
```

**2. Lower temperature reduces retries**:
```bash
# More deterministic = fewer failed/retried requests
+responses_create_params.temperature=0.2
```

**3. Test on subset first**:
```bash
# Run on 100 samples to estimate cost
+limit=100

# Measure actual token usage
jq '.usage.total_tokens' rollouts.jsonl | awk '{s+=$1} END {print "Avg tokens/sample:", s/NR}'

# Extrapolate to full dataset
```

**4. Use smaller models when appropriate**:
```bash
# For simple tasks, use cheaper model
+responses_create_params.model=gpt-4o-mini  # vs gpt-4o
```

---

## Next Steps

**Training preparation**: {ref}`training-rollout-patterns-training-prep`  
Generate datasets for different training objectives.

**Development patterns**: {ref}`training-rollout-patterns-development`  
Rapid iteration and debugging workflows.

**Scale patterns**: {ref}`training-rollout-patterns-scale`  
Handle large-scale and long-running collection jobs.

