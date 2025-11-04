(models-vllm)=

# vLLM - Self-Hosted Open Models

Configure and use vLLM for high-throughput inference with self-hosted open models in NeMo Gym.

---

## Overview

vLLM is a fast and efficient inference engine for large language models. NeMo Gym's vLLM model server enables you to deploy open-source models on your own infrastructure with optimized throughput for rollout collection.

**Key benefits**:

- High-throughput batch inference
- Full control over model deployment
- Support for open-source models
- Cost-effective for large-scale workloads
- Token-level information for training

---

## Prerequisites

Before configuring the vLLM model server:

- [ ] vLLM server running and accessible
- [ ] API endpoint URL
- [ ] API key (if authentication enabled)
- [ ] Model name/identifier

---

## Configuration

### Step 1: Set Environment Variables

Create or update `env.yaml` in the repository root:

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: your-api-key-here
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

### Step 2: Configure Model Server

Use the vLLM model configuration file:

```yaml
# responses_api_models/vllm_model/configs/vllm_model.yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: false
      uses_reasoning_parser: true
```

**Configuration options**:

- `base_url`: vLLM server endpoint (supports single URL or list for load balancing)
- `api_key`: Authentication key for the vLLM server
- `model`: Model identifier for inference requests
- `return_token_id_information`: Set to `true` to include token IDs for training (default: `false`)
- `uses_reasoning_parser`: Enable reasoning token parsing (default: `true`)

---

## Usage

### Running the Server

Start NeMo Gym with the vLLM model configuration:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml, \
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collecting Rollouts

Generate rollouts using the vLLM model:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/vllm_rollouts.jsonl \
  +limit=10
```

---

## Advanced Configuration

### Load Balancing Across Multiple vLLM Servers

Distribute requests across multiple vLLM instances:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      base_url:
        - http://vllm-server-1:8000/v1
        - http://vllm-server-2:8000/v1
        - http://vllm-server-3:8000/v1
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

NeMo Gym automatically distributes requests across the listed endpoints.

### Training Mode Configuration

Enable token-level information for RL training:

```yaml
# responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
policy_model:
  responses_api_models:
    vllm_model:
      return_token_id_information: true
```

This configuration includes token IDs and log probabilities in responses, required for certain RL training algorithms.

---

## Testing

Verify the vLLM model server integration:

```bash
ng_test +entrypoint=responses_api_models/vllm_model
```

---

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to vLLM server

**Solution**:

- Verify vLLM server is running: `curl http://localhost:8000/health`
- Check `base_url` in configuration matches vLLM server address
- Ensure network connectivity and firewall rules allow access

### Model Not Found

**Problem**: Model identifier not recognized

**Solution**:

- Verify model name matches the model loaded in vLLM server
- Check vLLM server logs for available models
- Ensure model is fully loaded before making requests

### Performance Issues

**Problem**: Slow inference or timeouts

**Solution**:

- Enable multiple vLLM servers with load balancing
- Adjust vLLM server parameters (batch size, tensor parallelism)
- Monitor vLLM server resource utilization (GPU memory, compute)

---

## Next Steps

- Configure a resource server to provide tools and verification
- Set up an agent to orchestrate model and resource interactions
- Generate rollouts at scale for RL training

:::{seealso}
- [About Models](../about/concepts/core-abstractions.md#models)
- [Collecting Rollouts](../get-started/collecting-rollouts.md)
:::

