(models-nvidia-nim)=

# NVIDIA NIM - Enterprise Model Deployment

Configure and use NVIDIA NIM microservices for enterprise-grade model deployment in NeMo Gym.

---

## Overview

NVIDIA NIM (NVIDIA Inference Microservices) provides production-ready inference for AI models with enterprise features including security, scalability, and support. NeMo Gym integrates with NVIDIA NIM to enable reliable model serving for agent training workflows.

**Key benefits**:
- Enterprise-grade reliability and support
- Production-ready deployment
- Optimized NVIDIA GPU performance
- Security and compliance features
- Scalable infrastructure

---

## Prerequisites

Before configuring NVIDIA NIM:

- [ ] NVIDIA NIM instance deployed and accessible
- [ ] API endpoint URL
- [ ] API key or authentication credentials
- [ ] Model identifier

---

## Configuration

### Step 1: Set Environment Variables

Create or update `env.yaml` in the repository root:

```yaml
policy_base_url: https://your-nim-endpoint.nvidia.com/v1
policy_api_key: your-nvidia-api-key
policy_model_name: meta/llama-3.1-8b-instruct
```

### Step 2: Configure Model Server

NVIDIA NIM uses OpenAI-compatible endpoints. Use the OpenAI model configuration with NIM endpoint:

```yaml
# responses_api_models/openai_model/configs/openai_model.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

**Configuration options**:
- `base_url`: NVIDIA NIM endpoint URL
- `api_key`: NVIDIA API key for authentication
- `model`: NIM model identifier

---

## Usage

### Running the Server

Start NeMo Gym with the NVIDIA NIM configuration:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collecting Rollouts

Generate rollouts using NVIDIA NIM:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/nim_rollouts.jsonl \
  +limit=10
```

---

## Available Models

NVIDIA NIM supports various model families. Refer to the [NVIDIA NIM documentation](https://developer.nvidia.com/nim) for the complete model catalog and availability.

**Common model identifiers**:
- `meta/llama-3.1-8b-instruct`
- `meta/llama-3.1-70b-instruct`
- `mistralai/mixtral-8x7b-instruct-v0.1`

Check your NIM instance for available models and their identifiers.

---

## Enterprise Features

### Security and Compliance

NVIDIA NIM provides enterprise security features:
- Secure API authentication
- Network isolation options
- Audit logging
- Compliance certifications

Configure security settings through your NIM deployment.

### Monitoring and Observability

Monitor NIM performance and usage:
- Request metrics and latency
- Resource utilization
- Error rates and debugging
- Usage tracking

Access monitoring through the NVIDIA NIM management interface.

---

## Testing

Verify the NVIDIA NIM integration:

```bash
ng_test +entrypoint=responses_api_models/openai_model
```

---

## Troubleshooting

### Authentication Issues

**Problem**: API key rejected or unauthorized access

**Solution**:
- Verify API key is valid and active
- Check API key has access to the specified model
- Ensure API key permissions include inference access

### Model Availability

**Problem**: Model not available or not found

**Solution**:
- Verify model identifier matches NIM catalog
- Check model is deployed in your NIM instance
- Confirm model access is enabled for your API key

### Rate Limiting

**Problem**: Requests throttled or rate limited

**Solution**:
- Review rate limit settings for your NIM instance
- Adjust rollout collection concurrency settings
- Contact NVIDIA support to increase rate limits

---

## Next Steps

- Configure a resource server to provide tools and verification
- Set up an agent to orchestrate model and resource interactions
- Generate rollouts at scale for RL training

:::{seealso}
- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim)
- [About Models](../about/concepts/core-abstractions.md#models)
- [Ecosystem Integration](../about/ecosystem.md)
:::

