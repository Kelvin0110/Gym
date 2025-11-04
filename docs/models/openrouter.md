(models-openrouter)=

# OpenRouter API

Configure and use OpenRouter for unified access to multiple model providers through a single API.

---

## Overview

OpenRouter provides a unified API gateway to access models from multiple providers including OpenAI, Anthropic, Google, and others. NeMo Gym can integrate with OpenRouter to enable flexible model selection and cost optimization across providers.

**Key benefits**:
- Unified API across multiple model providers
- Single integration for many models
- Cost comparison and optimization
- Provider fallback and redundancy
- Simplified billing and management

---

## Prerequisites

Before configuring OpenRouter:

- [ ] OpenRouter account
- [ ] OpenRouter API key
- [ ] Model selection from OpenRouter catalog

---

## Configuration

### Step 1: Set Environment Variables

Create or update `env.yaml` in the repository root:

```yaml
policy_base_url: https://openrouter.ai/api/v1
policy_api_key: sk-or-your-openrouter-api-key
policy_model_name: openai/gpt-4-turbo
```

### Step 2: Configure Model Server

OpenRouter uses OpenAI-compatible endpoints. Use the OpenAI model configuration with OpenRouter endpoint:

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
- `base_url`: OpenRouter API endpoint (`https://openrouter.ai/api/v1`)
- `api_key`: Your OpenRouter API key
- `model`: OpenRouter model identifier (format: `provider/model-name`)

---

## Usage

### Running the Server

Start NeMo Gym with OpenRouter configuration:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collecting Rollouts

Generate rollouts using OpenRouter:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/openrouter_rollouts.jsonl \
  +limit=10
```

---

## Available Models

OpenRouter provides access to models from multiple providers. Model identifiers use the format `provider/model-name`:

**OpenAI Models**:
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`

**Anthropic Models**:
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`

**Google Models**:
- `google/gemini-pro`
- `google/palm-2`

**Open Models**:
- `meta-llama/llama-3-70b-instruct`
- `mistralai/mixtral-8x7b-instruct`

Refer to [OpenRouter's model catalog](https://openrouter.ai/models) for the complete list, pricing, and availability.

---

## Cost Optimization

### Compare Model Costs

OpenRouter displays per-request costs across providers:
- Compare pricing before selecting models
- Switch between providers based on cost
- Monitor spending through OpenRouter dashboard

### Fallback Configuration

Configure model fallbacks for redundancy:
- Primary model for best quality
- Fallback to cost-effective alternative
- Automatic retry on provider errors

---

## Testing

Verify OpenRouter integration:

```bash
ng_test +entrypoint=responses_api_models/openai_model
```

---

## Troubleshooting

### API Key Issues

**Problem**: Authentication failure

**Solution**:
- Verify OpenRouter API key format starts with `sk-or-`
- Check API key is active in OpenRouter dashboard
- Ensure sufficient credits or payment method configured

### Model Not Available

**Problem**: Model identifier not found

**Solution**:
- Check model identifier format: `provider/model-name`
- Verify model is available in OpenRouter catalog
- Ensure provider access is enabled for your account

### Rate Limiting

**Problem**: Request throttling or rate limits

**Solution**:
- Review OpenRouter rate limits for your plan
- Reduce rollout collection concurrency
- Contact OpenRouter support for higher limits

---

## Next Steps

- Configure a resource server to provide tools and verification
- Set up an agent to orchestrate model and resource interactions
- Experiment with different model providers

:::{seealso}
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [About Models](../about/concepts/core-abstractions.md#models)
- [Collecting Rollouts](../get-started/collecting-rollouts.md)
:::

