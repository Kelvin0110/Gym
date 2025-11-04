(models-openai)=

# OpenAI Responses API

Configure and use OpenAI models through the Responses API format for rapid prototyping and experimentation.

---

## Overview

OpenAI provides state-of-the-art language models through a managed API service. NeMo Gym integrates with OpenAI's endpoints using the Responses API format, enabling quick experimentation without infrastructure management.

**Key benefits**:
- Rapid prototyping and experimentation
- No infrastructure setup required
- Access to latest OpenAI models
- Managed service with high availability
- Pay-per-use pricing

---

## Prerequisites

Before configuring the OpenAI model:

- [ ] OpenAI API account
- [ ] OpenAI API key
- [ ] Model selection (e.g., `gpt-4`, `gpt-3.5-turbo`)

---

## Configuration

### Step 1: Set Environment Variables

Create or update `env.yaml` in the repository root:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-openai-api-key-here
policy_model_name: gpt-4-turbo
```

### Step 2: Configure Model Server

Use the OpenAI model configuration file:

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
- `base_url`: OpenAI API endpoint (default: `https://api.openai.com/v1`)
- `api_key`: Your OpenAI API key
- `model`: OpenAI model identifier

---

## Usage

### Running the Server

Start NeMo Gym with the OpenAI model configuration:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml, \
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Collecting Rollouts

Generate rollouts using OpenAI models:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/openai_rollouts.jsonl \
  +limit=10
```

---

## Available Models

OpenAI provides several model families with different capabilities and pricing:

**GPT-4 Family**:
- `gpt-4-turbo`: Latest GPT-4 with enhanced speed
- `gpt-4`: Original GPT-4 model
- `gpt-4-32k`: Extended context length

**GPT-3.5 Family**:
- `gpt-3.5-turbo`: Fast and cost-effective
- `gpt-3.5-turbo-16k`: Extended context length

Refer to [OpenAI's model documentation](https://platform.openai.com/docs/models) for the complete list and latest model versions.

---

## LLM-as-a-Judge Configuration

Use OpenAI models as judge models for verification:

```yaml
# responses_api_models/openai_model/configs/openai_judge_model.yaml
judge_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      base_url: ${judge_base_url}
      api_key: ${judge_api_key}
      model: ${judge_model_name}
```

Set judge-specific environment variables in `env.yaml`:

```yaml
judge_base_url: https://api.openai.com/v1
judge_api_key: sk-your-openai-api-key-here
judge_model_name: gpt-4
```

---

## Testing

Verify the OpenAI model server integration:

```bash
ng_test +entrypoint=responses_api_models/openai_model
```

---

## Cost Management

### Estimate Costs

OpenAI charges per token. Estimate rollout collection costs:
- Track total tokens per rollout
- Multiply by number of rollouts
- Apply OpenAI pricing per model

### Optimize Costs

Reduce API usage costs:
- Use `gpt-3.5-turbo` for experimentation
- Limit rollout count during development
- Cache and reuse responses when appropriate
- Use shorter prompts and context

---

## Troubleshooting

### API Key Issues

**Problem**: Authentication failure or invalid API key

**Solution**:
- Verify API key is correct in `env.yaml`
- Check API key is active in OpenAI dashboard
- Ensure API key has sufficient quota/credits

### Rate Limiting

**Problem**: Too many requests error (429)

**Solution**:
- Reduce rollout collection concurrency
- Add retry logic with exponential backoff
- Upgrade OpenAI plan for higher rate limits

### Model Not Available

**Problem**: Model identifier not recognized

**Solution**:
- Verify model name matches OpenAI's model list
- Check model access is enabled for your account
- Use latest model versions from OpenAI documentation

---

## Next Steps

- Configure a resource server to provide tools and verification
- Set up an agent to orchestrate model and resource interactions
- Generate rollouts for agent training

:::{seealso}
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [About Models](../about/concepts/core-abstractions.md#models)
- [Collecting Rollouts](../get-started/collecting-rollouts.md)
:::

