(models-azure-openai)=

# Azure OpenAI Responses API

Configure and use Azure OpenAI endpoints for enterprise-grade LLM inference in Azure environments.

---

## Overview

Azure OpenAI Service provides OpenAI models through Microsoft Azure infrastructure, enabling enterprise deployments with Azure's security, compliance, and regional availability. NeMo Gym integrates with Azure OpenAI endpoints for agent training workflows in Azure-native environments.

**Key benefits**:
- Azure infrastructure and compliance
- Regional data residency options
- Enterprise security features
- Integration with Azure services
- Microsoft support and SLAs

---

## Prerequisites

Before configuring Azure OpenAI:

- [ ] Azure subscription with Azure OpenAI access
- [ ] Azure OpenAI resource deployed
- [ ] API endpoint URL
- [ ] API key
- [ ] API version
- [ ] Deployment name (model identifier)

---

## Configuration

### Step 1: Set Environment Variables

Create or update `env.yaml` in the repository root:

```yaml
policy_base_url: https://your-resource.openai.azure.com/v1/azure
policy_api_key: your-azure-api-key
policy_model_name: gpt-4-deployment-name
```

### Step 2: Configure Model Server

Use the Azure OpenAI model configuration file:

```yaml
# responses_api_models/azure_openai_model/configs/azure_openai_model.yaml
policy_model:
  responses_api_models:
    azure_openai_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

**Configuration options**:
- `base_url`: Azure OpenAI endpoint URL
- `api_key`: Azure OpenAI API key
- `model`: Azure deployment name (not the base model name)

---

## Usage

### Running the Server

Start NeMo Gym with Azure OpenAI configuration, specifying the API version:

```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
```

**API version parameter**:
- Set `api-version` to match your Azure OpenAI resource version
- Format: `YYYY-MM-DD` (e.g., `2024-10-21`)
- Check Azure portal for supported API versions

### Collecting Rollouts

Generate rollouts using Azure OpenAI:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/azure_rollouts.jsonl \
  +limit=10
```

---

## Azure Setup

### Create Azure OpenAI Resource

1. Navigate to Azure portal
2. Create an Azure OpenAI resource
3. Deploy a model (e.g., GPT-4, GPT-3.5-Turbo)
4. Note the endpoint URL and API key

### Deployment Names

Azure OpenAI uses custom deployment names. In `env.yaml`, use your deployment name, not the base model name:

```yaml
# Correct: Use your deployment name
policy_model_name: my-gpt-4-deployment

# Incorrect: Do not use base model name
policy_model_name: gpt-4
```

---

## LLM-as-a-Judge Configuration

Use Azure OpenAI as a judge model for verification:

```bash
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml, \
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
```

Generate judge-evaluated rollouts:

```bash
ng_collect_rollouts \
  +agent_name=equivalence_llm_judge_simple_agent \
  +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl \
  +output_jsonl_fpath=results/azure_judge_rollouts.jsonl \
  +limit=5
```

---

## Testing

Verify the Azure OpenAI model server integration:

```bash
ng_test +entrypoint=responses_api_models/azure_openai_model
```

---

## Troubleshooting

### API Version Issues

**Problem**: API version not supported or outdated

**Solution**:
- Check Azure portal for supported API versions
- Update `api-version` parameter to match current version
- Common format: `2024-10-21`

### Endpoint URL Format

**Problem**: Cannot connect to Azure OpenAI endpoint

**Solution**:
- Verify URL format: `https://<resource-name>.openai.azure.com/v1/azure`
- Ensure resource name matches Azure portal
- Check network connectivity to Azure endpoint

### Deployment Not Found

**Problem**: Model deployment name not recognized

**Solution**:
- Verify deployment name in Azure portal
- Use deployment name, not base model name
- Ensure deployment is active and provisioned

### Regional Availability

**Problem**: Model not available in selected region

**Solution**:
- Check Azure OpenAI regional availability
- Deploy resource in region where model is available
- Refer to Azure documentation for model-region matrix

---

## Next Steps

- Configure a resource server to provide tools and verification
- Set up an agent to orchestrate model and resource interactions
- Generate rollouts for agent training

:::{seealso}
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [About Models](../about/concepts/core-abstractions.md#models)
- [Collecting Rollouts](../get-started/collecting-rollouts.md)
:::

