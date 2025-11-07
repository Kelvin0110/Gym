(config-multi-server)=

# Multi-Server Configuration

Run multiple resource servers (training environments) simultaneously for diverse training scenarios.

---

## Overview

NeMo Gym supports running multiple resource servers in a single deployment. This enables:
- Training agents across multiple task domains
- Testing agent capabilities with diverse environments
- Production deployments with comprehensive toolsets

**How it works**:
- Each YAML config defines uniquely named servers
- Configs are merged together (later configs override earlier ones)
- Each server maintains isolated configuration scope
- Servers can reference each other by name

---

## Running Multiple Resource Servers

Combine multiple resource servers by listing all configuration files:

```bash
# Single resource server (math only)
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[$config_paths]"

# Multiple resource servers (math + search)
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

**What happens**:
1. NeMo Gym loads each YAML file in order
2. Configs are merged (later files can override earlier ones)
3. All servers start simultaneously
4. Agents can access any resource server by name

---

## Multi-Server Use Cases

### Training with Multiple Environments

Train an agent across diverse task types:

```bash
# Agent that handles coding, math, and search
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

**Training benefits**:
- Single model learns multiple capabilities
- Better generalization across task types
- Efficient multi-task training

---

### Testing Agent Capabilities

Evaluate agent performance across different domains:

```bash
# Collect rollouts across multiple servers
ng_collect_rollouts +agent_name=multi_task_agent \
    +input_jsonl_fpath=data/multi_task_test.jsonl \
    +output_jsonl_fpath=results/multi_task_rollouts.jsonl
```

**Testing benefits**:
- Comprehensive capability assessment
- Identify strengths and weaknesses per domain
- Compare performance across task types

---

### Production Deployment

Deploy comprehensive agent with diverse capabilities:

```bash
# Production config with multiple resource servers
config_paths="$MODEL_CONFIG,$MATH_CONFIG,$SEARCH_CONFIG,$WEATHER_CONFIG,$CODING_CONFIG"
ng_run "+config_paths=[$config_paths]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

**Production benefits**:
- Single deployment handles multiple use cases
- Consistent infrastructure across domains
- Easier maintenance and monitoring

---

## Configuration Patterns

### Pattern 1: Shared Model, Multiple Resources

One model server shared across multiple resource servers:

```yaml
# In model_config.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      model_name: gpt-4o-2024-11-20

# In math_config.yaml
math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # References shared model

# In search_config.yaml
search_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # Same shared model
```

**Use when**: Cost-effective deployment, consistent model across tasks

---

### Pattern 2: Specialized Models per Domain

Different models optimized for different resource servers:

```yaml
# Math-optimized model
math_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      
# Coding-optimized model
coding_model:
  responses_api_models:
    vllm_model:
      model_name: codellama-34b

math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: math_model

coding_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: coding_model
```

**Use when**: Domain-specific model optimization, testing model performance

---

### Pattern 3: Policy and Judge Models

Separate models for generation and evaluation:

```yaml
# Generation model
policy_model:
  responses_api_models:
    vllm_model:
      model_name: training-model

# Evaluation model
judge_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20

math_resources_server:
  resources_servers:
    library_judge_math:
      judge_model_server:
        type: responses_api_models
        name: judge_model

math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model
      resources_server:
        type: resources_servers
        name: math_resources_server
```

**Use when**: Training with verification, separate policy and critic models

See {doc}`../../tutorials/separate-policy-and-judge-models` for complete guide.

---

## Data Preparation with Multiple Servers

The same multi-server pattern applies to data preparation:

```bash
# Prepare data across multiple resource servers
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/multi_task \
    +mode=train_preparation
```

**Result**: Combined training dataset with samples from all resource servers.

---

## Training Framework Integration

Use the same config for rollout collection and training framework:

```bash
# 1. Collect rollouts with multiple servers
config_paths="..."
ng_collect_rollouts "+config_paths=[$config_paths]" \
    +output_jsonl_fpath=data/training_rollouts.jsonl

# 2. Use in NeMo-RL
nemo_rl_train \
    --gym_config_paths "$config_paths" \
    --rollout_data data/training_rollouts.jsonl
```

```{seealso}
The same multi-server pattern works seamlessly with training frameworks. See {doc}`../../training/handoff-to-training` for framework-specific integration.
```

---

## Managing Multiple Configurations

### Environment-Based Selection

```bash
# Development: Limited servers for fast iteration
config_paths="model.yaml,math.yaml"
ng_run "+config_paths=[$config_paths]"

# Staging: Add more servers
config_paths="model.yaml,math.yaml,search.yaml,coding.yaml"
ng_run "+config_paths=[$config_paths]"

# Production: Full suite
config_paths="$PROD_CONFIGS"  # Set via environment variable
ng_run "+config_paths=[$config_paths]" +default_host=0.0.0.0
```

---

### Config File Organization

```
configs/
├── base/
│   ├── models/
│   │   ├── policy_model.yaml
│   │   └── judge_model.yaml
│   └── resources/
│       ├── math.yaml
│       ├── search.yaml
│       └── coding.yaml
├── dev.yaml        # Minimal servers for dev
├── staging.yaml    # More servers for testing
└── prod.yaml       # All servers for production
```

**Usage**:

```bash
# Development
ng_run "+config_paths=[configs/dev.yaml]"

# Production
ng_run "+config_paths=[configs/prod.yaml]" +default_host=0.0.0.0
```

---

## Debugging Multi-Server Configurations

### Verify All Servers Load

```bash
ng_dump_config "+config_paths=[$config_paths]" | grep "entrypoint:"
# Should show one entry per server
```

### Check for Name Conflicts

```bash
ng_dump_config "+config_paths=[$config_paths]" > /tmp/config.yaml
# Search for duplicate top-level keys (server names)
grep "^[a-z_]*:" /tmp/config.yaml | sort | uniq -d
```

### Validate Server References

```bash
ng_dump_config "+config_paths=[$config_paths]" | grep -A 2 "server:"
# Verify all referenced servers exist
```

---

## Performance Considerations

**Resource usage**: Each server runs as a separate process
- Monitor total memory usage with multiple servers
- Scale horizontally if needed (run servers on different machines)

**Port allocation**: Each server needs a unique port
- Use `ng_dump_config` to verify no port conflicts
- Let NeMo Gym auto-assign ports (omit `port:` key)

**Startup time**: More servers = longer startup
- Start servers in order of dependency
- Use health checks to verify all servers are ready

---

## Common Patterns

### Minimal (Single Server)

```bash
config_paths="model.yaml,resources/math.yaml"
```

**Use for**: Development, focused testing, single-task training

---

### Standard (3-5 Servers)

```bash
config_paths="model.yaml,resources/math.yaml,resources/search.yaml,resources/coding.yaml"
```

**Use for**: Multi-task training, comprehensive evaluation

---

### Comprehensive (10+ Servers)

```bash
config_paths="models/*.yaml,resources/*.yaml"
```

**Use for**: Production, benchmark evaluation, model comparison

---

## Best Practices

**Configuration management**:
- Keep server configs in separate files for modularity
- Use base configs for common settings
- Override in environment-specific configs

**Naming conventions**:
- Descriptive server names (e.g., `math_resources_server`, not `server1`)
- Consistent suffixes (`_model`, `_resources_server`, `_agent`)
- Avoid conflicts with reserved keys

**Testing**:
- Test with minimal config first (1-2 servers)
- Gradually add servers to identify issues
- Use `ng_dump_config` to validate before running

**Deployment**:
- Use environment variables for config paths in production
- Separate sensitive values into `env.yaml`
- Version control all config files (except `env.yaml`)

---

## Related

- {doc}`index` - Configuration overview
- {doc}`reference` - Complete configuration reference
- {doc}`debugging` - Debug configuration issues
- {doc}`../../training/resource-servers/index` - Available resource servers

