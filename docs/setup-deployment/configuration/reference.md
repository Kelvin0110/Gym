(config-reference)=

# Configuration Reference

Complete anatomy of a NeMo Gym configuration file with all schemas and field definitions.

---

## Three-Level Naming Structure

NeMo Gym configurations use a three-level hierarchy:

```yaml
<unique_server_name>:              # Level 1: Unique identifier at runtime
  <server_type>:                   # Level 2: Type (agents, models, or resources)
    <server_implementation>:       # Level 3: Implementation type
      entrypoint: app.py
      # ... configuration
```

**Level 1: Unique Server Name**

- Must be unique across your entire configuration
- Used by other servers to reference this server
- Example: `library_judge_math`, `policy_model`, `my_agent`

**Level 2: Server Type**

- One of three types:
  - `responses_api_models` - Model inference servers
  - `resources_servers` - Training environment servers
  - `responses_api_agents` - Agent servers
  
**Level 3: Server Implementation**

- Specific implementation of that server type
- Example: `openai_model`, `vllm_model`, `simple_agent`, `library_judge_math`
- Can run multiple instances with different names at Level 1

---

## Complete Configuration Example

```yaml
# Resource Server Configuration
library_judge_math_resources_server:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
      # Server-specific configuration
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        input: []
      should_use_judge: false

# Model Server Configuration  
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_api_key: ${policy_api_key}
      model_name: gpt-4o-2024-11-20
      base_url: https://api.openai.com/v1

# Agent Server Configuration
library_judge_math_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      # Server references
      resources_server:
        type: resources_servers
        name: library_judge_math_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      # Dataset configuration
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/library_judge_math/data/train.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/library_judge_math/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/library_judge_math/data/example.jsonl
        num_repeats: 1
```

---

## Server References

Servers reference each other using the `ServerRef` pattern:

```yaml
server_reference:
  type: <server_type>       # responses_api_models, resources_servers, or responses_api_agents
  name: <unique_name>       # Level 1 unique server name
```

**Example**: Agent referencing model and resource servers:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      resources_server:
        type: resources_servers
        name: my_resources_server    # References Level 1 name
      model_server:
        type: responses_api_models
        name: my_model               # References Level 1 name
```

---

## Common Configuration Fields

### All Servers

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Field
  - Type
  - Description
* - `entrypoint`
  - str
  - Path to server implementation (e.g., `app.py`)
* - `host`
  - str
  - Host address (auto-assigned if not specified)
* - `port`
  - int
  - Port number (auto-assigned if not specified)
```

### Model Servers

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Field
  - Type
  - Description
* - `model_name`
  - str
  - Model identifier (e.g., `gpt-4o-2024-11-20`)
* - `base_url`
  - str
  - API endpoint URL
* - `openai_api_key`
  - str
  - API key (use `${variable}` for env.yaml references)
```

**Example**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_api_key: ${policy_api_key}
      model_name: gpt-4o-2024-11-20
      base_url: https://api.openai.com/v1
```

### Resource Servers

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Field
  - Type
  - Description
* - `domain`
  - str
  - Domain identifier for the resource server
```

Additional fields are server-specific. See individual resource server documentation.

**Example**:

```yaml
math_resources_server:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
      domain: mathematics
      judge_model_server:
        type: responses_api_models
        name: judge_model
      should_use_judge: false
```

### Agent Servers

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Field
  - Type
  - Description
* - `resources_server`
  - ServerRef
  - Reference to resource server
* - `model_server`
  - ServerRef
  - Reference to model server
* - `datasets`
  - list
  - Dataset configurations (optional)
```

**Example**:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets: [...]
```

---

## Dataset Configuration

Datasets are configured under agent servers:

```yaml
datasets:
- name: train                    # Dataset identifier
  type: train                    # Type: train, validation, or example
  jsonl_fpath: path/to/data.jsonl
  num_repeats: 1                 # Number of times to repeat dataset
  start_idx: 0                   # Optional: Start index for slicing
  end_idx: 1000                  # Optional: End index for slicing
  gitlab_identifier:             # Required for train/validation
    dataset_name: dataset_name
    version: 0.0.1
    artifact_fpath: train.jsonl
  license: Apache 2.0            # Required for train/validation
```

### Dataset Types

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Type
  - Requirements
  - Purpose
* - `train`
  - `gitlab_identifier` + `license`
  - Training data (main dataset)
* - `validation`
  - `gitlab_identifier` + `license`
  - Validation data (evaluation during training)
* - `example`
  - None (committed to git)
  - 5 examples for testing and documentation
```

### Dataset Fields

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Field
  - Type
  - Description
* - `name`
  - str
  - Dataset identifier (user-defined)
* - `type`
  - str
  - Dataset type: `train`, `validation`, or `example`
* - `jsonl_fpath`
  - str
  - Path to JSONL file (relative to repo root)
* - `num_repeats`
  - int
  - Number of times to repeat dataset (default: 1)
* - `start_idx`
  - int
  - Optional: Start index for dataset slicing
* - `end_idx`
  - int
  - Optional: End index for dataset slicing
* - `gitlab_identifier`
  - object
  - GitLab dataset registry information (required for train/validation)
* - `license`
  - str
  - Dataset license (required for train/validation)
```

### GitLab Identifier

```yaml
gitlab_identifier:
  dataset_name: bytedtsinghua_dapo17k
  version: 0.0.1
  artifact_fpath: train.jsonl
```

Required for `train` and `validation` datasets. Not required for `example` datasets (which are committed to git).

---

## Global Configuration Options

Reserved top-level keys for global settings:

```yaml
# Global settings (outside server configurations)
config_paths: [...]              # List of YAML config files
default_host: "127.0.0.1"        # Default host for all servers
ray_head_node_address: "ray://..." # Custom Ray cluster address

# Head server configuration
head_server:
  host: "127.0.0.1"
  port: 8000

# Profiling configuration
profiling_enabled: false
profiling_results_dirpath: results/profiling

# HTTP client configuration
global_aiohttp_connector_limit: 102400
global_aiohttp_connector_limit_per_host: 1024
```

### Global Options Reference

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Field
  - Type
  - Description
* - `config_paths`
  - list[str]
  - List of YAML configuration files to merge
* - `default_host`
  - str
  - Default host for all servers (default: `127.0.0.1`)
* - `ray_head_node_address`
  - str
  - Custom Ray cluster address (optional)
* - `head_server.host`
  - str
  - Head server host address
* - `head_server.port`
  - int
  - Head server port (default: 8000)
* - `profiling_enabled`
  - bool
  - Enable performance profiling (default: false)
* - `profiling_results_dirpath`
  - str
  - Directory for profiling results
* - `global_aiohttp_connector_limit`
  - int
  - Max concurrent connections (default: 102400)
* - `global_aiohttp_connector_limit_per_host`
  - int
  - Max connections per host (default: 1024)
```

---

## Configuration Best Practices

### Naming Conventions

**Resource servers**: `<name>_resources_server`

```yaml
math_resources_server:
  resources_servers:
    library_judge_math:
      # ...
```

**Agents**: `<name>_simple_agent` or `<environment>_<name>_agent`

```yaml
math_simple_agent:
  responses_api_agents:
    simple_agent:
      # ...
```

**Models**: `policy_model`, `judge_model`, or descriptive names

```yaml
policy_model:
  responses_api_models:
    openai_model:
      # ...
```

### Structure

- One agent per configuration file (with its dependencies)
- Models and resources can be shared across agents
- Use separate files for different environments (dev, staging, prod)

**Recommended directory structure**:

```
configs/
├── base/
│   ├── models/
│   │   ├── policy_model.yaml
│   │   └── judge_model.yaml
│   └── resources/
│       ├── math.yaml
│       └── search.yaml
├── dev.yaml
├── staging.yaml
└── prod.yaml
```

### Server References

- Always use Level 1 unique names in server references
- Verify references exist before running (use `ng_dump_config`)
- Server references are validated at startup

**Validation**:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep -A 2 "server:"
```

### Datasets

- Example datasets (5 examples) must be committed to git
- Train/validation datasets must specify `gitlab_identifier`
- Use `num_repeats` for data augmentation during training

**Example dataset requirements**:

```
resources_servers/my_server/data/
├── example.jsonl          # 5 examples (committed to git)
├── example_metrics.json   # Metrics from ng_prepare_data
└── example_rollouts.jsonl # Rollouts from ng_collect_rollouts
```

---

## Variable Substitution

Use `${variable}` syntax to reference values from `env.yaml`:

**In config YAML**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_api_key: ${policy_api_key}
      model_name: ${policy_model_name}
```

**In env.yaml**:

```yaml
policy_api_key: sk-your-actual-key
policy_model_name: gpt-4o-2024-11-20
```

**Best practices**:
- Store all secrets in `env.yaml` (never commit)
- Use descriptive variable names
- Document required variables in README
- Provide example `env.yaml.example` file

---

## Configuration Hierarchy

NeMo Gym loads configuration from three layers (lowest to highest priority):

```
1. YAML Files          → Base configuration (structure)
2. env.yaml            → Secrets and environment-specific values
3. Command-Line Args   → Runtime overrides
```

**Example merge**:

```yaml
# base.yaml
policy_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      temperature: 0.7

# env.yaml
policy_api_key: sk-abc123

# Command line
+policy_model.responses_api_models.openai_model.temperature=0.5

# Final result (after merge):
policy_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      openai_api_key: sk-abc123
      temperature: 0.5
```

---

## Validation and Schema

Configuration is validated at startup using Pydantic models. Common validation errors:

**Missing required field**:

```
ValidationError: field required (type=value_error.missing)
```

**Invalid server reference**:

```
ValueError: Server reference not found: {'type': 'responses_api_models', 'name': 'missing_model'}
```

**Invalid dataset type**:

```
ValidationError: value is not a valid enumeration member; permitted: 'train', 'validation', 'example'
```

Use `ng_dump_config` to validate configuration before running:

```bash
ng_dump_config "+config_paths=[config.yaml]"
# If this succeeds, configuration is valid
```

---

## Related

- {doc}`index` - Configuration overview
- {doc}`debugging` - Debug configuration issues
- {doc}`multi-server` - Multi-server patterns
- {doc}`../../about/concepts/configuration-system` - Configuration system concepts

```{seealso}
For live configuration validation, use `ng_dump_config "+config_paths=[...]"` to see the fully resolved configuration as NeMo Gym sees it.
```

