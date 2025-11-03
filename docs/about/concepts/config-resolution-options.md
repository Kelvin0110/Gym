# Configuration Resolution - Alternative Layouts

## Option 1: State-at-Each-Step Table

Shows what configuration exists in memory after each step, making the override behavior visible.

---

When you run this command:

```bash
ng_run "+config_paths=[model.yaml]" +policy_model_name=gpt-4o-mini
```

With `model.yaml` containing:
```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

And `env.yaml` containing:
```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-real-key
policy_model_name: gpt-4o-2024-11-20
```

Here's what happens:

```{list-table}
:header-rows: 1
:widths: 15 40 45

* - Step
  - Action
  - Configuration State
* - **1. Parse CLI**
  - Extract command-line arguments
  - `config_paths: ["model.yaml"]`<br>`policy_model_name: "gpt-4o-mini"`
* - **2. Load env.yaml**
  - Load environment secrets
  - `policy_base_url: "https://api.openai.com/v1"`<br>`policy_api_key: "sk-real-key"`<br>`policy_model_name: "gpt-4o-2024-11-20"`
* - **3. Load YAML**
  - Load model.yaml using variables
  - `policy_model:`<br>`  openai_model:`<br>`    openai_base_url: "https://api.openai.com/v1"` (from env.yaml)<br>`    openai_api_key: "sk-real-key"` (from env.yaml)<br>`    openai_model: "gpt-4o-2024-11-20"` (from env.yaml)
* - **4. Merge Layers**
  - Apply priority: YAML < env.yaml < CLI
  - **Final result:**<br>`policy_model_name: "gpt-4o-mini"` ✅ (CLI overrides env.yaml)<br>`policy_api_key: "sk-real-key"` ✅ (from env.yaml)<br>`policy_base_url: "https://api.openai.com/v1"` ✅ (from env.yaml)
* - **5. Validate**
  - Check references, populate defaults
  - Configuration ready ✅
```

**Key insight**: `policy_model_name` appears in both env.yaml (`gpt-4o-2024-11-20`) and CLI (`gpt-4o-mini`). CLI wins because it has highest priority.

**Evidence**: Resolution logic in `nemo_gym/global_config.py:132-201`

---

## Option 2: Layer Stacking Visualization

Shows layers visually stacking and overriding each other.

---

Configuration resolution works like layers stacking on top of each other, with each higher layer overriding values from below:

```
Command Line (Highest Priority)
├─ policy_model_name: gpt-4o-mini  ← Overrides env.yaml
└─ config_paths: [model.yaml]

         ⬇ OVERRIDES ⬇

env.yaml (Middle Priority)
├─ policy_base_url: https://api.openai.com/v1
├─ policy_api_key: sk-real-key
└─ policy_model_name: gpt-4o-2024-11-20  ← Gets overridden by CLI

         ⬇ OVERRIDES ⬇

YAML Files (Foundation)
└─ policy_model:
   └─ responses_api_models:
      └─ openai_model:
         ├─ openai_base_url: ${policy_base_url}  ← Resolved from env.yaml
         ├─ openai_api_key: ${policy_api_key}    ← Resolved from env.yaml
         └─ openai_model: ${policy_model_name}   ← Resolved from CLI (after override)
```

**Resolution steps:**

1. **Parse CLI arguments**: Extract `config_paths` and `policy_model_name=gpt-4o-mini`
2. **Load env.yaml**: Load secrets and environment-specific values
3. **Load YAML files**: Load each file in `config_paths`, resolving `${variables}` from env.yaml
4. **Merge with priority**: `OmegaConf.merge(yaml_files, env_yaml, cli_args)` — last layer wins
5. **Validate**: Check server references exist, populate defaults (hosts, ports)

**Final merged configuration:**

```yaml
policy_model_name: gpt-4o-mini                     # From CLI (overrode env.yaml)
policy_api_key: sk-real-key                        # From env.yaml
policy_base_url: https://api.openai.com/v1         # From env.yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_base_url: https://api.openai.com/v1   # Variable resolved
      openai_api_key: sk-real-key                  # Variable resolved
      openai_model: gpt-4o-mini                    # Variable resolved (with override)
```

**Evidence**: Merge logic in `nemo_gym/global_config.py:199-201`

```python
global_config_dict = OmegaConf.merge(
    *extra_configs,        # YAML files (lowest priority)
    dotenv_extra_config,   # env.yaml (middle priority)
    global_config_dict     # CLI args (highest priority)
)
```

---

## Option 3: Follow-One-Variable Through All Layers

Tracks a single variable through the resolution process to show override behavior clearly.

---

Let's follow what happens to `policy_model_name` as configuration is resolved:

**Setup:**

Command:
```bash
ng_run "+config_paths=[model.yaml]" +policy_model_name=gpt-4o-mini
```

env.yaml:
```yaml
policy_model_name: gpt-4o-2024-11-20
```

model.yaml:
```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_model: ${policy_model_name}
```

### Resolution Walkthrough

::::{tab-set}

:::{tab-item} Step 1: Parse CLI
**What happens**: Hydra extracts command-line arguments

**Configuration created**:
```python
{
  "config_paths": ["model.yaml"],
  "policy_model_name": "gpt-4o-mini"  # ← This will eventually win
}
```

**Status of policy_model_name**: Defined as `gpt-4o-mini`
:::

:::{tab-item} Step 2: Load env.yaml
**What happens**: Load environment-specific secrets

**Configuration created**:
```python
{
  "policy_base_url": "https://api.openai.com/v1",
  "policy_api_key": "sk-real-key",
  "policy_model_name": "gpt-4o-2024-11-20"  # ← Different value!
}
```

**Status of policy_model_name**: Now there are TWO values in separate dictionaries
- CLI says: `gpt-4o-mini`
- env.yaml says: `gpt-4o-2024-11-20`
:::

:::{tab-item} Step 3: Load YAML Files
**What happens**: Load model.yaml and resolve `${policy_model_name}`

**Question**: Which value gets used for the variable reference?

**Answer**: At this point, env.yaml and CLI are merged temporarily to resolve config_paths. So `${policy_model_name}` sees BOTH values, but CLI takes priority.

**Result**:
```python
{
  "policy_model": {
    "responses_api_models": {
      "openai_model": {
        "openai_model": "gpt-4o-mini"  # ← Variable resolved with CLI value
      }
    }
  }
}
```
:::

:::{tab-item} Step 4: Final Merge
**What happens**: All three layers merge with priority

**Code**:
```python
OmegaConf.merge(
    yaml_files,         # Has openai_model: "gpt-4o-mini" (already resolved)
    env_yaml,           # Has policy_model_name: "gpt-4o-2024-11-20"
    cli_args            # Has policy_model_name: "gpt-4o-mini"
)
```

**Result**: CLI's `policy_model_name: "gpt-4o-mini"` wins at the top level

**Final configuration**:
```python
{
  "policy_model_name": "gpt-4o-mini",  # ← CLI value (highest priority)
  "policy_api_key": "sk-real-key",
  "policy_model": {
    "responses_api_models": {
      "openai_model": {
        "openai_model": "gpt-4o-mini"  # ← Already resolved to CLI value
      }
    }
  }
}
```
:::

:::{tab-item} Step 5: Validate
**What happens**: System checks configuration is valid

**Validations**:
- ✅ All server references exist
- ✅ No missing mandatory values
- ✅ Hosts/ports populated

**Result**: Configuration is cached and ready to use
:::

::::

**Key Takeaway**: When a variable appears in multiple layers, the highest-priority layer (CLI) wins. Variable interpolation (`${...}`) happens during YAML loading, so it already sees the merged CLI + env.yaml values.

**Evidence**: Priority logic in `nemo_gym/global_config.py:199-201`

---

## Which option feels clearest?

- **Option 1 (Table)**: Shows state evolution clearly, good for seeing what's in memory
- **Option 2 (Visual Stacking)**: Shows override hierarchy visually, good for understanding priority
- **Option 3 (Follow Variable)**: Shows concrete example of override behavior, good for understanding the mechanics

Let me know which approach resonates most and I can refine it!

