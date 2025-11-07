(config-debugging)=

# Configuration Debugging

Debug and validate your NeMo Gym configuration to catch errors before running.

---

## Inspect Resolved Configuration

View the fully resolved configuration after all three layers (YAML → env.yaml → CLI) are merged:

```bash
# Basic usage
ng_dump_config "+config_paths=[config.yaml]"

# Grep for specific values
ng_dump_config "+config_paths=[config.yaml]" | grep policy_api_key

# With multiple configs
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_dump_config "+config_paths=[$config_paths]"
```

**Use `ng_dump_config` to**:
- Debug configuration issues before running servers
- Verify variable substitution from `env.yaml` works correctly
- Confirm CLI overrides apply as expected
- Understand the final configuration NeMo Gym sees
- Troubleshoot server startup problems

```{tip}
Run `ng_dump_config` before `ng_run` to catch configuration errors early. It uses the exact same config resolution logic as `ng_run`.
```

---

## Common Debugging Scenarios

### Check env.yaml Variable Resolution

Verify that variables from `env.yaml` are properly substituted:

```bash
# env.yaml contains: policy_api_key: sk-abc123
ng_dump_config "+config_paths=[config.yaml]" | grep api_key
# Should show: openai_api_key: sk-abc123
```

**Common issues**:
- Variable not defined in `env.yaml`
- Typo in variable name (e.g., `${policy_key}` vs `${policy_api_key}`)
- Wrong `env.yaml` file loaded (check `+dotenv_path`)

---

### Verify CLI Overrides Apply

Test that command-line overrides work as expected:

```bash
ng_dump_config "+config_paths=[config.yaml]" +policy_model_name=gpt-4o-mini | grep model_name
# Should show the overridden value: model_name: gpt-4o-mini
```

**Override syntax**:
- Use `+key=value` for top-level keys
- Use `+server.subsection.key=value` for nested keys
- Quote values with spaces: `+key="value with spaces"`

---

### Identify Port Conflicts

Check which ports are assigned to each server:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep port
# Shows all assigned ports across servers
```

**What to look for**:
- Duplicate port numbers (conflict)
- Ports outside allowed range
- Ports conflicting with other services (8000, 5432, etc.)

**Solution**: Explicitly set ports or let NeMo Gym auto-assign:

```yaml
my_server:
  responses_api_models:
    openai_model:
      host: "127.0.0.1"
      port: 8080  # Explicitly set to avoid conflicts
```

---

### Check Server References

Verify that server references point to valid server names:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep -A 2 "server:"
# Shows all server references
```

**Common reference issues**:
- Referencing non-existent server name
- Wrong server type (e.g., `type: responses_api_agents` instead of `responses_api_models`)
- Typo in server name

**Example valid reference**:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # Must match Level 1 name of a model server
```

---

### Validate Dataset Paths

Ensure dataset files exist at specified paths:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep jsonl_fpath
# Lists all dataset file paths

# Then verify files exist
ls resources_servers/multineedle/data/*.jsonl
```

**Dataset path issues**:
- Relative path is wrong (relative to repo root)
- File doesn't exist yet (need to generate or download)
- Typo in filename

---

### Debug Multiple Configuration Files

When using multiple config files, verify the merge order:

```bash
config_paths="base.yaml,prod.yaml,override.yaml"
ng_dump_config "+config_paths=[$config_paths]"
```

**Merge behavior**:
- Files merged left-to-right (later files override earlier ones)
- Each file can add new servers or override existing ones
- CLI args override everything

**Debugging merge conflicts**:

```bash
# Check each config individually
ng_dump_config "+config_paths=[base.yaml]" > /tmp/base.yaml
ng_dump_config "+config_paths=[prod.yaml]" > /tmp/prod.yaml

# Compare to see what changed
diff /tmp/base.yaml /tmp/prod.yaml
```

---

## Configuration Validation Workflow

Follow this workflow to validate configuration before deployment:

```bash
# 1. Dump and review configuration
ng_dump_config "+config_paths=[config.yaml]" > /tmp/resolved_config.yaml

# 2. Check for required variables
grep "policy_api_key" /tmp/resolved_config.yaml
grep "judge_api_key" /tmp/resolved_config.yaml

# 3. Verify all servers are configured
grep "entrypoint:" /tmp/resolved_config.yaml

# 4. Check port assignments
grep "port:" /tmp/resolved_config.yaml

# 5. Validate dataset paths exist
grep "jsonl_fpath:" /tmp/resolved_config.yaml | while read line; do
    path=$(echo "$line" | cut -d':' -f2 | tr -d ' ')
    ls "$path" 2>/dev/null || echo "Missing: $path"
done
```

---

## Troubleshooting Common Errors

### Error: "Server reference not found"

```
ValueError: Server reference {'type': 'responses_api_models', 'name': 'policy_model'} not found
```

**Cause**: Agent references a server that doesn't exist in configuration.

**Solution**: 
1. Run `ng_dump_config` and search for the server name
2. Check spelling and ensure server is defined
3. Verify server is in same config file or earlier in `config_paths`

---

### Error: "API key not set"

```
ValueError: openai_api_key is required but not set
```

**Cause**: Variable reference `${policy_api_key}` not resolved from `env.yaml`.

**Solution**:
1. Check `env.yaml` exists and contains the key
2. Verify variable name matches exactly (case-sensitive)
3. Try: `ng_dump_config "+config_paths=[config.yaml]" | grep api_key`

---

### Error: "Port already in use"

```
OSError: [Errno 48] Address already in use
```

**Cause**: Another service is using the assigned port.

**Solution**:
1. Find which port is conflicting: `ng_dump_config ... | grep port`
2. Change port in config or let NeMo Gym auto-assign (remove `port:` key)
3. Kill conflicting process: `lsof -ti:8000 | xargs kill`

---

### Error: "Dataset file not found"

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train.jsonl'
```

**Cause**: Dataset path doesn't exist.

**Solution**:
1. Check path is relative to repo root: `ls data/train.jsonl`
2. Generate example data or download from GitLab
3. Verify path in config matches actual file location

---

## Advanced Debugging

### Compare Configurations Across Environments

```bash
# Development
ng_dump_config "+config_paths=[config.yaml]" "+dotenv_path=env.dev.yaml" > /tmp/dev.yaml

# Production  
ng_dump_config "+config_paths=[config.yaml]" "+dotenv_path=env.prod.yaml" > /tmp/prod.yaml

# Compare
diff /tmp/dev.yaml /tmp/prod.yaml
```

### Extract Specific Server Configuration

```bash
# Get just the policy model config
ng_dump_config "+config_paths=[config.yaml]" | grep -A 20 "^policy_model:"
```

### Validate Configuration in CI/CD

```bash
# In CI pipeline
ng_dump_config "+config_paths=[${CONFIG_PATH}]" > resolved_config.yaml
python scripts/validate_config.py resolved_config.yaml
```

---

## Related

- {doc}`../configuration/index` - Configuration overview
- {doc}`../configuration/reference` - Complete configuration reference
- {doc}`../operations/index` - Debugging running servers

