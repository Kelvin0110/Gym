# TextWorld Resources Server

Integrates: https://github.com/microsoft/TextWorld

Unlike reasoining gym, this provides native multi turn environments.


```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```


```bash
ng_run "+config_paths=[resources_servers/textworld/configs/textworld.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```


```bash
ng_collect_rollouts +agent_name=textworld_simple_agent +input_jsonl_fpath=resources_servers/textworld/data/example.jsonl +output_jsonl_fpath=resources_servers/textworld/data/example_rollouts.jsonl +limit=5 +num_repeats=null +num_samples_in_parallel=null
```

```bash
ng_collect_rollouts +agent_name=textworld_simple_agent +input_jsonl_fpath=resources_servers/textworld/data/example.jsonl +output_jsonl_fpath=resources_servers/textworld/data/example_rollouts.jsonl +limit=null +num_repeats=null +num_samples_in_parallel=null
```

```bash
ng_viewer +jsonl_fpath=resources_servers/textworld/data/example_rollouts.jsonl
```


```bash
ng_test +entrypoint=resources_servers/textworld
```

```bash
python resources_servers/textworld/scripts/generate_games.py
python resources_servers/textworld/scripts/create_examples.py
```

