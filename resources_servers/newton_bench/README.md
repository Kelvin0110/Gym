# NewtonBench Resource Server

NeMo Gym environment for [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench), to train and test LLM agents to discover scientific laws through interactive experimentation.


configure `env.yaml` to point to your vLLM server (or openai):
```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B
```


```bash
config_paths="resources_servers/newton_bench/configs/newton_bench.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```


```bash
ng_collect_rollouts \
    +agent_name=newton_bench_simple_agent \
    +input_jsonl_fpath=resources_servers/newton_bench/data/example.jsonl \
    +output_jsonl_fpath=results/newton_bench_output.jsonl \
    +limit=5
```