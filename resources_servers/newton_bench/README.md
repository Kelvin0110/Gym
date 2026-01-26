# NewtonBench Resource Server

NeMo Gym environment for [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench), to train and test LLM agents to discover scientific laws through interactive experimentation. The benchmark includes **324 scientific law discovery tasks** across **12 physics domains** (Gravitation, Electrostatics, Magnetostatics, Thermal Conduction, Geometrical Optics, Nuclear Physics, Oscillations, Physical Optics, Acoustics, Elasticity, Statistical Mechanics, Calorimetry).

## Prerequisites

Clone the [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench) repository into the NeMo Gym repository root so that experiment modules can be loaded when running tools (e.g. `run_experiment_*`).

```bash
git clone https://github.com/HKUST-KnowComp/NewtonBench.git
```

Run the command above from the NeMo Gym repository root so that the `NewtonBench` directory exists at the root (e.g. `Gym/NewtonBench/`).

## Dataset Generation

Generate discovery tasks with varying physics domains, equation difficulties, system complexities, and noise levels:

**Generate full training dataset:**
```bash
python resources_servers/newton_bench/generate_dataset.py
```

**Generate training dataset by specific modules and equation difficulties:**
```bash
python resources_servers/newton_bench/generate_dataset.py \
    --modules m0_gravity,m1_coulomb_force \
    --difficulties easy,medium
```

**Generate training dataset by specific system complexities and noise levels:**
```bash
python resources_servers/newton_bench/generate_dataset.py \
    --systems vanilla_equation,complex_system \
    --noise-levels 0.0,0.01
```

**Generate training dataset with python code execution tool enabled:**
```bash
python resources_servers/newton_bench/generate_dataset.py --code-assisted
```

## Rollout Collection

### Configure env.yaml
Configure `env.yaml` to point to your vLLM server (or openai):
```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B
```

### Launch servers
```bash
config_paths="resources_servers/newton_bench/configs/newton_bench.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### Collect rollouts
```bash
ng_collect_rollouts \
    +agent_name=newton_bench_simple_agent \
    +input_jsonl_fpath=resources_servers/newton_bench/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/newton_bench/data/example_rollouts.jsonl \
    +limit=5
```

## Running Tests
```bash
source resources_servers/newton_bench/.venv/bin/activate 
pytest resources_servers/newton_bench/tests/test_app.py
```

## Licensing information
- **Code:** Apache 2.0
- **Data:** Apache 2.0
- **NewtonBench Benchmark:** MIT (Copyright (c) 2025 HKUST-KnowComp)

### Dependencies 
- **nemo_gym:** Apache 2.0
