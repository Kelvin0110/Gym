- [NeMo-Gym](#nemo-gym)
- [Setup](#setup)
- [Run a Simple Chat Agent](#run-a-simple-chat-agent)
- [Development](#development)

# NeMo-Gym
# Setup
Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Initialize environment
```bash
uv venv --python 3.13
source .venv/bin/activate
```

Install NeMo Gym
```bash
uv pip install -e ".[dev]"
```


# Run a Simple Chat Agent
Create your `env.yaml` file in this directory. This `env.yaml` file is used to store sensitive information like API keys. Copy in the following information and add your OpenAI API key.
```yaml
my_gpt4p1_simple_chat_agent:
  responses_api_agents:
    simple_chat_agent:
      openai_api_key: {your OpenAI API key}
```


Run server. `ng_run` or `nemo_gym_run` stands for `Nemo Gym Run`. Nemo Gym will run the file at the entrypoint you specify and will use the configs you set under config_paths. The config resolution order is earlier config paths < later config paths < env.yaml < command line args.
```bash
ng_run '+config_paths=[responses_api_agents/simple_chat_agent/configs/gpt4p1.yaml]'
```
Take a look at the `responses_api_agents/simple_chat_agent/configs/gpt4p1.yaml` for more information!


Query the server.
```bash
python responses_api_agents/simple_chat_agent/client.py
```


Run the Simple Chat Agent tests. `ng_test` or `nemo_gym_test` stands for `Nemo Gym Test`.
```bash
ng_test +entrypoint=responses_api_agents/simple_chat_agent
```


# Development
Lint
```bash
ruff check --fix
```

Format
```bash
ruff format
```

Run Nemo Gym tests
```bash
pytest --cov="." -n 8 --durations=10
```

View test coverage
```bash
coverage html
```
