import json

from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = run(ServerClient.load_from_global_config())
task = server_client.post(
    server_name="my_gpt4p1_simple_chat_agent",
    url_path="/v1/responses",
    json={
        "input": [{"role": "user", "content": "hello"}],
    },
)
result = run(task)
print(json.dumps(result.json()["output"], indent=4))
