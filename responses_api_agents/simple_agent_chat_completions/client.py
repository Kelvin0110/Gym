import json

from asyncio import run

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="simple_agent_chat_completions",
    url_path="/v1/responses",
    json=NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "what's it like in sf?"},
        ],
    ),
)
result = run(task)
print(json.dumps(result.json()["output"], indent=4))
