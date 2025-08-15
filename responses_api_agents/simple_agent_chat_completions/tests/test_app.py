from nemo_gym.server_utils import ServerClient, ModelServerRef

from app import SimpleAgentChatCompletions, SimpleAgentChatCompletionsConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleAgentChatCompletionsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgentChatCompletions(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
