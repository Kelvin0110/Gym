from app import SimpleChatAgent, SimpleChatAgentConfig


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleChatAgentConfig(
            host="0.0.0.0",
            port=8080,
            openai_base_url="https://api.openai.com/v1",
            openai_api_key="dummy_key",
            openai_model_name="dummy_model",
        )
        SimpleChatAgent(config=config)
