from app import SimpleChatAgent, SimpleChatAgentConfig, ResourcesServerRef


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleChatAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            weather_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            openai_base_url="https://api.openai.com/v1",
            openai_api_key="dummy_key",
            openai_model_name="dummy_model",
        )
        SimpleChatAgent(config=config)
