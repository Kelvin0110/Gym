from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)

import anthropic

print(
    f"This simple chat agent has an extra dependency on the `anthropic` library as a demonstration, found at: {anthropic.__file__}"
)


class SimpleChatAgentConfig(BaseResponsesAPIAgentConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model_name: str


class SimpleChatAgent(SimpleResponsesAPIAgent):
    config: SimpleChatAgentConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        return super().model_post_init(context)

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body.setdefault("model", self.config.openai_model_name)
        return await self._client.responses.create(**body)


if __name__ == "__main__":
    SimpleChatAgent.run_webserver()
