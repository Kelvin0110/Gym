import json

import asyncio

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput

from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.server_utils import ResourcesServerRef, ServerClient

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
    weather_server: ResourcesServerRef

    openai_base_url: str
    openai_api_key: str
    openai_model_name: str


class SimpleChatAgent(SimpleResponsesAPIAgent):
    config: SimpleChatAgentConfig

    def model_post_init(self, context):
        self._openai_client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        self._server_client = asyncio.run(ServerClient.load_from_global_config())
        return super().model_post_init(context)

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body.setdefault("model", self.config.openai_model_name)

        new_outputs = []
        while True:
            new_body: NeMoGymResponseCreateParamsNonStreaming = body.copy()
            new_body["input"] = body["input"] + new_outputs

            response = await self._openai_client.responses.create(**new_body)

            output = response.output
            new_outputs.extend((o.model_dump() for o in output))
            if output[-1].type != "function_call":
                break

            output_function_call: ResponseFunctionToolCall = output[-1]

            api_response = await self._server_client.post(
                server_name=self.config.weather_server.name,
                url_path=f"/{output_function_call.name}",
                json=json.loads(output_function_call.arguments),
            )

            tool_response = FunctionCallOutput(
                type="function_call_output",
                call_id=output_function_call.call_id,
                output=json.dumps(api_response.json()),
            )
            new_outputs.append(tool_response)

        final_response = response.model_copy()
        final_response.output = new_outputs
        return final_response


if __name__ == "__main__":
    SimpleChatAgent.run_webserver()
