from abc import abstractmethod

from fastapi import FastAPI, Body

from nemo_gym.server_utils import BaseRunServerConfig, BaseServer, SimpleServer
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class BaseResponsesAPIAgentConfig(BaseRunServerConfig):
    pass


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        app.post("/v1/responses")(self.responses)

        return app

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        pass
