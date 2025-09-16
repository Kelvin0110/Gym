try:
    import aviary  # noqa: F401
except ImportError:
    raise ImportError("Aviary is not installed. Please install it with `pip install fhaviary`")

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
)


class AviaryResourcesServerConfig(BaseResourcesServerConfig):
    pass


class AviarySeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int


class AviarySeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    obs: list[NeMoGymEasyInputMessage]
    tools: list[FunctionToolParam]


class AviaryStepRequest(BaseModel):
    env_id: str
    action: list[NeMoGymResponseFunctionToolCall]


class AviaryStepResponse(BaseModel):
    obs: list[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput]
    reward: float
    done: bool


class AviaryNeMoGymResponse(NeMoGymResponse):
    env_id: str


class AviaryAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: AviaryNeMoGymResponse


class AviaryAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
