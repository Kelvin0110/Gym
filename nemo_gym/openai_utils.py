from openai import AsyncOpenAI
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses import Response

from nemo_gym.server_utils import GLOBAL_HTTPX_CLIENT


class NeMoGymResponseCreateParamsNonStreaming(ResponseCreateParamsNonStreaming):
    pass


class NeMoGymResponse(Response):
    pass


class NeMoGymAsyncOpenAI(AsyncOpenAI):
    def __init__(self, **kwargs) -> None:
        # TODO: this setup is take from https://github.com/NVIDIA/NeMo-Skills/blob/80dc78ac758c4cac81c83a43a729e7ca1280857b/nemo_skills/inference/model/base.py#L318
        # However, there may still be a lingering issue regarding saturating at 100 max connections
        kwargs["http_client"] = GLOBAL_HTTPX_CLIENT
        kwargs["timeout"] = None  # Enforce no timeout

        super().__init__(**kwargs)
