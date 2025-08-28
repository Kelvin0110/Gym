from pydantic import BaseModel
import requests
import json
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.search_parsing_utils import box_parser, _extract_last_assistant_text  


class BaseSearchQueryRequest(BaseModel):
    query: str
    topk: int


class OfflineSearchResourcesServerConfig(BaseResourcesServerConfig):
    base_url: str #please spin this up by yourself

class BaseGetSearchQueryResponse(BaseModel):
    search_results: str

class OfflineSearchRunRequest(BaseRunRequest):
    expected_answer: str

class OfflineSearchVerifyRequest(BaseVerifyRequest, OfflineSearchRunRequest):
    pass

class OfflineSearchVerifyResponse(BaseVerifyResponse):
    parsed_option: str

class OfflineSearchResourcesServer(SimpleResourcesServer):
    config: OfflineSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)

        return app
    
    async def search(self, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        url = f"{self.config.base_url}/retrieve"
        
        payload = {
            "queries": [body.query],
            "topk": 10,
            "return_scores": False #FIXME: we keep this as false for now
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            json_str = json.dumps(response.json())
            print(f"offline search results: {json_str}")
            return BaseGetSearchQueryResponse(search_results=json_str)
        except Exception as e:
            return BaseGetSearchQueryResponse(search_results=f"Error: Unexpected error - {str(e)}")
        

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        expected_answer = body.expected_answer
        response_text = _extract_last_assistant_text(body)
        parsed_option = box_parser(response_text)
        if parsed_option == expected_answer:
            reward = 1.0
        else:
            reward = 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward, parsed_option=parsed_option)


if __name__ == "__main__":
    OfflineSearchResourcesServer.run_webserver()
