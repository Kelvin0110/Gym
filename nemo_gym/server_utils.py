from typing import Any, List, Type

from abc import abstractmethod

from os import getenv

from pathlib import Path


from threading import Thread

from socket import socket

import json

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

from pydantic import BaseModel, TypeAdapter, ConfigDict

from httpx import Limits, AsyncClient, AsyncHTTPTransport, Response
from httpx._types import (
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
)

from fastapi import FastAPI

import uvicorn

from nemo_gym import PARENT_DIR


"""
We create a single global httpx client as recommended by https://www.python-httpx.org/async/
```
In order to get the most benefit from connection pooling, make sure you're not instantiating multiple client instances - for example by using async with inside a "hot loop". This can be achieved either by having a single scoped client that's passed throughout wherever it's needed, or by having a single global client instance.
```

In principle, we use no timeout since various api or model calls may take an indefinite amount of time. Right now, we have no timeout, even for connection errors which may be problematic. We may want to revisit more granular httpx.Timeout later on.

Eventually, we may also want to parameterize the max connections. For now, we set the max connections to just some very large number.
"""
GLOBAL_HTTPX_CLIENT = AsyncClient(
    limits=Limits(max_keepalive_connections=1500, max_connections=1500),
    transport=AsyncHTTPTransport(retries=3),
    timeout=None,
)


_GLOBAL_CONFIG_DICT = None
NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME = "NEMO_GYM_CONFIG_DICT"
NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME = "NEMO_GYM_CONFIG_PATH"
CONFIG_PATHS_KEY_NAME = "config_paths"
ENTRYPOINT_KEY_NAME = "entrypoint"
DEFAULT_HOST_KEY_NAME = "default_host"
HEAD_SERVER_KEY_NAME = "head_server"
NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = [
    CONFIG_PATHS_KEY_NAME,
    ENTRYPOINT_KEY_NAME,
    DEFAULT_HOST_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
]

DEFAULT_HEAD_SERVER_PORT = 11000


def get_global_config_dict() -> DictConfig:
    global _GLOBAL_CONFIG_DICT
    if _GLOBAL_CONFIG_DICT is not None:
        return _GLOBAL_CONFIG_DICT

    nemo_gym_config_dict_str_from_env = getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
    if nemo_gym_config_dict_str_from_env:
        global_config_dict = OmegaConf.create(nemo_gym_config_dict_str_from_env)

        _GLOBAL_CONFIG_DICT = global_config_dict

        return global_config_dict

    # This function is just to get the config object out of the hydra main call.
    # Need a closure. We simply use an outer ref of a list
    config_list = []

    @hydra.main(config_path=None, version_base=None)
    def inner_hydra_wrapper(cfg: DictConfig) -> DictConfig:
        config_list.append(cfg)

    inner_hydra_wrapper()

    global_config_dict: DictConfig = config_list[0]

    ta = TypeAdapter(List[str])
    config_paths = global_config_dict.get(CONFIG_PATHS_KEY_NAME) or []
    config_paths = ta.validate_python(config_paths)

    dotenv_path = Path(PARENT_DIR) / "env.yaml"
    if dotenv_path.exists():
        # We append here since OmegaConf.merge merges left to right, with later configs overriding previous configs.
        config_paths.append(dotenv_path)

    if config_paths:
        extra_configs: List[DictConfig] = []
        for config_path in config_paths:
            config_path = Path(config_path)
            # Assume relative to the parent dir
            if not config_path.is_absolute():
                config_path = PARENT_DIR / config_path

            extra_config = OmegaConf.load(config_path)
            extra_configs.append(extra_config)

        # global_config_dict is the last config arg here since we want command line args to override everything else.
        global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)

    default_host = global_config_dict.get(DEFAULT_HOST_KEY_NAME) or "127.0.0.1"
    for key, server_type_config_dict in global_config_dict.items():
        if key in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
            continue

        server_type_config_dict: DictConfig
        for server_config_dict in server_type_config_dict.values():
            server_config_dict: DictConfig

            for server_instance_config_dict in server_config_dict.values():
                server_instance_config_dict: DictConfig

                # Populate the host and port values if they are not present in the config.
                with open_dict(server_instance_config_dict):
                    if not server_instance_config_dict.get("host"):
                        server_instance_config_dict["host"] = default_host
                    if not server_instance_config_dict.get("port"):
                        server_instance_config_dict["port"] = find_open_port()

    if not global_config_dict.get(HEAD_SERVER_KEY_NAME):
        with open_dict(global_config_dict):
            global_config_dict[HEAD_SERVER_KEY_NAME] = {
                "host": default_host,
                "port": DEFAULT_HEAD_SERVER_PORT,
            }

    _GLOBAL_CONFIG_DICT = global_config_dict

    return global_config_dict


def get_first_server_config_dict(
    global_config_dict: DictConfig, top_level_path: str
) -> DictConfig:
    # Traverse three levels deep total
    server_config_dict = global_config_dict[top_level_path]
    server_config_dict = list(server_config_dict.values())[0]
    server_config_dict = list(server_config_dict.values())[0]

    return server_config_dict


def find_open_port() -> int:  # pragma: no cover
    with socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


class BaseServerConfig(BaseModel):
    host: str
    port: int


class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str


class ServerClient(BaseModel):
    head_server_config: BaseServerConfig

    global_config_dict: DictConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load_head_server_config(cls) -> BaseServerConfig:
        global_config_dict = get_global_config_dict()
        server_config_dict = global_config_dict[HEAD_SERVER_KEY_NAME]
        config = BaseServerConfig.model_validate(server_config_dict)
        return config

    @classmethod
    async def load_from_global_config(cls) -> "ServerClient":
        head_server_config = cls.load_head_server_config()

        response = await GLOBAL_HTTPX_CLIENT.get(
            f"http://{head_server_config.host}:{head_server_config.port}/global_config_dict_yaml",
        )

        global_config_dict_yaml = response.content.decode()
        global_config_dict = OmegaConf.create(json.loads(global_config_dict_yaml))

        return cls(
            head_server_config=head_server_config, global_config_dict=global_config_dict
        )

    async def get(
        self,
        server_name: str,
        url_path: str,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Parameters
        ----------
        server_name: str
            The name of the server you are trying to call.
        url_path: str
            The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(
            self.global_config_dict, server_name
        )
        return await GLOBAL_HTTPX_CLIENT.get(
            f"http://{server_config_dict.host}:{server_config_dict.port}{url_path}",
            params=params,
            headers=headers,
            **kwargs,
        )

    async def post(
        self,
        server_name: str,
        url_path: str,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Parameters
        ----------
        server_name: str
            The name of the server you are trying to call.
        url_path: str
            The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(
            self.global_config_dict, server_name
        )
        return await GLOBAL_HTTPX_CLIENT.post(
            f"http://{server_config_dict.host}:{server_config_dict.port}{url_path}",
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            **kwargs,
        )


class BaseServer(BaseModel):
    """
    All instances of BaseServer are queryable using ServerClient.
    """

    config: BaseRunServerConfig

    @classmethod
    def load_config_from_global_config(cls) -> "BaseRunServerConfig":
        config_path_str = getenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        global_config_dict = get_global_config_dict()
        server_config_dict = get_first_server_config_dict(
            global_config_dict, config_path_str
        )

        server_config_cls: Type[BaseRunServerConfig] = cls.model_fields[
            "config"
        ].annotation
        server_config = server_config_cls.model_validate(server_config_dict)

        return server_config


class SimpleServer(BaseServer):
    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

    @classmethod
    def run_webserver(cls) -> None:  # pragma: no cover
        server_config = cls.load_config_from_global_config()
        server = cls(config=server_config)

        app = server.setup_webserver()

        uvicorn.run(
            app,
            host=server.config.host,
            port=server.config.port,
            # TODO eventually we want to make this FastAPI server served across multiple processes or workers.
            # Right now this will always use one process.
            # workers=server.config.num_fastapi_workers,
            # We don't have any explicit lifespan logic, so instead of defaulting to "auto"
            # We just turn lifespan off
            lifespan="off",
        )


class HeadServer(SimpleServer):
    config: BaseServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)

        return app

    @classmethod
    def run_webserver(cls) -> None:  # pragma: no cover
        config = ServerClient.load_head_server_config()
        server = cls(config=config)

        app = server.setup_webserver()

        config = uvicorn.Config(
            app,
            host=server.config.host,
            port=server.config.port,
        )
        server = uvicorn.Server(config=config)

        thread = Thread(target=server.run, daemon=True)
        thread.start()

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())
