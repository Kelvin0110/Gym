from nemo_gym.server_utils import BaseRunServerConfig, BaseServer, SimpleServer


class BaseResourcesServerConfig(BaseRunServerConfig):
    pass


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig
