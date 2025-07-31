from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
)


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="")

        class TestSimpleResourcesServer(SimpleResourcesServer):
            def setup_webserver(self):
                pass

        agent = TestSimpleResourcesServer(config=config)
        agent.setup_webserver()
