from nemo_gym.cli import RunConfig


class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="")
