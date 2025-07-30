from os import environ

from pathlib import Path


from subprocess import Popen

import asyncio
from asyncio import Future

import shlex

from omegaconf import OmegaConf

from pydantic import BaseModel

from nemo_gym.server_utils import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    HeadServer,
    get_global_config_dict,
)


def _setup_env_command(dir_path: Path) -> str:  # pragma: no cover
    return f"""cd {dir_path} \\
    && uv venv --allow-existing \\
    && source .venv/bin/activate \\
    && uv pip install -r requirements.txt \\
   """


def _run_command(command: str, working_directory: Path) -> Popen:  # pragma: no cover
    custom_env = environ.copy()
    custom_env["PYTHONPATH"] = (
        f"{working_directory.absolute()}:{custom_env.get('PYTHONPATH', '')}"
    )
    print(f"Executing command:\n{command}\n")
    return Popen(command, shell=True, env=custom_env)


class RunConfig(BaseModel):
    entrypoint: str


def run():  # pragma: no cover
    global_config_dict = get_global_config_dict()

    # Assume Nemo Gym Run is for a single agent.
    escaped_config_dict_yaml_str = shlex.quote(OmegaConf.to_yaml(global_config_dict))

    # We always run the head server in this `run` command.
    HeadServer.run_webserver()

    top_level_paths = [
        k
        for k in global_config_dict.keys()
        if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    ]

    for top_level_path in top_level_paths:
        server_config_dict = global_config_dict[top_level_path]
        first_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[first_key]
        second_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[second_key]

        # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
        entrypoint_fpath = Path(server_config_dict.entrypoint)
        assert not entrypoint_fpath.is_absolute()

        dir_path = Path(first_key, second_key)

        command = f"""{_setup_env_command(dir_path)} \\
    && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
    {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
    python {str(entrypoint_fpath)}"""

        _run_command(command, dir_path)

    async def sleep():
        await Future()

    try:
        asyncio.run(sleep())
    except KeyboardInterrupt:
        pass
    finally:
        print("NeMo Gym finished!")


def test():  # pragma: no cover
    config_dict = get_global_config_dict()
    run_config = RunConfig.model_validate(config_dict)

    # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
    dir_path = Path(run_config.entrypoint)
    assert not dir_path.is_absolute()

    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    command = f"""{_setup_env_command(dir_path)} && pytest -n 8"""
    try:
        _run_command(command, dir_path)
    except KeyboardInterrupt:
        pass
    finally:
        print("NeMo Gym finished!")
