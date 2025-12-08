# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Server management commands."""
from pathlib import Path

import click

from nemo_gym.cli import RunHelper
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict


@click.group()
def server():
    """Server management commands."""
    pass


@server.command()
@click.option(
    "--config-paths",
    "-c",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Paths to YAML configuration files.",
)
def run(config_paths: tuple):
    """
    Start NeMo Gym servers for agents, models, and resources.

    This command reads configuration from YAML files and starts all configured servers.
    The configuration files should define server instances with their entrypoints and settings.

    Example:

        ng server run -c resources_servers/weather/config.yaml -c responses_api_models/openai/config.yaml
    """
    # Convert tuple to list and create config
    config_list = list(config_paths)

    # Create global config dict parser config
    global_config_dict_parser_config = GlobalConfigDictParserConfig(config_paths=config_list)

    # Get global config
    global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)

    # Start servers using existing RunHelper
    rh = RunHelper()
    rh.start(global_config_dict_parser_config)
    rh.run_forever()


@server.command()
@click.option(
    "--entrypoint",
    "-e",
    required=True,
    help="Entrypoint for the new server (e.g., resources_servers/my_server).",
)
def init(entrypoint: str):
    """
    Initialize a new resources server with template files and directory structure.

    Example:

        ng server init --entrypoint resources_servers/my_server
    """
    # Validate entrypoint format
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or len(entrypoint_path.parts) != 2:
        raise click.BadParameter(
            "Entrypoint must be a relative path with 2 parts (e.g., resources_servers/my_server)"
        )

    from nemo_gym.cli import init_resources_server

    # Call old init function
    init_resources_server()
