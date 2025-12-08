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
"""Test commands."""
from pathlib import Path

import click

from nemo_gym.cli import TestConfig


@click.group()
def test():
    """Test commands for NeMo Gym servers and modules."""
    pass


@test.command()
@click.option(
    "--entrypoint",
    "-e",
    required=True,
    help="Entrypoint for this command. Must be a relative path with 2 parts (e.g., resources_servers/my_server).",
)
@click.option(
    "--validate-data",
    is_flag=True,
    default=False,
    help="Validate the example data (examples, metrics, rollouts, etc) for this server.",
)
def server(entrypoint: str, validate_data: bool):
    """
    Test a specific server module by running its pytest suite and optionally validating example data.

    Example:

        ng test server --entrypoint resources_servers/example_simple_weather

        ng test server --entrypoint resources_servers/example_simple_weather --validate-data
    """
    # Validate entrypoint format
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or len(entrypoint_path.parts) != 2:
        raise click.BadParameter(
            "Entrypoint must be a relative path with 2 parts (e.g., resources_servers/my_server)"
        )

    # Create config dict and validate
    config_dict = {
        "entrypoint": entrypoint,
        "should_validate_data": validate_data,
    }

    test_config = TestConfig.model_validate(config_dict)

    # Run tests using existing test logic
    from nemo_gym.cli import test as old_test

    # Call old test function with the config
    old_test()


@test.command(name="all")
@click.option(
    "--fail-on-mismatch",
    is_flag=True,
    default=False,
    help="Fail if the number of server modules doesn't match the number with tests.",
)
def test_all(fail_on_mismatch: bool):
    """
    Run tests for all server modules in the project.

    Example:

        ng test all

        ng test all --fail-on-mismatch
    """
    from nemo_gym.cli import test_all as old_test_all

    # Call old test_all function
    old_test_all()
