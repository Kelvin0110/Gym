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
"""
New CLI implementation for NeMo Gym.

This module provides a redesigned CLI with a single entry point and organized subcommands.
"""
import json
import os
import platform
import sys

import click
import psutil
from importlib.metadata import version as md_version

from nemo_gym import PARENT_DIR, __version__
from nemo_gym.cli_new.config import config
from nemo_gym.cli_new.data import data
from nemo_gym.cli_new.dataset import dataset
from nemo_gym.cli_new.server import server
from nemo_gym.cli_new.test import test


@click.group()
@click.version_option(version=__version__, package_name="nemo_gym")
def cli():
    """
    NeMo Gym - Build and train LLMs with Reinforcement Learning.

    Use 'ng <command> --help' for more information on a specific command.
    """
    pass


# Register command groups
cli.add_command(server)
cli.add_command(data)
cli.add_command(dataset)
cli.add_command(config)
cli.add_command(test)


@cli.command()
@click.option("--json", "json_format", is_flag=True, help="Output in JSON format for programmatic use.")
def version(json_format: bool):
    """
    Display gym version and system information.

    Example:

        ng version

        ng version --json
    """
    version_info = {
        "nemo_gym": __version__,
        "python": platform.python_version(),
        "python_path": sys.executable,
        "installation_path": str(PARENT_DIR),
    }

    key_deps = [
        "openai",
        "ray",
    ]

    dependencies = {dep: md_version(dep) for dep in key_deps}
    version_info["dependencies"] = dependencies

    # System info
    version_info["system"] = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpus": os.cpu_count(),
    }

    # Memory info
    mem = psutil.virtual_memory()
    version_info["system"]["memory_gb"] = round(mem.total / (1024**3), 2)

    # Output
    if json_format:
        print(json.dumps(version_info))
    else:
        click.echo(f"NeMo Gym v{version_info['nemo_gym']}")
        click.echo(f"Python {version_info['python']} ({version_info['python_path']})")
        click.echo(f"Installation: {version_info['installation_path']}")
        click.echo()
        click.echo("Key Dependencies:")
        for dep, ver in version_info["dependencies"].items():
            click.echo(f"  {dep}: {ver}")
        click.echo()
        click.echo("System:")
        for key, value in version_info["system"].items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
