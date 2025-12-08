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
"""Configuration utility commands."""
import click
from omegaconf import OmegaConf


@click.group()
def config():
    """Configuration utilities."""
    pass


@config.command()
@click.option(
    "--config-paths",
    "-c",
    multiple=True,
    type=click.Path(exists=True),
    help="Paths to YAML configuration files.",
)
def dump(config_paths: tuple):
    """
    Display the resolved Hydra configuration for debugging purposes.

    Example:

        ng config dump -c config1.yaml -c config2.yaml
    """
    from nemo_gym.global_config import get_global_config_dict

    # Get global config
    global_config_dict = get_global_config_dict()

    # Print as YAML
    print(OmegaConf.to_yaml(global_config_dict, resolve=True))


@config.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration file to validate.",
)
def validate(config: str):
    """
    Validate a configuration file.

    Example:

        ng config validate --config my_config.yaml
    """
    import yaml

    try:
        # Try to load and parse the YAML file
        with open(config) as f:
            config_data = yaml.safe_load(f)

        click.secho(f"✓ Configuration file is valid: {config}", fg="green")

        # Show summary
        if isinstance(config_data, dict):
            click.echo(f"\nFound {len(config_data)} top-level keys:")
            for key in config_data.keys():
                click.echo(f"  - {key}")

    except yaml.YAMLError as e:
        click.secho(f"✗ Invalid YAML syntax: {e}", fg="red", err=True)
        raise click.Abort()
    except Exception as e:
        click.secho(f"✗ Error reading file: {e}", fg="red", err=True)
        raise click.Abort()
