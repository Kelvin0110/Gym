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
"""Data collection and preparation commands."""
import click

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


@click.group()
def data():
    """Data collection and preparation commands."""
    pass


@data.command()
@click.option("--agent", required=True, help="The agent to collect rollouts from.")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file path.",
)
@click.option("--output", required=True, type=click.Path(), help="Output JSONL file path.")
@click.option("--limit", type=int, help="Maximum number of examples to load and take from the input dataset.")
@click.option(
    "--num-repeats",
    type=int,
    help="The number of times to repeat each example. Useful for calculating mean@k (e.g., mean@4 or mean@16).",
)
@click.option(
    "--num-samples-in-parallel",
    type=int,
    help="Limit the number of concurrent samples running at once.",
)
def collect(agent: str, input_file: str, output: str, limit: int, num_repeats: int, num_samples_in_parallel: int):
    """
    Perform a batch of rollout collection.

    Example:

        ng data collect --agent simple_weather_simple_agent --input weather_query.jsonl --output weather_rollouts.jsonl --limit 100 --num-repeats 4 --num-samples-in-parallel 10
    """
    import asyncio

    # Build config dict
    config_dict = {
        "agent_name": agent,
        "input_jsonl_fpath": input_file,
        "output_jsonl_fpath": output,
    }

    if limit is not None:
        config_dict["limit"] = limit
    if num_repeats is not None:
        config_dict["num_repeats"] = num_repeats
    if num_samples_in_parallel is not None:
        config_dict["num_samples_in_parallel"] = num_samples_in_parallel

    # Validate config
    config = RolloutCollectionConfig.model_validate(config_dict)

    # Run collection
    rch = RolloutCollectionHelper()
    asyncio.run(rch.run_from_config(config))


@data.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory path where processed datasets and metrics will be saved.",
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["train_preparation", "example_validation"], case_sensitive=False),
    help="Processing mode: 'train_preparation' prepares train/validation datasets, 'example_validation' validates example data for PR submission.",
)
@click.option(
    "--download",
    is_flag=True,
    default=False,
    help="Automatically download missing datasets from remote registries.",
)
@click.option(
    "--config-paths",
    "-c",
    multiple=True,
    type=click.Path(exists=True),
    help="Paths to YAML configuration files.",
)
def prepare(output_dir: str, mode: str, download: bool, config_paths: tuple):
    """
    Prepare and validate training data, generating metrics and statistics for datasets.

    Example:

        ng data prepare --output-dir data/example_multi_step --mode example_validation -c resources_servers/example_multi_step/configs/example_multi_step.yaml -c responses_api_models/openai_model/configs/openai_model.yaml
    """
    from nemo_gym.train_data_utils import TrainDataProcessorConfig, prepare_data

    # Build config dict
    config_dict = {
        "output_dirpath": output_dir,
        "mode": mode,
        "should_download": download,
    }

    # Add config paths if provided
    if config_paths:
        config_dict["config_paths"] = list(config_paths)

    # Validate and run
    prepare_data()


@data.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to a local JSONL file to view.",
)
def view(input_file: str):
    """
    Launch a Gradio interface to view and explore dataset rollouts interactively.

    Example:

        ng data view --input weather_rollouts.jsonl
    """
    from nemo_gym.dataset_viewer import main

    # Call viewer with the input file
    main()
