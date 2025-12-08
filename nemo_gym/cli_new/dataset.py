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
"""Dataset registry commands."""
import click


@click.group()
def dataset():
    """Dataset registry operations."""
    pass


@dataset.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="Path to the JSONL file to upload.")
@click.option(
    "--registry",
    required=True,
    type=click.Choice(["gitlab", "huggingface", "hf"], case_sensitive=False),
    help="Target registry (gitlab or huggingface).",
)
@click.option("--dataset-name", required=True, help="Name of the dataset.")
@click.option("--version", help="Version of the dataset (required for GitLab, format: x.x.x).")
@click.option("--resource-config", type=click.Path(exists=True), help="Path to resource server config file (for HuggingFace naming).")
def upload(source: str, registry: str, dataset_name: str, version: str, resource_config: str):
    """
    Upload a JSONL dataset to a registry (GitLab or HuggingFace).

    Example:

        ng dataset upload --source data.jsonl --registry gitlab --dataset-name my_dataset --version 0.0.1
    """
    registry_lower = registry.lower()

    if registry_lower == "gitlab":
        # GitLab upload
        if not version:
            raise click.BadParameter("--version is required for GitLab uploads")

        from nemo_gym.gitlab_utils import upload_jsonl_dataset_cli

        upload_jsonl_dataset_cli()

    elif registry_lower in ("huggingface", "hf"):
        # HuggingFace upload
        from nemo_gym.dataset_orchestrator import upload_jsonl_dataset_to_hf_cli

        upload_jsonl_dataset_to_hf_cli()

    else:
        raise click.BadParameter(f"Unknown registry: {registry}")


@dataset.command()
@click.option("--name", "dataset_name", required=True, help="Name of the dataset to download.")
@click.option(
    "--registry",
    required=True,
    type=click.Choice(["gitlab", "huggingface", "hf"], case_sensitive=False),
    help="Source registry (gitlab or huggingface).",
)
@click.option("--output", required=True, type=click.Path(), help="Where to save the downloaded dataset.")
@click.option("--version", help="Version of the dataset (required for GitLab, format: x.x.x).")
@click.option("--artifact", help="Artifact filename to download (required for GitLab).")
@click.option("--repo-id", help="HuggingFace repository ID (format: 'organization/dataset-name').")
def download(dataset_name: str, registry: str, output: str, version: str, artifact: str, repo_id: str):
    """
    Download a JSONL dataset from a registry (GitLab or HuggingFace).

    Example:

        ng dataset download --name my_dataset --registry gitlab --version 0.0.1 --artifact train.jsonl --output data/train.jsonl
    """
    registry_lower = registry.lower()

    if registry_lower == "gitlab":
        # GitLab download
        if not version or not artifact:
            raise click.BadParameter("--version and --artifact are required for GitLab downloads")

        from nemo_gym.gitlab_utils import download_jsonl_dataset_cli

        download_jsonl_dataset_cli()

    elif registry_lower in ("huggingface", "hf"):
        # HuggingFace download
        if not repo_id:
            raise click.BadParameter("--repo-id is required for HuggingFace downloads")

        from nemo_gym.dataset_orchestrator import download_jsonl_dataset_from_hf_cli

        download_jsonl_dataset_from_hf_cli()

    else:
        raise click.BadParameter(f"Unknown registry: {registry}")


@dataset.command()
@click.option("--name", "dataset_name", required=True, help="Name of the dataset to delete.")
@click.option(
    "--registry",
    required=True,
    type=click.Choice(["gitlab"], case_sensitive=False),
    help="Registry to delete from (currently only GitLab supported).",
)
@click.confirmation_option(prompt="Are you sure you want to delete this dataset?")
def delete(dataset_name: str, registry: str):
    """
    Delete a dataset from GitLab Model Registry (prompts for confirmation).

    Example:

        ng dataset delete --name old_dataset --registry gitlab
    """
    if registry.lower() != "gitlab":
        raise click.BadParameter("Delete is currently only supported for GitLab registry")

    from nemo_gym.dataset_orchestrator import delete_jsonl_dataset_from_gitlab_cli

    delete_jsonl_dataset_from_gitlab_cli()


@dataset.command()
@click.option("--dataset-name", required=True, help="Name of the dataset to migrate.")
@click.option("--from-gitlab", "from_registry", flag_value="gitlab", default=True, help="Source is GitLab (default).")
@click.option("--to-hf", "to_registry", flag_value="huggingface", default=True, help="Destination is HuggingFace (default).")
def migrate(dataset_name: str, from_registry: str, to_registry: str):
    """
    Migrate a dataset from GitLab to HuggingFace (uploads to HF then deletes from GitLab).

    Example:

        ng dataset migrate --dataset-name my_dataset
    """
    if from_registry != "gitlab" or to_registry != "huggingface":
        raise click.BadParameter("Currently only GitLab → HuggingFace migration is supported")

    from nemo_gym.dataset_orchestrator import upload_jsonl_dataset_to_hf_and_delete_gitlab_cli

    upload_jsonl_dataset_to_hf_and_delete_gitlab_cli()
