#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys
from pathlib import Path

from resources_servers.newton_bench.schemas import MODULE_REQUEST_CLASSES_MAPPING, get_tool_schema
from resources_servers.newton_bench.prompt_utils import get_physics_prompt


def generate_record(
    record_id: int,
    module_name: str,
    difficulty: str,
    system: str,
    noise_level: float,
    law_version: str,
    is_code_assisted: bool = True
):
    task_prompt = get_physics_prompt(
        module_name=module_name,
        system=system,
        is_code_assisted=is_code_assisted,
        noise_level=noise_level
    )

    tools = [get_tool_schema(module_name)]

    if is_code_assisted:
        tools.append({
            "type": "function",
            "name": "execute_python",
            "description": "Execute Python code for mathematical reasoning, hypothesis testing, and data analysis. Pre-imported libraries: numpy (as np), pandas (as pd), scipy, and math.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        })

    record = {
        "id": record_id,
        "module_name": module_name,
        "difficulty": difficulty,
        "system": system,
        "noise_level": noise_level,
        "law_version": law_version,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": task_prompt},
                {
                    "role": "user",
                    "content": f"Begin your scientific discovery process for the {module_name}. Design experiments, analyze data, and discover the underlying law.",
                },
            ],
            "tools": tools,
            "parallel_tool_calls": True,
        },
    }

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-assisted",
        action="store_true",
        default=False,
        help="Include Python code execution support (default: False)",
    )
    parser.add_argument(
        "--modules",
        type=str,
        help="Comma-separated list of modules to generate (default: all)",
    )
    args = parser.parse_args()

    is_code_assisted = args.code_assisted
    
    if args.modules:
        target_modules = [m.strip() for m in args.modules.split(",")]
    else:
        target_modules = list(MODULE_REQUEST_CLASSES_MAPPING.keys())

    print(f"Generating datasets with code_assisted={is_code_assisted} and target modules={target_modules}")

    base_example_configs = [
        {"difficulty": "easy", "system": "vanilla_equation", "noise_level": 0.0},
        {"difficulty": "easy", "system": "vanilla_equation", "noise_level": 0.0001},
        {"difficulty": "medium", "system": "vanilla_equation", "noise_level": 0.0},
        {"difficulty": "hard", "system": "vanilla_equation", "noise_level": 0.001},
        {"difficulty": "easy", "system": "simple_system", "noise_level": 0.0},
        {"difficulty": "easy", "system": "complex_system", "noise_level": 0.0},
    ]

    example_configs = []
    for base_config in base_example_configs:
        for law_version in ["v0", "v1", "v2"]:
            example_configs.append({
                "module_name": "m0_gravity",
                **base_config,
                "law_version": law_version,
                "is_code_assisted": is_code_assisted
            })

    train_configs = []
    for module_name in target_modules:
        for difficulty in ["easy", "medium", "hard"]:
            for system in ["vanilla_equation", "simple_system", "complex_system"]:
                for noise_level in [0.0, 0.0001, 0.001, 0.01]:
                    for law_version in ["v0", "v1", "v2"]:
                        train_configs.append({
                            "module_name": module_name,
                            "difficulty": difficulty,
                            "system": system,
                            "noise_level": noise_level,
                            "law_version": law_version,
                            "is_code_assisted": is_code_assisted
                        })

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    example_path = output_dir / "example.jsonl"
    print(f"Generating example dataset: {example_path}")
    with open(example_path, "w") as f:
        for idx, config in enumerate(example_configs):
            record = generate_record(idx, **config)
            f.write(json.dumps(record) + "\n")
    print(f"Generated {len(example_configs)} example records")

    train_path = output_dir / "train.jsonl"
    print(f"Generating train dataset: {train_path}")
    with open(train_path, "w") as f:
        for idx, config in enumerate(train_configs):
            record = generate_record(idx, **config)
            f.write(json.dumps(record) + "\n")
    print(f"Generated {len(train_configs)} train records")

    print("\nDataset generation complete!")
    print(f"  Example: {example_path}")
    print(f"  Train: {train_path}")


if __name__ == "__main__":
    main()
