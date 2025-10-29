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
import re
import sys
from pathlib import Path

NEWTON_BENCH_PATH = Path(__file__).parent.parent.parent / "NewtonBench"
sys.path.insert(0, str(NEWTON_BENCH_PATH))

from modules.m0_gravity import prompts as gravity_prompts
from modules.m0_gravity.m0_types import ExperimentSystem


def convert_xml_to_openai(prompt: str, is_code_assisted: bool) -> str:
    prompt = re.sub(
        r'use the <run_experiment> tag',
        'call the **run_experiment function**',
        prompt,
        flags=re.IGNORECASE
    )

    xml_format_pattern = r'\*Your Request:\*\s*<run_experiment>.*?</run_experiment>\s*\*System Response:\*.*?<experiment_output>.*?</experiment_output>'
    function_format = '''**How experiments work:**
To run experiments, call the `run_experiment` function. Each function call runs ONE experiment.
To run multiple experiments, make multiple function calls (the system supports parallel calls).

**Example:**
- First call: run_experiment(mass1=1.0, mass2=1.0, distance=1.0)
- Second call: run_experiment(mass1=2.0, mass2=1.0, distance=1.0)
- etc.

**Response:**
Each function call returns the measurement result for that experiment.'''

    prompt = re.sub(xml_format_pattern, function_format, prompt, flags=re.DOTALL)

    prompt = re.sub(r'Provide a JSON array specifying the parameters for one or arbitrarily many experimental sets\.?',
                   'Run experiments by calling the function with the appropriate parameters.',
                   prompt, flags=re.IGNORECASE)

    prompt = re.sub(r'<experiment_output>.*?</experiment_output>', '', prompt, flags=re.DOTALL)

    if not is_code_assisted:
        python_section_pattern = r'\*\*IMPORTANT:.*?through <python> tags\.\*\*.*?(?=\*\*Workflow:|\*\*Final Submission:)'
        prompt = re.sub(python_section_pattern, '', prompt, flags=re.DOTALL)

        python_example_pattern = r'\*\*Examples:\*\*.*?(?=\*\*Workflow:|\*\*Final Submission:)'
        prompt = re.sub(python_example_pattern, '', prompt, flags=re.DOTALL)

        prompt = re.sub(r'\d+\.\s+\*\*Use <python> tags\*\*.*?\n', '', prompt)
        prompt = re.sub(r'.*?<python>.*?\n', '', prompt)
        prompt = re.sub(r'.*?<python_output>.*?\n', '', prompt)

    prompt = re.sub(r'\n\n\n+', '\n\n', prompt)

    return prompt.strip()


def generate_tool_definitions(system: str, is_code_assisted: bool = True):
    base_params = {
        "mass1": {
            "type": "number",
            "description": "Mass of the first object (positive real number)",
        },
        "mass2": {
            "type": "number",
            "description": "Mass of the second object (positive real number)",
        },
        "distance": {
            "type": "number",
            "description": "Distance between the two objects (positive real number)",
        },
    }

    if system in [ExperimentSystem.SIMPLE_SYSTEM, ExperimentSystem.COMPLEX_SYSTEM]:
        base_params.update(
            {
                "initial_velocity": {
                    "type": "number",
                    "description": "Initial velocity (optional)",
                },
                "duration": {"type": "number", "description": "Duration to track (optional)"},
                "time_step": {
                    "type": "number",
                    "description": "Time between measurements (optional)",
                },
            }
        )

    tools = [
        {
            "type": "function",
            "name": "run_experiment",
            "description": "Run a physics experiment to measure gravitational force or motion.",
            "parameters": {
                "type": "object",
                "properties": base_params,
                "required": ["mass1", "mass2", "distance"],
            },
            "strict": True,
        },
    ]

    if is_code_assisted:
        tools.append({
            "type": "function",
            "name": "execute_python",
            "description": "Execute Python code for data analysis. You have access to numpy, scipy, and pandas.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
            "strict": True,
        })

    return tools


def generate_record(
    record_id: int,
    difficulty: str,
    system: str,
    noise_level: float,
    law_version: str,
    is_code_assisted: bool = True
):
    task_prompt = gravity_prompts.get_task_prompt(
        system=system, is_code_assisted=is_code_assisted, noise_level=noise_level
    )

    task_prompt = convert_xml_to_openai(task_prompt, is_code_assisted)

    tools = generate_tool_definitions(system, is_code_assisted=is_code_assisted)

    record = {
        "id": record_id,
        "difficulty": difficulty,
        "system": system,
        "noise_level": noise_level,
        "law_version": law_version,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": task_prompt},
                {
                    "role": "user",
                    "content": "Begin your scientific discovery process. Design experiments, analyze data, and discover the underlying law.",
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
    args = parser.parse_args()

    is_code_assisted = args.code_assisted
    print(f"Generating datasets with code_assisted={is_code_assisted}")

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
                **base_config,
                "law_version": law_version,
                "is_code_assisted": is_code_assisted
            })

    train_configs = []
    for difficulty in ["easy", "medium", "hard"]:
        for system in ["vanilla_equation", "simple_system", "complex_system"]:
            for noise_level in [0.0, 0.0001, 0.001, 0.01]:
                for law_version in ["v0", "v1", "v2"]:
                    train_configs.append({
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
