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

import json
from pathlib import Path

def format_grid(grid):
    """Format a grid as a string for display in the prompt."""
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def create_arc_prompt(task_data, task_id):
    prompt = f"You are solving ARC-AGI task {task_id}.\n\n"
    prompt += "Here are the training examples that demonstrate the pattern:\n\n"

    # few shot examples
    for i, example in enumerate(task_data["train"]):
        prompt += f"Example {i+1}:\n"
        prompt += "Input:\n"
        prompt += format_grid(example["input"])
        prompt += "\n\nOutput:\n"
        prompt += format_grid(example["output"])
        prompt += "\n\n"

    test_input = task_data["test"][0]["input"]
    prompt += "Now solve this test case following the same pattern:\n"
    prompt += "Test Input:\n"
    prompt += format_grid(test_input)
    prompt += "\n\nProvide your solution as a 2D array in this exact format: [[row1],[row2],...]"
    prompt += "\nFor example: [[1,2,3],[4,5,6],[7,8,9]]"

    return prompt

def create_dataset():
    arc_data_dir = Path("../../ARC-AGI/data/evaluation")

    Path("data").mkdir(exist_ok=True)

    dataset = []

    for task_file in sorted(arc_data_dir.glob("*.json")):
        task_id = task_file.stem

        with open(task_file) as f:
            task_data = json.load(f)

        prompt = create_arc_prompt(task_data, task_id)

        expected_output = task_data["test"][0]["output"]
        test_input = task_data["test"][0]["input"]

        entry = {
            "responses_create_params": {
                "input": [{"role": "user", "content": prompt}]
            },
            "train": task_data["train"],
            "test_input": test_input,
            "expected_output": expected_output,
            "task_id": task_id
        }

        dataset.append(entry)

    full_output_file = Path("data/arc_agi_eval.jsonl")
    with open(full_output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Created full dataset with {len(dataset)} tasks at {full_output_file}")

    example_output_file = Path("data/example.jsonl")
    with open(example_output_file, 'w') as f:
        for entry in dataset[:5]:
            f.write(json.dumps(entry) + '\n')

    print(f"Created example dataset with 5 tasks at {example_output_file}")

if __name__ == "__main__":
    create_dataset()