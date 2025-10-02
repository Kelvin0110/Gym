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

"""Create datasets for nemo gym. Each example specifies a game file and the initial prompt format for the model."""

import json
from pathlib import Path

import textworld
from textworld import EnvInfos


def create_examples(
    games_dir: str = "resources_servers/textworld/games",
    output_file: str = "resources_servers/textworld/data/example.jsonl",
):
    game_files = sorted(Path(games_dir).glob("*.json"))

    if not game_files:
        raise FileNotFoundError(f"No .json game files found in {games_dir}. Run generate_games.py first.")

    game_files = game_files[:5]

    examples = []

    for game_file in game_files:
        print(f"Creating example for {game_file.name}...")

        game = textworld.Game.load(str(game_file))
        objective = game.objective

        execute_command_tool = {
            "name": "execute_command",
            "type": "function",
            "description": "Execute a text command in the game world. Use natural language commands like 'go north', 'take apple', 'open door', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The text command to execute in the game",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        example = {
            "game_file": game_file.name,
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": f"You are playing a text adventure game.\n\nObjective: {objective}\n\nUse the execute_command tool to interact with the game world. You can move between rooms, pick up objects, examine things, and perform actions to complete your quest.",
                    }
                ],
                "tools": [execute_command_tool],
            },
        }

        examples.append(example)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nCreated {len(examples)} examples in {output_file}")

    print("\nSample example:")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    create_examples()
