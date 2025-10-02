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

import os
import shutil
from pathlib import Path

import textworld
from textworld import GameOptions


def generate_games(output_dir: str = "resources_servers/textworld/games", num_games: int = 5):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    games_config = [
        {"seed": 1, "nb_rooms": 3, "nb_objects": 5, "quest_length": 3},
        {"seed": 2, "nb_rooms": 4, "nb_objects": 7, "quest_length": 4},
        {"seed": 3, "nb_rooms": 3, "nb_objects": 6, "quest_length": 3},
        {"seed": 4, "nb_rooms": 5, "nb_objects": 8, "quest_length": 5},
        {"seed": 5, "nb_rooms": 4, "nb_objects": 6, "quest_length": 4},
    ]

    generated_games = []

    for i, config in enumerate(games_config):
        print(f"\nGenerating game {i+1}/{num_games}...")

        options = GameOptions()
        options.seeds = config["seed"]
        options.nb_rooms = config["nb_rooms"]
        options.nb_objects = config["nb_objects"]
        options.quest_length = config["quest_length"]

        game_file, game = textworld.make(options)

        target_name = f"game_{i+1}.z8"
        target_path = os.path.join(output_dir, target_name)
        shutil.move(game_file, target_path)

        json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
        if os.path.exists(json_file):
            target_json = target_path.replace(".ulx", ".json").replace(".z8", ".json")
            shutil.move(json_file, target_json)

        print(f"Created {target_name}")
        print(f"  Rooms: {config['nb_rooms']}, Objects: {config['nb_objects']}, Quest length: {config['quest_length']}")
        print(f"  Objective: {game.objective[:100]}...")
        print(f"  Max score: {game.max_score}")

        generated_games.append({
            "file": target_name,
            "objective": game.objective,
            "max_score": game.max_score,
            "walkthrough": game.walkthrough if hasattr(game, "walkthrough") else [],
        })

    print(f"\nSuccessfully generated {len(generated_games)} games in {output_dir}")
    return generated_games


if __name__ == "__main__":
    generate_games()
