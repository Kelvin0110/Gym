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
from asyncio import run

from nemo_gym.server_utils import ServerClient


def test_run_experiment():
    server_client = ServerClient.load_from_global_config()

    task = server_client.post(
        server_name="newton_bench",
        url_path="/run_experiment",
        json={
            "mass1": 100.0,
            "mass2": 50.0,
            "distance": 10.0,
        },
    )
    result = run(task)
    print("Vanilla equation experiment:")
    print(json.dumps(run(result.json()), indent=4))
    print()

# requires server config 'system: simple_system'
def test_run_experiment_simple_system():
    server_client = ServerClient.load_from_global_config()

    task = server_client.post(
        server_name="newton_bench",
        url_path="/run_experiment",
        json={
            "mass1": 100.0,
            "mass2": 50.0,
            "distance": 10.0,
            "initial_velocity": 0.0,
            "duration": 5.0,
            "time_step": 0.5,
        },
    )
    result = run(task)
    print("Simple system experiment:")
    response_data = run(result.json())
    if isinstance(response_data.get("result"), dict):
        print(f"Time points: {len(response_data['result'].get('time', []))}")
        print(f"Sample data: {json.dumps({k: v[:3] if isinstance(v, list) else v for k, v in response_data['result'].items()}, indent=4)}")
    else:
        print(json.dumps(response_data, indent=4))
    print()


if __name__ == "__main__":
    print("Testing NewtonBench resource server\n")
    print("=" * 60)
    test_run_experiment()
    print("=" * 60)
    # test_run_experiment_simple_system() # requires server config 'system: simple_system'
