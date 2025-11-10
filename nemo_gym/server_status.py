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
from time import time
from typing import List, Optional

import psutil
from omegaconf import OmegaConf
from pydantic import BaseModel

from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME
from nemo_gym.server_utils import ServerStatus


class ServerProcessInfo(BaseModel):
    """Information about a running server process"""

    pid: int
    server_type: str  # "resources_server", "responses_api_model", "responses_api_agent"
    name: str  # e.g. "simple_weather", "policy_model", etc.
    process_name: str  # config path from env var
    host: Optional[str]
    port: Optional[int]
    url: Optional[str]
    uptime_seconds: float
    status: ServerStatus  # "success", "connection_error", etc.
    entrypoint: Optional[str]


def parse_server_info(proc, cmdline: List[str], env: dict) -> Optional[ServerProcessInfo]:
    """Takes process, command line, and env and returns server process information"""
    try:
        process_name = env.get(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        if not process_name:
            return None

        config_dict_yaml = env.get(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
        if not config_dict_yaml:
            return None

        global_config_dict = OmegaConf.create(config_dict_yaml)

        # TODO: implement getting specific server config
        server_config_dict = {}

        top_level_config = global_config_dict[process_name]
        server_type = list(top_level_config.keys())[0]
        server_type_config = top_level_config[server_type]
        server_name = list(server_type_config.keys())[0]

        host = server_config_dict.get("host")
        port = server_config_dict.get("port")

        url = f"http://{host}:{port}" if host and port else None

        entrypoint = None
        for i, arg in enumerate(cmdline):
            if arg == "python" and i + 1 < len(cmdline):
                entrypoint = cmdline[i + 1]
                break

        current_time = time.time()
        create_time = proc.info["create_time"]
        uptime_seconds = current_time - create_time

        return ServerProcessInfo(
            pid=proc.info["pid"],
            server_type=server_type,
            name=server_name,
            process_name=process_name,
            host=host,
            port=port,
            url=url,
            uptime_seconds=uptime_seconds,
            status=ServerStatus.RUNNING,  # TODO: implement status checking
            entrypoint=entrypoint,
        )
    except (KeyError, IndexError, AttributeError, TypeError):
        return None


class StatusCommand:
    """Main class to check server status"""

    def discover_servers(self) -> List[ServerProcessInfo]:
        """Find all running NeMo Gym server processes"""
        servers = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "environ"]):
            try:
                env = proc.info.get("environ", {})
                if not env:
                    continue

                cmdline = proc.info["cmdline"]
                server_info = parse_server_info(proc, cmdline, env)

                if server_info:
                    servers.append(server_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return servers

    def display_status(self, servers: List[ServerProcessInfo]) -> None:
        """Show server info in a table"""
        # TODO: flesh out
        if not servers:
            print("No NeMo Gym servers found running.")
            return
