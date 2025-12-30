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

from unittest.mock import MagicMock, AsyncMock
from fastapi import Request
from pytest import fixture, mark, raises

from resources_servers.newton_bench.app import (
    NewtonBenchResourcesServer, 
    NewtonBenchResourcesServerConfig,
    NewtonBenchSeedSessionRequest,
    NewtonBenchRunRequest,
    NewtonBenchVerifyRequest,
    RunExperimentResponse
)
from nemo_gym.server_utils import ServerClient, SESSION_ID_KEY


class TestApp:
    SERVER_NAME = "newton_bench"

    @fixture
    def config(self) -> NewtonBenchResourcesServerConfig:
        return NewtonBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @fixture
    def server(self, config: NewtonBenchResourcesServerConfig) -> NewtonBenchResourcesServer:
        server_mock = MagicMock(spec=ServerClient)
        return NewtonBenchResourcesServer(config=config, server_client=server_mock)

    @fixture
    def mock_request(self) -> Request:
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: "test_session_123"}
        return request

    def test_sanity(self, server: NewtonBenchResourcesServer) -> None:
        assert server is not None

    @mark.asyncio
    async def test_seed_session_success(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0"
        )
        
        response = await server.seed_session(mock_request, seed_request)
        assert response is not None
        
        # Verify session state was stored
        session_id = mock_request.session[SESSION_ID_KEY]
        assert session_id in server.session_metadata
        metadata = server.session_metadata[session_id]
        assert metadata["module_name"] == "m0_gravity"
        assert metadata["difficulty"] == "easy"

    @mark.asyncio
    async def test_run_experiment_success(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        # 1. Seed the session first
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        # 2. Create the handler for m0_gravity
        handler = server._create_module_handler("m0_gravity")
        
        # 3. Prepare run request
        # Note: The actual request body class depends on the module, but for testing we can pass a dict-like object
        # or the actual Pydantic model if we import it. Here we use a simple dict which the handler supports.
        run_body = {
            "mass1": 10.0,
            "mass2": 20.0,
            "distance": 5.0,
            "initial_velocity": 0.0,
            "duration": 10.0,
            "time_step": 0.1
        }
        
        # 4. Call the handler
        # We need to mock the return value of the core module to avoid running actual physics code if desired,
        # but running the actual code is also fine for integration testing.
        # For unit testing, let's assume we want to verify the flow.
        
        response = await handler(mock_request, run_body)
        
        assert isinstance(response, RunExperimentResponse)
        # Since we are using the real physics module (unless mocked), we expect a float result for vanilla_equation
        assert isinstance(response.result, (float, int))

    @mark.asyncio
    async def test_run_experiment_wrong_module(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        # 1. Seed session for GRAVITY
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)

        # 2. Try to call COULOMB handler
        handler = server._create_module_handler("m1_coulomb_force")
        
        # 3. Expect 400 error
        from fastapi import HTTPException
        with raises(HTTPException) as exc_info:
            await handler(mock_request, {})
        
        assert exc_info.value.status_code == 400
        assert "Session configured for 'm0_gravity'" in exc_info.value.detail

    @mark.asyncio
    async def test_run_experiment_uninitialized_session(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        # 1. Do NOT seed session
        
        # 2. Create handler
        handler = server._create_module_handler("m0_gravity")
        
        # 3. Expect 400 error
        from fastapi import HTTPException
        with raises(HTTPException) as exc_info:
            await handler(mock_request, {})
            
        assert exc_info.value.status_code == 400
        assert "Session not initialized" in exc_info.value.detail
