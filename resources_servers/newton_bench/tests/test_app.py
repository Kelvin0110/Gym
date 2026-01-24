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

import asyncio
import time
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Request
from pytest import fixture, mark, raises

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient, SESSION_ID_KEY

from resources_servers.newton_bench.app import (
    NewtonBenchEndSessionRequest,
    NewtonBenchEndSessionResponse,
    NewtonBenchResourcesServer,
    NewtonBenchResourcesServerConfig,
    NewtonBenchSeedSessionRequest,
    NewtonBenchRunRequest,
    NewtonBenchVerifyRequest,
    NewtonBenchVerifyResponse,
    RunExperimentResponse,
    ExecutePythonRequest,
    ExecutePythonResponse,
)


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

    def _make_response_with_law(self, text: str) -> NeMoGymResponse:
        """Build minimal NeMoGymResponse with message + output_text containing given text."""
        return NeMoGymResponse.model_validate({
            "id": "resp_test",
            "object": "response",
            "created_at": 1756531804.0,
            "model": "test",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        })

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
        with raises(HTTPException) as exc_info:
            await handler(mock_request, {})

        assert exc_info.value.status_code == 400
        assert "Session not initialized" in exc_info.value.detail

    @mark.asyncio
    async def test_seed_session_invalid_module_name(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test seed_session with invalid module_name."""
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="invalid_module_xyz",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0",
        )
        with raises(HTTPException) as exc_info:
            await server.seed_session(mock_request, seed_request)
        assert exc_info.value.status_code == 400
        assert "Invalid module_name" in exc_info.value.detail

    # ==================== Verify Tests ====================

    @mark.asyncio
    async def test_verify_success_symbolic_match(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when law is symbolically equivalent."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        law_text = "def discovered_law(mass1, mass2, distance):\n    C = 8e-8\n    return (C * mass1 * mass2) / (distance ** 2)"
        mock_response = self._make_response_with_law(
            f"My discovery:\n<final_law>\n{law_text}\n</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": True,
                "rmsle": 0.1,
                "exact_accuracy": 0.95,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 1.0
        assert response.symbolic_equivalent is True
        assert response.extracted_law is not None

    @mark.asyncio
    async def test_verify_success_numerical_match(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when law is numerically correct (RMSLE < 1e-5) but not symbolically equivalent."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        law_text = "def discovered_law(mass1, mass2, distance):\n    return 8e-8 * mass1 * mass2 / (distance ** 2)"
        mock_response = self._make_response_with_law(f"<final_law>\n{law_text}\n</final_law>")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": False,
                "rmsle": 1e-6,
                "exact_accuracy": 0.99,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 1.0
        assert response.rmsle == 1e-6
        assert response.symbolic_equivalent is False

    @mark.asyncio
    async def test_verify_failure_no_law_tags(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when response has no <final_law> tags."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        mock_response = self._make_response_with_law(
            "I think the law is F = G * m1 * m2 / r^2"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert response.extracted_law is None
        assert "No law found" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_failure_uninitialized_session(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() without seed_session."""
        mock_response = self._make_response_with_law("<final_law>def f(): pass</final_law>")
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with raises(HTTPException) as exc_info:
            await server.verify(mock_request, verify_request)
        assert exc_info.value.status_code == 400
        assert "Session not initialized" in exc_info.value.detail

    @mark.asyncio
    async def test_verify_failure_missing_module_name(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when session metadata is missing module_name."""
        sid = mock_request.session[SESSION_ID_KEY]
        server.session_metadata[sid] = {
            "difficulty": "easy",
            "system": "vanilla_equation",
            "noise_level": 0.0,
            "law_version": "v0",
            "last_used": time.time(),
        }
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert "Missing module_name" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_evaluation_exception(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when core.evaluate_law() raises an exception."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.side_effect = ValueError("Invalid function syntax")
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert response.extracted_law is not None
        assert "Evaluation failed" in (response.evaluation_error or "")
        assert "Invalid function syntax" in (response.evaluation_error or "")

    @mark.asyncio
    async def test_verify_failure_both_wrong(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test verify() when law is both symbolically and numerically incorrect."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        mock_response = self._make_response_with_law(
            "<final_law>def discovered_law(mass1, mass2, distance): return mass1 + mass2</final_law>"
        )
        verify_request = NewtonBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "test", "type": "message"}]},
            response=mock_response,
        )
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_mod = {"core": MagicMock(), "param_description": None}
            mock_mod["core"].evaluate_law.return_value = {
                "symbolic_equivalent": False,
                "rmsle": 100.0,
                "exact_accuracy": 0.0,
            }
            mock_load.return_value = mock_mod
            response = await server.verify(mock_request, verify_request)
        assert isinstance(response, NewtonBenchVerifyResponse)
        assert response.reward == 0.0
        assert response.symbolic_equivalent is False
        assert response.rmsle == 100.0

    # ==================== _extract_law_from_response Tests ====================

    def test_extract_law_multiple_tags_uses_last(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() when multiple <final_law> tags exist (uses last)."""
        text = (
            "First:\n<final_law>def wrong(): pass</final_law>\n\n"
            "Corrected:\n<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>"
        )
        mock_response = self._make_response_with_law(text)
        extracted = server._extract_law_from_response(mock_response)
        assert extracted == "def discovered_law(mass1, mass2, distance): return 1.0"
        assert "wrong" not in (extracted or "")

    def test_extract_law_malformed_missing_end_tag(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() with malformed tags (missing closing tag)."""
        text = "<final_law>def discovered_law(mass1, mass2, distance): return 1.0\n"
        mock_response = self._make_response_with_law(text)
        extracted = server._extract_law_from_response(mock_response)
        assert extracted is None

    def test_extract_law_multiple_messages(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() when tags are in second message."""
        resp = NeMoGymResponse.model_validate({
            "id": "resp_test",
            "object": "response",
            "created_at": 1756531804.0,
            "model": "test",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "First message", "annotations": []}],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<final_law>def discovered_law(mass1, mass2, distance): return 1.0</final_law>",
                            "annotations": [],
                        }
                    ],
                },
            ],
        })
        extracted = server._extract_law_from_response(resp)
        assert extracted == "def discovered_law(mass1, mass2, distance): return 1.0"

    def test_extract_law_empty_tags(self, server: NewtonBenchResourcesServer) -> None:
        """Test _extract_law_from_response() with empty <final_law></final_law> tags."""
        mock_response = self._make_response_with_law("<final_law></final_law>")
        extracted = server._extract_law_from_response(mock_response)
        assert extracted == ""

    # ==================== _create_module_handler Tests ====================

    def test_create_module_handler_missing_request_class(
        self, server: NewtonBenchResourcesServer
    ) -> None:
        """Test _create_module_handler() when module not in MODULE_REQUEST_CLASSES_MAPPING."""
        with patch("resources_servers.newton_bench.app.MODULE_REQUEST_CLASSES_MAPPING", {}):
            with raises(RuntimeError) as exc_info:
                server._create_module_handler("nonexistent_module")
            assert "Missing request class" in str(exc_info.value)

    @mark.asyncio
    async def test_create_module_handler_import_error(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test dynamic handler when module import fails."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        handler = server._create_module_handler("m0_gravity")
        with patch("resources_servers.newton_bench.app._load_module") as mock_load:
            mock_load.side_effect = ImportError("Module not found")
            with raises(HTTPException) as exc_info:
                await handler(mock_request, {"mass1": 10.0, "mass2": 20.0, "distance": 5.0})
            assert exc_info.value.status_code == 500
            assert "Module not found" in exc_info.value.detail

    # ==================== _cleanup_sessions and last_used Tests ====================

    @mark.asyncio
    async def test_cleanup_sessions_expired(self, server: NewtonBenchResourcesServer) -> None:
        """Test _cleanup_sessions() removes sessions that exceed TTL."""
        req1 = MagicMock(spec=Request)
        req1.session = {SESSION_ID_KEY: "session_1"}
        req2 = MagicMock(spec=Request)
        req2.session = {SESSION_ID_KEY: "session_2"}
        await server.seed_session(req1, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        await server.seed_session(req2, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        server.session_metadata["session_1"]["last_used"] = time.time() - 2000
        server.session_metadata["session_2"]["last_used"] = time.time() - 100
        server._sessions["session_1"] = MagicMock()
        server._sessions["session_2"] = MagicMock()
        server._cleanup_sessions()
        assert "session_1" not in server.session_metadata
        assert "session_1" not in server._sessions
        assert "session_2" in server.session_metadata
        assert "session_2" in server._sessions

    @mark.asyncio
    async def test_cleanup_sessions_closes_handles(self, server: NewtonBenchResourcesServer) -> None:
        """Test _cleanup_sessions() calls close() on SessionHandle objects."""
        req = MagicMock(spec=Request)
        req.session = {SESSION_ID_KEY: "session_1"}
        await server.seed_session(req, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        mock_handle = MagicMock()
        server._sessions["session_1"] = mock_handle
        server.session_metadata["session_1"]["last_used"] = time.time() - 2000
        server._cleanup_sessions()
        mock_handle.close.assert_called_once()

    @mark.asyncio
    async def test_last_used_updated_on_activity(
        self, server: NewtonBenchResourcesServer, mock_request: Request
    ) -> None:
        """Test that last_used timestamp is updated on each endpoint call."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        sid = mock_request.session[SESSION_ID_KEY]
        initial_time = server.session_metadata[sid]["last_used"]
        await asyncio.sleep(0.05)
        await server.execute_python(mock_request, ExecutePythonRequest(code="x = 1"))
        updated_time = server.session_metadata[sid]["last_used"]
        assert updated_time > initial_time

    def test_config_defaults(self) -> None:
        """Test that config defaults are set correctly."""
        config = NewtonBenchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        )
        assert config.max_execution_time == 60
        assert config.session_ttl == 1800

    # ==================== Execute Python Tests ====================

    @mark.asyncio
    async def test_execute_python_basic(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test basic Python code execution."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 5\ny = 10\nx + y")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "15"
        assert response.stdout == ""
        assert response.stderr == ""
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_with_numpy(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test Python code execution with numpy."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="import numpy as np\narr = np.array([1, 2, 3])\narr.sum()")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "6"
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_session_persistence(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that variables persist across multiple execute_python calls in the same session."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        # First call: define variable
        request_body1 = ExecutePythonRequest(code="x = 42")
        response1 = await server.execute_python(mock_request, request_body1)
        assert response1.success is True
        
        # Second call: use variable from first call
        request_body2 = ExecutePythonRequest(code="x * 2")
        response2 = await server.execute_python(mock_request, request_body2)
        
        assert isinstance(response2, ExecutePythonResponse)
        assert response2.success is True
        assert response2.result == "84"  # 42 * 2

    @mark.asyncio
    async def test_execute_python_with_print(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test Python code execution with print statements."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="print('Hello, World!')\nprint('Test output')")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert "Hello, World!" in response.stdout
        assert "Test output" in response.stdout
        assert response.error_message is None

    @mark.asyncio
    async def test_execute_python_expression_result(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that the last expression value is captured correctly."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="import math\nmath.sqrt(16)")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "4.0"

    @mark.asyncio
    async def test_execute_python_no_expression_result(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test code that doesn't have a bare expression at the end."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 5\ny = 10\nz = x + y")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result is None  # Last line is assignment, not expression

    @mark.asyncio
    async def test_execute_python_syntax_error(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test validation of syntax errors."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 5 +\n  # Invalid syntax")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "Syntax error" in response.error_message

    @mark.asyncio
    async def test_execute_python_dangerous_import_os(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that dangerous imports like 'os' are blocked."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="import os\nos.system('ls')")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_dangerous_eval(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that dangerous operations like eval() are blocked."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="eval('1+1')")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_dangerous_open(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that file operations like open() are blocked."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="f = open('test.txt', 'w')")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "dangerous operation" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_runtime_error(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test handling of runtime errors during execution."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 5 / 0")  # Division by zero

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        # Runtime errors during execution raise exceptions which are caught
        # The exception is caught by the worker process and propagated back
        # This results in success=False with error_message set
        assert response.success is False
        assert response.error_message is not None
        assert "ZeroDivisionError" in response.error_message or "division by zero" in response.error_message.lower()

    @mark.asyncio
    async def test_execute_python_stderr_field(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that stderr field is properly returned in response."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 42")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        # Verify stderr field exists (even if empty)
        assert hasattr(response, 'stderr')
        assert isinstance(response.stderr, str)
        # Note: We can't test stderr content easily without sys.stderr access,
        # which is blocked for security. The stderr field is captured automatically
        # by redirect_stderr in the execution environment.

    @mark.asyncio
    async def test_execute_python_with_seed_session(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test execute_python after seed_session to verify module access works."""
        # Seed session with a module
        seed_request = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity",
            difficulty="easy",
            system="vanilla_equation",
            noise_level=0.0,
            law_version="v0"
        )
        await server.seed_session(mock_request, seed_request)
        
        # Execute Python code (even without using the module, it should work)
        request_body = ExecutePythonRequest(code="x = 100\ny = 200\nx + y")
        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result == "300"

    @mark.asyncio
    async def test_execute_python_without_seed_session(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test execute_python without seed_session raises 400."""
        request_body = ExecutePythonRequest(code="import math\nmath.pi * 2")

        with raises(HTTPException) as exc_info:
            await server.execute_python(mock_request, request_body)

        assert exc_info.value.status_code == 400
        assert "Session not initialized" in exc_info.value.detail

    @mark.asyncio
    async def test_execute_python_multiple_sessions(self, server: NewtonBenchResourcesServer) -> None:
        """Test that different sessions maintain separate execution environments."""
        request1 = MagicMock(spec=Request)
        request1.session = {SESSION_ID_KEY: "session_1"}
        request2 = MagicMock(spec=Request)
        request2.session = {SESSION_ID_KEY: "session_2"}
        seed = NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        )
        await server.seed_session(request1, seed)
        await server.seed_session(request2, seed)

        # Set x=10 in session 1
        body1 = ExecutePythonRequest(code="x = 10")
        response1 = await server.execute_python(request1, body1)
        assert response1.success is True
        
        # Set x=20 in session 2
        body2 = ExecutePythonRequest(code="x = 20")
        response2 = await server.execute_python(request2, body2)
        assert response2.success is True
        
        # Verify x in session 1 is still 10
        body1_check = ExecutePythonRequest(code="x")
        response1_check = await server.execute_python(request1, body1_check)
        assert response1_check.success is True
        assert response1_check.result == "10"
        
        # Verify x in session 2 is 20
        body2_check = ExecutePythonRequest(code="x")
        response2_check = await server.execute_python(request2, body2_check)
        assert response2_check.success is True
        assert response2_check.result == "20"

    @mark.asyncio
    async def test_end_session(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that end_session cleans up session handles."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="x = 5")
        response = await server.execute_python(mock_request, request_body)
        assert response.success is True

        session_id = mock_request.session[SESSION_ID_KEY]
        assert session_id in server._sessions

        end_response = await server.end_session(mock_request, NewtonBenchEndSessionRequest())

        assert isinstance(end_response, NewtonBenchEndSessionResponse)
        assert session_id not in server._sessions
        assert session_id not in server.session_metadata

    @mark.asyncio
    async def test_end_session_nonexistent(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that end_session handles non-existent sessions gracefully."""
        end_response = await server.end_session(mock_request, NewtonBenchEndSessionRequest())

        assert isinstance(end_response, NewtonBenchEndSessionResponse)

    @mark.asyncio
    async def test_execute_python_complex_calculation(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test complex mathematical calculations."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        request_body = ExecutePythonRequest(code="""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()
std = arr.std()
result = mean + std * 2
result
""")

        response = await server.execute_python(mock_request, request_body)
        
        assert isinstance(response, ExecutePythonResponse)
        assert response.success is True
        assert response.result is not None
        result_value = float(response.result)
        assert result_value > 0  # Should be a positive number

    @mark.asyncio
    async def test_execute_python_function_definition(self, server: NewtonBenchResourcesServer, mock_request: Request) -> None:
        """Test that function definitions work and persist across calls."""
        await server.seed_session(mock_request, NewtonBenchSeedSessionRequest(
            module_name="m0_gravity", difficulty="easy", system="vanilla_equation",
            noise_level=0.0, law_version="v0"
        ))
        # Define a function
        body1 = ExecutePythonRequest(code="def add(a, b):\n    return a + b")
        response1 = await server.execute_python(mock_request, body1)
        assert response1.success is True
        
        # Use the function
        body2 = ExecutePythonRequest(code="add(15, 25)")
        response2 = await server.execute_python(mock_request, body2)
        
        assert isinstance(response2, ExecutePythonResponse)
        assert response2.success is True
        assert response2.result == "40"
