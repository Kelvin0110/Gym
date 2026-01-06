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

import asyncio
import ast
import io
import multiprocessing
import re
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import math
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

import importlib
import inspect
import logging

from resources_servers.newton_bench.schemas import MODULE_REQUEST_CLASSES_MAPPING

REPO_ROOT = Path(__file__).parent.parent.parent
NEWTON_BENCH_PATH = REPO_ROOT / "NewtonBench"
if str(NEWTON_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(NEWTON_BENCH_PATH))

_loaded_modules: dict = {}


def _validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Python code for safety and correctness.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__',
            r'eval\(',
            r'exec\(',
            r'open\(',
            r'file\(',
            r'input\(',
            r'raw_input\(',
            r'compile\(',
            r'globals\(',
            r'locals\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains potentially dangerous operation: {pattern}"
        
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def _get_last_expr_value(code: str, globals_dict: dict, locals_dict: dict):
    """
    Try to evaluate the last line of the submitted code and return its
    string representation. If the last line is not a bare expression
    (or evaluation fails), return None.
    """
    lines = code.strip().split("\n")
    if not lines:
        return None

    last_line = lines[-1].strip()

    if last_line.startswith(("print", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
        return None

    try:
        return str(eval(last_line, globals_dict, locals_dict))
    except Exception:
        return None


def _run_code_in_existing_env(code, globals_d, locals_d, timeout_s):
    """Re-uses the same globals/locals dictionary between calls."""

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()

    def _handle_timeout(signum, frame):
        raise TimeoutError("code timed-out")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_s)
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_d, locals_d)
            result = _get_last_expr_value(code, globals_d, locals_d)
    finally:
        signal.alarm(0)
    return stdout_capture.getvalue(), stderr_capture.getvalue(), result


def _session_worker(child_conn, max_execution_time: int):
    """Runs forever in its own process, keeping globals between calls."""
    
    safe_builtins_list = [
        "abs", "all", "any", "ascii", "bin", "bool", "callable", "chr", "complex", 
        "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", 
        "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", 
        "map", "max", "min", "next", "object", "oct", "ord", "pow", "print", "range", 
        "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", 
        "type", "zip", "Exception", "ArithmeticError", "AssertionError", "AttributeError",
        "BufferError", "EOFError", "ImportError", "IndexError", "KeyError",
        "MemoryError", "NameError", "NotImplementedError", "OSError",
        "OverflowError", "ReferenceError", "RuntimeError", "StopIteration",
        "SyntaxError", "SystemError", "TypeError", "ValueError", "ZeroDivisionError",
        "__import__"
    ]
    
    exec_globals = {
        "__builtins__": {name: __builtins__.get(name) if isinstance(__builtins__, dict) else getattr(__builtins__, name) 
                         for name in safe_builtins_list if (isinstance(__builtins__, dict) and name in __builtins__) or hasattr(__builtins__, name)},
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "pd": pd,
        "pandas": pd,
        "math": math,
    }
    
    exec_locals = {}
    while True:
        msg = child_conn.recv()
        if msg["cmd"] == "exec":
            code = msg["code"]
            try:
                out, err, res = _run_code_in_existing_env(code, exec_globals, exec_locals, max_execution_time)
                child_conn.send({"ok": True, "out": out, "err": err, "res": res})
            except Exception as e:
                child_conn.send({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
        elif msg["cmd"] == "close":
            break


class _SessionHandle:
    """Light wrapper around one long-lived worker process."""

    def __init__(self, max_execution_time: int):
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn
        self._max_execution_time = max_execution_time
        self._proc = multiprocessing.Process(
            target=_session_worker,
            args=(child_conn, max_execution_time),
            daemon=True,
        )
        self._proc.start()
        self.is_closed = False

    def exec(self, code: str):
        try:
            self._conn.send({"cmd": "exec", "code": code})
        except (BrokenPipeError, EOFError, ConnectionError):
            self.close()
            raise 

        if self._conn.poll(self._max_execution_time + 5):
            try:
                reply = self._conn.recv()
            except (BrokenPipeError, EOFError, ConnectionError):
                self.close()
                raise

            if reply["ok"]:
                return reply["out"], reply["err"], reply["res"]
            
            error_msg = reply["error"]
            if "traceback" in reply:
                error_msg += f"\nTraceback:\n{reply['traceback']}"
            raise RuntimeError(error_msg)
            
        self.close()
        raise TimeoutError("Execution timed out (worker unresponsive)")

    def close(self):
        if self.is_closed:
            return
        self.is_closed = True
        
        try:
            self._conn.send({"cmd": "close"})
        except Exception:
            logging.debug("Failed to send close command to worker")
        
        try:
            self._conn.close()
        except Exception:
            logging.exception("Error while closing session pipe")
        
        self._proc.join(timeout=1)
        if self._proc.is_alive():
            logging.warning(f"Session process {self._proc.pid} still alive after close; escalating to terminate.")
            try:
                self._proc.terminate()
                self._proc.join(timeout=1)
                if self._proc.is_alive():
                    logging.error(f"Session process {self._proc.pid} resisted terminate; resorting to kill.")
                    self._proc.kill()
                    self._proc.join()
            except Exception:
                logging.exception(f"Error while force-terminating process {self._proc.pid}")
        
        try:
            self._proc.close()
        except Exception:
            logging.exception("Error during final Process object cleanup")


def _load_module(module_name: str):
    if not module_name:
        raise ImportError("No module_name provided to _load_module")
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]
    try:
        core = importlib.import_module(f"modules.{module_name}.core")
    except Exception as e:
        raise ImportError(f"Unable to import core for NewtonBench module '{module_name}': {e}") from e
    param_description = None
    try:
        prompts_mod = importlib.import_module(f"modules.{module_name}.prompts")
        param_description = getattr(prompts_mod, "PARAM_DESCRIPTION", None)
    except Exception:
        param_description = None
    _loaded_modules[module_name] = {"core": core, "param_description": param_description}
    return _loaded_modules[module_name]


class NewtonBenchResourcesServerConfig(BaseResourcesServerConfig):
    max_execution_time: int = 60    # 1 minute
    session_ttl: int = 1800         # 30 minutes


class RunExperimentResponse(BaseModel):
    result: Union[float, dict]  # float for vanilla_equation, dict for systems


class ExecutePythonRequest(BaseModel):
    code: str


class ExecutePythonResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    result: Optional[str] = None


class NewtonBenchRunRequest(BaseRunRequest):
    difficulty: str
    system: str
    noise_level: float


class NewtonBenchSeedSessionRequest(BaseSeedSessionRequest):
    module_name: str
    difficulty: str
    system: str
    law_version: str
    noise_level: float


class NewtonBenchVerifyRequest(BaseVerifyRequest):
    pass


class NewtonBenchVerifyResponse(BaseVerifyResponse):
    difficulty: Optional[str] = None
    system: Optional[str] = None
    noise_level: Optional[float] = None
    law_version: Optional[str] = None
    extracted_law: Optional[str] = None
    rmsle: Optional[float] = None
    exact_accuracy: Optional[float] = None
    symbolic_equivalent: Optional[bool] = None
    evaluation_error: Optional[str] = None


class NewtonBenchEndSessionRequest(BaseModel):
    pass

class NewtonBenchEndSessionResponse(BaseModel):
    pass

class NewtonBenchResourcesServer(SimpleResourcesServer):
    config: NewtonBenchResourcesServerConfig
    session_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _sessions: Dict[str, _SessionHandle] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            cleanup_task = asyncio.create_task(self._background_cleanup_task())
            
            async with parent_lifespan(app):
                yield
            
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            
            for sid in list(self._sessions.keys()):
                try:
                    self._sessions[sid].close()
                except Exception:
                    pass
                del self._sessions[sid]

        app.router.lifespan_context = lifespan

        modules_dir = NEWTON_BENCH_PATH / "modules"
        try:
            if modules_dir.exists() and modules_dir.is_dir():
                for child in sorted(modules_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    module_name = child.name
                    if module_name == "common":
                        continue

                    route_path = f"/run_experiment_{module_name}"
                    app.add_api_route(route_path, self._create_module_handler(module_name), methods=["POST"])
        except Exception:
            logging.exception("Failed to dynamically register module endpoints")

        app.post("/execute_python")(self.execute_python)
        app.post("/end_session")(self.end_session)

        return app

    async def seed_session(
        self, request: Request, body: NewtonBenchSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        module_name = body.module_name

        if module_name not in MODULE_REQUEST_CLASSES_MAPPING:
            raise HTTPException(status_code=400, detail=f"Invalid module_name '{module_name}'.")

        self.session_metadata[session_id] = {
            "module_name": module_name,
            "difficulty": body.difficulty,
            "system": body.system,
            "noise_level": body.noise_level,
            "law_version": body.law_version,
            "last_used": time.time(),
        }
        return BaseSeedSessionResponse()
    
    async def verify(self, request: Request, body: NewtonBenchVerifyRequest) -> NewtonBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

        metadata["last_used"] = time.time()
        difficulty = metadata.get("difficulty")
        system = metadata.get("system")
        noise_level = metadata.get("noise_level")
        law_version = metadata.get("law_version")

        extracted_law = self._extract_law_from_response(body.response)

        if extracted_law is None:
            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                difficulty=difficulty,
                system=system,
                noise_level=noise_level,
                law_version=law_version,
                reward=0.0,
                extracted_law=None,
                evaluation_error="No law found in response. Expected <final_law>...</final_law> tags.",
            )

        # Use NewtonBench eval func
        try:
            module_name = metadata.get("module_name")
            if not module_name:
                return NewtonBenchVerifyResponse(
                    **body.model_dump(),
                    difficulty=difficulty,
                    system=system,
                    noise_level=noise_level,
                    law_version=law_version,
                    reward=0.0,
                    extracted_law=extracted_law,
                    evaluation_error="Missing module_name in configuration.",
                )

            _mod = _load_module(module_name)
            core = _mod["core"]
            param_description = _mod.get("param_description", None)
    
            eval_result = core.evaluate_law(
                llm_function_str=extracted_law,
                param_description=param_description,
                difficulty=difficulty,
                law_version=law_version,
            )

            # Symbolic equivalence uses LLM judge
            is_symbolically_correct = eval_result.get("symbolic_equivalent", False)
            
            rmsle = eval_result.get("rmsle", float('inf'))
            is_numerically_correct = rmsle < 1e-5

            reward = 1.0 if (is_symbolically_correct or is_numerically_correct) else 0.0

            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                difficulty=difficulty,
                system=system,
                noise_level=noise_level,
                law_version=law_version,
                reward=reward,
                extracted_law=extracted_law,
                rmsle=eval_result.get("rmsle"),
                exact_accuracy=eval_result.get("exact_accuracy"),
                symbolic_equivalent=eval_result.get("symbolic_equivalent"),
                evaluation_error=eval_result.get("error"),
            )

        except Exception as e:
            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                difficulty=difficulty,
                system=system,
                noise_level=noise_level,
                law_version=law_version,
                reward=0.0,
                extracted_law=extracted_law,
                evaluation_error=f"Evaluation failed: {str(e)}",
            )

    def _extract_law_from_response(self, response: Any) -> Optional[str]:
        for output in reversed(response.output):
            if output.type == "message" and output.role == "assistant":
                text_content = ""
                for content in output.content:
                    if content.type == "output_text":
                        text_content += content.text

                start_tag = "<final_law>"
                end_tag = "</final_law>"
                start_index = text_content.rfind(start_tag)
                if start_index == -1:
                    continue
                end_index = text_content.find(end_tag, start_index)
                if end_index == -1:
                    continue

                return text_content[start_index + len(start_tag):end_index].strip()

        return None

    def _create_module_handler(self, module_name: str):
        model_cls = MODULE_REQUEST_CLASSES_MAPPING.get(module_name)
        if model_cls is None:
            raise RuntimeError(f"Missing request class for NewtonBench module '{module_name}'")

        async def handler(request: Request, body: Any):
            session_id = request.session.get(SESSION_ID_KEY)
            metadata = self.session_metadata.get(session_id)
            if not metadata:
                raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

            session_module_name = metadata.get("module_name")
            if session_module_name != module_name:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Session configured for '{session_module_name}', but received run_experiment call for '{module_name}'."
                )

            metadata["last_used"] = time.time()
            difficulty = metadata.get("difficulty")
            system = metadata.get("system")
            noise_level = metadata.get("noise_level")
            law_version = metadata.get("law_version")

            try:
                body_dict = body.model_dump()
            except Exception:
                body_dict = dict(body)

            effective_kwargs = dict(body_dict or {})
            effective_kwargs.setdefault("difficulty", difficulty)
            effective_kwargs.setdefault("system", system)
            effective_kwargs.setdefault("noise_level", noise_level)
            effective_kwargs.setdefault("law_version", law_version)

            try:
                _mod = _load_module(module_name)
                core = _mod["core"]
            except ImportError as ie:
                raise HTTPException(status_code=500, detail=str(ie))

            try:
                # Use NewtonBench experiment runner
                result = core.run_experiment_for_module(**effective_kwargs)
                return RunExperimentResponse(result=result)

            except Exception as e:
                return RunExperimentResponse(result={"error": str(e)})

        handler.__name__ = f"run_experiment_{module_name}"

        try:
            params = [
                inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
                inspect.Parameter("body", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=model_cls),
            ]
            handler.__signature__ = inspect.Signature(parameters=params)
        except Exception:
            logging.debug("Failed to set dynamic signature for handler %s", handler.__name__)

        return handler

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> ExecutePythonResponse:
        """Execute Python code in a session-based environment."""
        sid = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(sid)
        if not metadata:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

        metadata["last_used"] = time.time()
        loop = asyncio.get_running_loop()
        try:
            is_valid, error_message = _validate_python_code(body.code)
            if not is_valid:
                return ExecutePythonResponse(
                    success=False,
                    stdout="",
                    stderr="",
                    error_message=error_message,
                )

            if sid in self._sessions and self._sessions[sid].is_closed:
                del self._sessions[sid]

            if sid not in self._sessions:
                self._sessions[sid] = _SessionHandle(
                    self.config.max_execution_time
                )
            handle = self._sessions[sid]

            try:
                stdout, stderr, result = await loop.run_in_executor(
                    None,
                    handle.exec,
                    body.code,
                )
                return ExecutePythonResponse(
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    result=result,
                )
            except Exception as e:
                if sid in self._sessions and self._sessions[sid].is_closed:
                    del self._sessions[sid]
                raise e
                
        except Exception as e:
            return ExecutePythonResponse(
                success=False,
                stdout="",
                stderr="",
                error_message=str(e),
            )

    async def _background_cleanup_task(self):
        """Periodically check and remove expired sessions."""
        while True:
            try:
                self._cleanup_sessions()
            except Exception:
                logging.exception("Error in background cleanup task")
            await asyncio.sleep(600)  # Check every 10 minutes

    def _cleanup_sessions(self):
        """Remove sessions that have been inactive for longer than session_ttl."""
        now = time.time()
        
        for sid in list(self.session_metadata.keys()):
            last_activity = self.session_metadata[sid].get("last_used", 0)

            if now - last_activity > self.config.session_ttl:
                if sid in self._sessions:
                    try:
                        self._sessions[sid].close()
                    except Exception:
                        pass
                    del self._sessions[sid]
                
                del self.session_metadata[sid]

    async def end_session(self, request: Request, body: NewtonBenchEndSessionRequest) -> NewtonBenchEndSessionResponse:
        """Clean up session handle for Python execution and metadata."""
        sid = request.session[SESSION_ID_KEY]
        if sid not in self.session_metadata:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")

        if sid in self._sessions:
            self._sessions[sid].close()
            del self._sessions[sid]
        
        del self.session_metadata[sid]
        return NewtonBenchEndSessionResponse()

if __name__ == "__main__":
    NewtonBenchResourcesServer.run_webserver()