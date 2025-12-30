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

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

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
        # It's okay if prompts or PARAM_DESCRIPTION are missing for some modules
        param_description = None
    _loaded_modules[module_name] = {"core": core, "param_description": param_description}
    return _loaded_modules[module_name]


class NewtonBenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class RunExperimentResponse(BaseModel):
    result: Union[float, dict]  # float for vanilla_equation, dict for systems


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


class NewtonBenchSeedSessionResponse(BaseSeedSessionResponse):
    result: Optional[dict] = None


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


class NewtonBenchResourcesServer(SimpleResourcesServer):
    config: NewtonBenchResourcesServerConfig
    session_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

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

        return app

    async def seed_session(
        self, request: Request, body: NewtonBenchSeedSessionRequest
    ) -> NewtonBenchSeedSessionResponse:
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
        }
        return NewtonBenchSeedSessionResponse()

    async def verify(self, request: Request, body: NewtonBenchVerifyRequest) -> NewtonBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            return NewtonBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                evaluation_error="Session not initialized. Please call seed_session first.",
            )

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

if __name__ == "__main__":
    NewtonBenchResourcesServer.run_webserver()