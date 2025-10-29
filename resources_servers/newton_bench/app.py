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

from fastapi import FastAPI, Request
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

NEWTON_BENCH_PATH = Path(__file__).parent.parent.parent / "NewtonBench"
if str(NEWTON_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(NEWTON_BENCH_PATH))

from modules.m0_gravity import core as gravity_core
from modules.m0_gravity.prompts import PARAM_DESCRIPTION


class NewtonBenchResourcesServerConfig(BaseResourcesServerConfig):
    module_name: str = "m0_gravity"  # only gravity is supported now
    difficulty: str = "easy"  #easy, medium, hard
    system: str = "vanilla_equation"  # vanilla_equation, simple_system, complex_system
    noise_level: float = 0.0
    law_version: Optional[str] = "v0"  # v0, v1, v2, or None for random
    domain: str = "math"


class RunExperimentRequest(BaseModel):
    mass1: float
    mass2: float
    distance: float
    
    # for simple_system and complex_system
    initial_velocity: Optional[float] = None
    duration: Optional[float] = None
    time_step: Optional[float] = None


class RunExperimentResponse(BaseModel):
    result: Union[float, dict]  # float for vanilla_equation, dict for systems


class NewtonBenchRunRequest(BaseRunRequest):
    difficulty: str
    system: str
    noise_level: float


class NewtonBenchSeedSessionRequest(BaseSeedSessionRequest):
    difficulty: str
    system: str
    noise_level: float
    law_version: str


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
        app.post("/run_experiment")(self.run_experiment)
        return app

    async def seed_session(
        self, request: Request, body: NewtonBenchSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_metadata[session_id] = {
            "difficulty": body.difficulty,
            "system": body.system,
            "noise_level": body.noise_level,
            "law_version": body.law_version,
        }
        return BaseSeedSessionResponse()

    async def run_experiment(self, request: Request, body: RunExperimentRequest) -> RunExperimentResponse:
        try:
            session_id = request.session[SESSION_ID_KEY]
            metadata = self.session_metadata.get(session_id, {})

            # TODO: should we remove fallback to config
            difficulty = metadata.get("difficulty", self.config.difficulty)
            system = metadata.get("system", self.config.system)
            noise_level = metadata.get("noise_level", self.config.noise_level)
            law_version = metadata.get("law_version", self.config.law_version)

            kwargs = {}
            if body.initial_velocity is not None:
                kwargs["initial_velocity"] = body.initial_velocity
            if body.duration is not None:
                kwargs["duration"] = body.duration
            if body.time_step is not None:
                kwargs["time_step"] = body.time_step

            # Use NewtonBench experiment runner
            result = gravity_core.run_experiment_for_module(
                mass1=body.mass1,
                mass2=body.mass2,
                distance=body.distance,
                noise_level=noise_level,
                difficulty=difficulty,
                system=system,
                law_version=law_version,
                **kwargs,
            )

            return RunExperimentResponse(result=result)

        except Exception as e:
            return RunExperimentResponse(result={"error": str(e)})

    async def verify(self, request: Request, body: NewtonBenchVerifyRequest) -> NewtonBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        metadata = self.session_metadata.get(session_id, {})
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
            eval_result = gravity_core.evaluate_law(
                llm_function_str=extracted_law,
                param_description=PARAM_DESCRIPTION,
                difficulty=difficulty,
                law_version=law_version,
            )

            # Symbolic equivalence uses LLM judge
            is_symbolically_correct = eval_result.get("symbolic_equivalent", False)
            
            rmsle = eval_result.get("rmsle", float('inf'))
            is_numerically_correct = rmsle < 0.01

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

                match = re.search(r"<final_law>(.*?)</final_law>", text_content, re.DOTALL)
                if match:
                    return match.group(1).strip()

        return None

if __name__ == "__main__":
    NewtonBenchResourcesServer.run_webserver()