# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# uses Unsloth vLLM engine for both memory optimizations (FP8, LoRA, weight-sharing, standby mode)
# and batched inference with token IDs and logprobs for RL training. 
 
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionDeveloperMessageParam,
    NeMoGymChatCompletionMessageParam,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionToolParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymFunctionDefinition,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
    TokenIDLogProbMixin,
)

logger = logging.getLogger(__name__)


class UnslothVLLMModelConfig(BaseResponsesAPIModelConfig):
    # Model
    model_name: str = Field(description="Unsloth model name or path to checkpoint")
    load_in_fp8: bool = Field(default=True, description="Use FP8 quantization (60% less VRAM)")
    load_in_4bit: bool = Field(default=False, description="Use 4-bit quantization (fallback if no FP8)")
    max_seq_length: int = Field(default=8192, description="Maximum context length")

    # LoRA
    lora_r: Optional[int] = Field(default=None, description="LoRA rank (None to skip LoRA)")
    lora_alpha: Optional[int] = Field(default=None, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.0, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(default=None, description="LoRA target modules")

    # vLLM
    gpu_memory_utilization: float = Field(default=0.85, description="GPU memory utilization for vLLM")
    max_logprobs: int = Field(default=1, description="Number of top-k logprobs per token (1=chosen token only, matches vllm_model)")
    enable_prefix_caching: bool = Field(default=True, description="Enable prefix caching in vLLM")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism size")

    # Batching
    batch_size: int = Field(
        default=64, description="Maximum number of requests to batch together"
    )
    batch_timeout_seconds: float = Field(
        default=0.1, description="Maximum time to wait for batch to fill (in seconds, e.g., 0.1 = 100ms)"
    )
    max_concurrent_requests: int = Field(
        default=64, description="Maximum number of concurrent inference requests (uses semaphore)"
    )

    # Standby
    enable_standby: bool = Field(default=True, description="Enable standby mode for sleep/wake")
    unsloth_vllm_standby: bool = Field(default=True, description="Enable Unsloth standby mode")

    return_token_id_information: bool = Field(
        default=True, description="Return token IDs and logprobs for RL training"
    )
    uses_reasoning_parser: bool = Field(default=True, description="Parse <think> tags for reasoning")
    uses_hermes_tool_call_parser: bool = Field(
        default=False, description="Parse Hermes-style <tool_call> tags for tool calling"
    )


@dataclass
class BatchedRequest:
    request_id: str
    prompt_text: str
    sampling_params: "SamplingParams"
    body: NeMoGymResponseCreateParamsNonStreaming
    future: asyncio.Future = field(default_factory=asyncio.Future)


class ConverterState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    # We are mapping from Response input items to chat completions messages, which is many to one.
    # Our state will accumulate the reasoning, chat, and tool calls for assistant messages.
    content_buffer: str = ""  # Buffer for reasoning and chat
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)
    reasoning_buffer: List[str] = Field(default_factory=list)  # Unsloth-specific: buffer reasoning before wrapping in think tags

    # Will only be populated if return_token_id_information is True.
    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.content_buffer or self.tool_calls_buffer):
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
            tool_calls=self.tool_calls_buffer,
        )

        # We check here that self.token_information is non-empty since it's possible that some assistant messages are entirely inputs and are not generated by the model in this trajectory.
        if self.return_token_id_information and self.token_information:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.tool_calls_buffer = []


class UnslothConverter(BaseModel):
    return_token_id_information: bool

    # Reasoning handling may change across models and model families

    THINK_TAG_PATTERN: ClassVar = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{t}</think>" for t in texts if t)

    @classmethod
    def _parse_think_tags(cls, content: str) -> Tuple[List[str], str]:
        # Extract reasoning content from between <think></think> tags.
        matches = cls.THINK_TAG_PATTERN.findall(content)
        # Remove reasoning from main content
        cleaned = cls.THINK_TAG_PATTERN.sub("", content)
        return matches, cleaned

    # Response create params to Chat Completion create params
    def responses_to_chat_completion_create_params(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        responses_create_params = responses_create_params.model_dump(exclude_unset=True)

        # Tracks messages including reasoning for each respective message type helper function
        state = ConverterState(
            return_token_id_information=self.return_token_id_information
        )

        # Input can be a string. Wrap in a ResponseInput-like
        response_input = responses_create_params["input"]
        if isinstance(response_input, str):
            wrapped_input = {
                "content": [
                    {
                        "text": response_input,
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
            input_messages = [wrapped_input]
        else:
            input_messages = responses_create_params.pop("input", [])

        for m in input_messages:
            if not m.get("type") and m.get("role"):
                m["type"] = "message"

            match m["type"]:
                case "message":
                    self._format_message(m, state)
                case "reasoning":
                    self._format_reasoning(m, state)
                case "function_call":
                    self._format_function_call(m, state)
                case "function_call_output":
                    self._format_function_call_output(m, state)
                case _:  # pragma: no cover
                    raise NotImplementedError(f"Unsupported message type: {m}")

            if self.return_token_id_information and m.get("prompt_token_ids"):
                state.token_information = TokenIDLogProbMixin(
                    prompt_token_ids=m["prompt_token_ids"],
                    generation_token_ids=m["generation_token_ids"],
                    generation_log_probs=m["generation_log_probs"],
                )

        state.flush_assistant()

        model = responses_create_params.pop("model", None)
        if model is not None:
            responses_create_params["model"] = model

        # The corresponding parameter to `max_output_tokens`` is `max_tokens`
        max_output_tokens = responses_create_params.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            responses_create_params["max_tokens"] = max_output_tokens

        tools = responses_create_params.pop("tools", None)
        if tools is not None:
            responses_create_params["tools"] = []
            for tool_dict in tools:
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)
                responses_create_params["tools"].append(
                    NeMoGymChatCompletionToolParam(type="function", function=NeMoGymFunctionDefinition(**tool_dict))
                )

        chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=state.messages,
            **responses_create_params,
        )

        return chat_completion_create_params

    def _format_function_call_output(
        self,
        m: dict,
        state: ConverterState,
    ) -> None:
        state.flush_assistant()

        assert "call_id" in m
        converted = NeMoGymChatCompletionToolMessageParam(
            content=m["output"],
            role="tool",
            tool_call_id=m["call_id"],
        )
        state.messages.append(converted)

    def _format_message(
        self,
        m: dict,
        state: ConverterState,
    ) -> None:
        content = m["content"]

        # for unsloth, convert list content to string for tokenizer.apply_chat_template()
        if isinstance(content, list):
            content = "".join([part.get("text", "") for part in content if part.get("type") in ["input_text", "text"]])

        match m["role"]:
            case "assistant":
                final_content = ""
                if isinstance(content, str):
                    final_content += content
                else:
                    raise NotImplementedError(
                        f"Expected m['content'] to be str or list[dict], but got {type(content).__name__!r}: {content!r}"
                    )

                if state.reasoning_buffer:
                    final_content = self._wrap_reasoning_in_think_tags(state.reasoning_buffer) + final_content
                    state.reasoning_buffer = []

                converted = []
                state.content_buffer += final_content
            case "user":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionUserMessageParam(
                        content=content,
                        role="user",
                    )
                ]
            # TODO: Revisit this in case we need separate handling. Not all chat templates may support the 'developer' role.
            case "system":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionSystemMessageParam(
                        content=content,
                        role="system",
                    )
                ]
            case "developer":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionDeveloperMessageParam(
                        content=content,
                        role="developer",
                    )
                ]
            case _:  # pragma: no cover
                raise NotImplementedError(f"Unrecognized role for message: `{m['role']}`")

        state.messages.extend(converted)

    def _format_reasoning(
        self,
        m: dict,
        state: ConverterState,
    ) -> None:
        """
        Collects text from 'reasoning' messages and appends it to a buffer.

        This is done to group together one (or multiple) reasoning message(s) into a single,
        cohesive block, later prepending it to a subsequent assistant message.
        See: https://gitlab-master.nvidia.com/bxyu/nemo-gym#reasoning-in-the-response-api
        """
        reasoning_text = "".join([s.get("text", "") for s in m.get("summary", [])])
        if reasoning_text:
            state.reasoning_buffer.append(reasoning_text)

    def _format_function_call(
        self,
        m: dict,
        state: ConverterState,
    ) -> None:
        assert "call_id" in m
        tool_call = NeMoGymChatCompletionMessageToolCallParam(
            id=m["call_id"],
            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                arguments=m["arguments"],
                name=m["name"],
            ),
            type="function",
        )
        state.tool_calls_buffer.append(tool_call)

    def vllm_output_to_nemo_response(
        self,
        request_id: str,
        vllm_output: Any,
        body: NeMoGymResponseCreateParamsNonStreaming,
        uses_reasoning_parser: bool,
        uses_hermes_tool_call_parser: bool = False,
    ) -> NeMoGymResponse:
        from vllm.outputs import RequestOutput

        assert isinstance(vllm_output, RequestOutput), f"Expected RequestOutput, got {type(vllm_output)}"

        completion = vllm_output.outputs[0]
        text = completion.text

        response_output = []

        # parse reasoning if enabled
        if uses_reasoning_parser:
            reasoning_texts, text = self._parse_think_tags(text)
            if reasoning_texts:
                reasoning_item = NeMoGymResponseReasoningItem(
                    id=f"rs_{uuid4().hex}",
                    type="reasoning",
                    summary=[NeMoGymSummary(text=rt, type="summary_text") for rt in reasoning_texts],
                    status="completed",
                )
                response_output.append(reasoning_item)

        # parse tool calls if enabled
        tool_calls_raw = []
        if uses_hermes_tool_call_parser:
            tool_calls_raw, text = self._parse_hermes_tool_calls(text)

        # If model stops with no chat or tool calls, we add an output item with empty or null content
        has_empty_output = not (response_output or tool_calls_raw)

        if text or has_empty_output:
            output_message = NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                role="assistant",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
                status="completed",
                type="message",
            )
            response_output.append(output_message)

        for tc in tool_calls_raw:
            assert "id" in tc
            response_output.append(
                NeMoGymResponseFunctionToolCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    call_id=tc["id"],
                    type="function_call",
                    status="completed",
                    id=tc["id"],
                )
            )

        # Add token ids and log probs to last output item
        if self.return_token_id_information:
            prompt_token_ids = vllm_output.prompt_token_ids
            generation_token_ids = completion.token_ids
            generation_log_probs = self._extract_logprobs(completion.logprobs, generation_token_ids)

            last_response_output_item = response_output[-1]
            train_cls = RESPONSES_TO_TRAIN[last_response_output_item.__class__]
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
            )

        output_items: List[NeMoGymResponseOutputItem] = response_output

        response = NeMoGymResponse(
            id=request_id,
            created_at=int(time()),
            model=body.model or "unsloth-vllm",
            object="response",
            output=output_items,
            tool_choice=body.tool_choice if hasattr(body, "tool_choice") else "auto",
            parallel_tool_calls=body.parallel_tool_calls if hasattr(body, "parallel_tool_calls") else None,
            tools=body.tools if hasattr(body, "tools") else None,
            temperature=body.temperature if hasattr(body, "temperature") else None,
            top_p=body.top_p if hasattr(body, "top_p") else None,
            background=body.background if hasattr(body, "background") else None,
            max_output_tokens=body.max_output_tokens if hasattr(body, "max_output_tokens") else None,
            max_tool_calls=body.max_tool_calls if hasattr(body, "max_tool_calls") else None,
            previous_response_id=body.previous_response_id if hasattr(body, "previous_response_id") else None,
            prompt=body.prompt if hasattr(body, "prompt") else None,
            reasoning=body.reasoning if hasattr(body, "reasoning") else None,
            service_tier=body.service_tier if hasattr(body, "service_tier") else None,
            text=body.text if hasattr(body, "text") else None,
            top_logprobs=body.top_logprobs if hasattr(body, "top_logprobs") else None,
            truncation=body.truncation if hasattr(body, "truncation") else None,
            metadata=body.metadata if hasattr(body, "metadata") else None,
            instructions=body.instructions if hasattr(body, "instructions") else None,
            user=body.user if hasattr(body, "user") else None,
        )

        return response

    def _extract_logprobs(
        self, logprobs_list: Optional[List[Dict]], token_ids: List[int]
    ) -> List[float]:
        if not logprobs_list:
            return []

        generation_log_probs = []
        for i, token_logprob_dict in enumerate(logprobs_list):
            if i >= len(token_ids):
                break
            token_id = token_ids[i]
            if token_id in token_logprob_dict:
                generation_log_probs.append(token_logprob_dict[token_id].logprob)
            else:
                available_tokens = list(token_logprob_dict.keys())
                raise ValueError(
                    f"Sampled token_id {token_id} not found in logprobs at position {i}. "
                    f"Available token_ids: {available_tokens} "
                    f"This indicates a bug in vLLM or a mismatch between token_ids and logprobs."
                )

        return generation_log_probs

    def _extract_reasoning_from_content(self, content: str) -> Tuple[List[str], str]:
        # TODO: Currently only parses reasoning wrapped in <think>...</think> tags.
        # Maybe parameterize to support other model formats in the future.
        return self._parse_think_tags(content)

    def _parse_hermes_tool_calls(self, text: str) -> Tuple[List[dict], str]:
        """ Hermes format: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

        Returns:
            Tuple of (tool_calls_list, remaining_text_without_tool_calls)
        """
        tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )

        if "<tool_call>" not in text:
            return [], text

        try:
            function_call_tuples = tool_call_regex.findall(text)

            raw_function_calls = [
                json.loads(match[0] if match[0] else match[1])
                for match in function_call_tuples
            ]

            tool_calls = [
                {
                    "id": f"call_{uuid4().hex}",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(
                            function_call["arguments"], ensure_ascii=False
                        ),
                    }
                }
                for function_call in raw_function_calls
            ]

            content = text[:text.find("<tool_call>")]

            return tool_calls, content if content else ""

        except Exception as e:
            logger.warning(f"Error parsing Hermes tool calls: {e}")
            return [], text


class UnslothVLLMModel(SimpleResponsesAPIModel):
    config: UnslothVLLMModelConfig

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def model_post_init(self, context):
        logger.info("Initializing Unsloth vLLM model...")

        if self.config.enable_standby:
            os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
            logger.info("Standby enabled (UNSLOTH_VLLM_STANDBY=1)")

        from unsloth import FastLanguageModel

        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"- FP8: {self.config.load_in_fp8}, 4-bit: {self.config.load_in_4bit}")
        logger.info(f"- Max sequence length: {self.config.max_seq_length}")
        logger.info(f"- Max logprobs: {self.config.max_logprobs}")

        try:
            load_kwargs = {
                "model_name": self.config.model_name,
                "fast_inference": True,
                "max_seq_length": self.config.max_seq_length,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_logprobs": self.config.max_logprobs,
                "unsloth_vllm_standby": self.config.unsloth_vllm_standby,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "tensor_parallel_size": self.config.tensor_parallel_size,
            }

            if self.config.load_in_fp8:
                load_kwargs["load_in_fp8"] = True
                load_kwargs["load_in_4bit"] = False
            elif self.config.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["load_in_fp8"] = False

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
            logger.info("Model loaded successfully")
        except Exception as e:
            import traceback
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Model name: {self.config.model_name}")
            logger.error(f"Config: {self.config}")
            raise RuntimeError(f"Failed to load Unsloth model '{self.config.model_name}': {e}") from e

        if self.config.lora_r:
            logger.info(f"Adding LoRA adapter (r={self.config.lora_r})")
            target_modules = self.config.target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha or self.config.lora_r,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
            )
            logger.info("LoRA adapter added")

        self.vllm_engine = self.model.vllm_engine
        self.fast_generate = self.model.fast_generate

        self._converter = UnslothConverter(
            return_token_id_information=self.config.return_token_id_information
        )

        # TODO: replace this batching with async vllm engine
        self._batch_queue: Optional[asyncio.Queue[BatchedRequest]] = None
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._inference_semaphore: Optional[asyncio.Semaphore] = None
        self._sleep_lock: Optional[asyncio.Lock] = None
        self._is_sleeping = False

        logger.info(f"Batch size: {self.config.batch_size}, timeout: {self.config.batch_timeout_seconds}s")
        logger.info(f"Max concurrent requests: {self.config.max_concurrent_requests}")
        logger.info("Unsloth vLLM initialization complete")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM usage: {vram_gb:.2f} GB")

        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.on_event("startup")
        async def startup_batch_processor():
            self._batch_queue = asyncio.Queue()
            self._inference_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            self._sleep_lock = asyncio.Lock()
            self._batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Batch processor and async components initialized")

        # TODO: test this
        # https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl
        if self.config.enable_standby:
            @app.post("/v1/engine/sleep")
            async def sleep_engine():
                async with self._sleep_lock:
                    if not self._is_sleeping:
                        logger.info("Putting vLLM engine to sleep...")
                        self.vllm_engine.sleep()
                        torch.cuda.empty_cache()
                        self._is_sleeping = True
                        vram_gb = torch.cuda.memory_allocated() / 1e9
                        logger.info(f"Engine sleeping. VRAM usage: {vram_gb:.2f} GB")
                return {"status": "sleeping"}

            @app.get("/v1/engine/status")
            async def engine_status():
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                status = {
                    "status": "sleeping" if self._is_sleeping else "awake",
                    "vram_usage_gb": vram_gb,
                }
                if self._inference_semaphore is not None:
                    status["concurrent_requests"] = self.config.max_concurrent_requests - self._inference_semaphore._value
                    status["available_slots"] = self._inference_semaphore._value
                return status

        return app

    # TODO: Use an async vllm engine instead of this hack
    async def _batch_processor(self):
        # builds a batch of requests before timeout or batch_size is reached
        # since fast_generate is sync and not thread-safe, we need to batch requests somehow (this works for now)
        
        logger.info("DEBUG: Batch processor started")

        while True:
            batch: List[BatchedRequest] = []

            try:
                # Wait for first request
                first_request = await self._batch_queue.get()
                batch.append(first_request)

                # Try to collect more requests up to batch_size or timeout
                deadline = time() + self.config.batch_timeout_seconds

                while len(batch) < self.config.batch_size:
                    remaining_time = deadline - time()
                    if remaining_time <= 0:
                        break

                    try:
                        request = await asyncio.wait_for(
                            self._batch_queue.get(),
                            timeout=remaining_time
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

                # Process the batch
                if batch:
                    logger.debug(f"Processing batch of {len(batch)} requests")
                    await self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                # Set exceptions on all pending futures
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    async def _process_batch(self, batch: List[BatchedRequest]):
        try:
            prompts = [req.prompt_text for req in batch]
            sampling_params_list = [req.sampling_params for req in batch]

            start_time = time()
            outputs = await asyncio.to_thread(
                self.fast_generate,
                prompts,
                sampling_params=sampling_params_list
            )
            elapsed = time() - start_time

            logger.debug(f"Batch of {len(batch)} completed in {elapsed:.2f}s ({elapsed/len(batch):.2f}s per request)")

            for req, vllm_output in zip(batch, outputs):
                try:
                    response = self._converter.vllm_output_to_nemo_response(
                        request_id=req.request_id,
                        vllm_output=vllm_output,
                        body=req.body,
                        uses_reasoning_parser=self.config.uses_reasoning_parser,
                        uses_hermes_tool_call_parser=self.config.uses_hermes_tool_call_parser,
                    )
                    req.future.set_result(response)
                except Exception as e:
                    logger.error(f"Error converting output for request {req.request_id}: {e}", exc_info=True)
                    req.future.set_exception(e)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            # Set exception on all pending futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """
        Process a single inference request by enqueueing it for batched processing.
        The batch processor will collect multiple requests and process them together.
        """
        request_id = f"req-{uuid4()}"

        # wake engine if sleeping (from unsloth standby mode)
        async with self._sleep_lock:
            if self._is_sleeping:
                logger.info("Waking vLLM engine...")
                self._is_sleeping = False

        # limit concurrent requests
        async with self._inference_semaphore:
            logger.debug(f"Enqueueing request {request_id}")

            chat_params = self._converter.responses_to_chat_completion_create_params(body)

            prompt_text = self.tokenizer.apply_chat_template(
                chat_params.messages,
                tools=chat_params.tools if chat_params.tools else None,
                tokenize=False,
                add_generation_prompt=True
            )

            from vllm import SamplingParams

            sampling_params_dict = {
                "temperature": body.temperature if body.temperature is not None else 1.0,
                "top_p": body.top_p if body.top_p is not None else 1.0,
            }

            if body.max_output_tokens is not None:
                sampling_params_dict["max_tokens"] = body.max_output_tokens
            else:
                sampling_params_dict["max_tokens"] = self.config.max_seq_length

            if self.config.return_token_id_information:
                sampling_params_dict["logprobs"] = self.config.max_logprobs

            sampling_params = SamplingParams(**sampling_params_dict)

            batched_request = BatchedRequest(
                request_id=request_id,
                prompt_text=prompt_text,
                sampling_params=sampling_params,
                body=body,
            )
            await self._batch_queue.put(batched_request)

            response = await batched_request.future

            return response

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> Any:
        raise NotImplementedError(
            "Chat completions endpoint not implemented. Use responses endpoint instead."
        )


if __name__ == "__main__":
    UnslothVLLMModel.run_webserver()
