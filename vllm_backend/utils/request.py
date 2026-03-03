# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import base64
import json
from abc import abstractmethod
from io import BytesIO
from typing import Callable, Dict, List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
from vllm.inputs.data import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.outputs import (
    EmbeddingOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.pooling_params import PoolingParams
from vllm.utils import random_uuid

from utils.vllm_backend_utils import TritonSamplingParams


class RequestBase:
    def __init__(
        self, request, executor_callback: Callable, output_dtype: np.dtype, logger
    ):
        self.triton_request = request
        self.executor_callback = executor_callback
        self.output_dtype = output_dtype
        self.logger = logger
        self.id = random_uuid()
        self.stream = False
        self.prepend_input = False

    @abstractmethod
    def _get_input_tensors(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self):
        raise NotImplementedError

    @abstractmethod
    def create_response(self, request_output, *args, **kwargs):
        raise NotImplementedError


class GenerateRequest(RequestBase):
    def __init__(
        self,
        request,
        executor_callback: Callable,
        output_dtype: np.dtype,
        logger,
        lora_repository: Optional[Dict[str, str]] = None,
        supported_loras: Optional[List[str]] = None,
    ):
        super().__init__(request, executor_callback, output_dtype, logger)
        # Attributes for generate requests
        if lora_repository is not None:
            self.lora_repository = lora_repository
        if supported_loras is not None:
            self.supported_loras = supported_loras

    def _get_input_tensors(self):
        # prompt
        prompt = pb_utils.get_input_tensor_by_name(
            self.triton_request, "text_input"
        ).as_numpy()[0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        # image
        images = pb_utils.get_input_tensor_by_name(self.triton_request, "image")
        if images:
            images_vllm = []
            for image_np in images.as_numpy():
                image_b = base64.b64decode(image_np.decode("utf-8"))
                image_rgb = Image.open(BytesIO(image_b)).convert("RGB")
                images_vllm.append(image_rgb)
            if len(images_vllm) > 0:
                prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images_vllm},
                }

        # stream
        stream = pb_utils.get_input_tensor_by_name(self.triton_request, "stream")
        if stream:
            stream = stream.as_numpy()[0]
        else:
            stream = False

        # prepend_input / exclude_input_in_output
        prepend_input = pb_utils.get_input_tensor_by_name(
            self.triton_request, "exclude_input_in_output"
        )
        if prepend_input:
            # When `exclude_input_in_output` is False, we want to prepend input prompt
            # to output, thus prepend_input should be True, and vice versa.
            prepend_input = not prepend_input.as_numpy()[0]
        elif prepend_input is None and stream:
            prepend_input = False
        else:
            prepend_input = True
        if prepend_input and stream:
            raise ValueError(
                "When streaming, `exclude_input_in_output` = False is not allowed."
            )

        # parameters / sampling_parameters
        # An alternative mechanism to receive serialized parameters as an input
        # tensor, because request parameters are not yet supported via BLS.
        sampling_parameters = pb_utils.get_input_tensor_by_name(
            self.triton_request, "sampling_parameters"
        )
        if sampling_parameters:
            parameters = sampling_parameters.as_numpy()[0].decode("utf-8")
        else:
            parameters = self.triton_request.parameters()

        # additional outputs
        additional_outputs = {
            "return_finish_reason": None,
            "return_cumulative_logprob": None,
            "return_logprobs": None,
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(self.triton_request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, stream, prepend_input, parameters, additional_outputs

    async def execute(self):
        (
            prompt,
            self.stream,
            self.prepend_input,
            parameters,
            self.additional_outputs,
        ) = self._get_input_tensors()

        sampling_params = TritonSamplingParams.from_dict(parameters, self.logger)
        lora_name = sampling_params.lora_name
        lora_request = None
        if lora_name is not None:
            lora_id = str(self.supported_loras.index(lora_name) + 1)
            lora_int_id = int(lora_id)
            lora_local_path = self.lora_repository[lora_name]
            lora_request = LoRARequest(lora_id, lora_int_id, lora_local_path)

        response_iterator = self.executor_callback(
            prompt, sampling_params, self.id, lora_request=lora_request
        )

        async for response in response_iterator:
            yield response

    def create_response(
        self,
        request_output: RequestOutput,
        request_output_state: dict,
        prepend_input: bool,
    ):
        output_tensors = []

        # text_output
        prepend_prompt = ""
        if "prev_lens_text_output" not in request_output_state:
            # this is the first response
            if prepend_input:
                prepend_prompt = request_output.prompt
            request_output_state["prev_lens_text_output"] = [0] * len(
                request_output.outputs
            )
        prev_lens = request_output_state["prev_lens_text_output"]
        text_output = [
            (prepend_prompt + output.text[prev_len:]).encode("utf-8")
            for output, prev_len in zip(request_output.outputs, prev_lens)
        ]
        request_output_state["prev_lens_text_output"] = [
            len(output.text) for output in request_output.outputs
        ]
        output_tensors.append(
            pb_utils.Tensor(
                "text_output", np.asarray(text_output, dtype=self.output_dtype)
            )
        )

        # finish_reason
        if self.additional_outputs["return_finish_reason"]:
            finish_reason = [
                str(output.finish_reason) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "finish_reason", np.asarray(finish_reason, dtype=np.object_)
                )
            )

        # cumulative_logprob
        if self.additional_outputs["return_cumulative_logprob"]:
            cumulative_logprob = [
                output.cumulative_logprob for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "cumulative_logprob",
                    np.asarray(cumulative_logprob, dtype=np.float32),
                )
            )

        # logprobs
        # https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/sequence.py#L37-L58
        if self.additional_outputs["return_logprobs"]:
            if "prev_lens_logprobs" not in request_output_state:
                request_output_state["prev_lens_logprobs"] = [0] * len(
                    request_output.outputs
                )
            logprobs = []
            for i in range(len(request_output.outputs)):
                output = request_output.outputs[i]
                if output.logprobs is None:
                    logprobs.append("null".encode("utf-8"))
                    continue
                prev_len = request_output_state["prev_lens_logprobs"][i]
                request_output_state["prev_lens_logprobs"][i] = len(output.logprobs)
                logprobs_py = []
                for logprob_d_vllm in output.logprobs[prev_len:]:
                    logprob_d_py = {}
                    for token_id, logprob_vllm in logprob_d_vllm.items():
                        logprob_d_py[token_id] = {
                            "logprob": logprob_vllm.logprob,
                            "rank": logprob_vllm.rank,
                            "decoded_token": logprob_vllm.decoded_token,
                        }
                    logprobs_py.append(logprob_d_py)
                logprobs.append(json.dumps(logprobs_py).encode("utf-8"))
            output_tensors.append(
                pb_utils.Tensor("logprobs", np.asarray(logprobs, dtype=np.object_))
            )

        # num_input_tokens
        if self.additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # num_output_tokens
        if self.additional_outputs["return_num_output_tokens"]:
            if "prev_lens_num_output_tokens" not in request_output_state:
                request_output_state["prev_lens_num_output_tokens"] = [0] * len(
                    request_output.outputs
                )
            prev_lens = request_output_state["prev_lens_num_output_tokens"]
            num_output_tokens = [
                (len(output.token_ids) - prev_len)
                for output, prev_len in zip(request_output.outputs, prev_lens)
            ]
            request_output_state["prev_lens_num_output_tokens"] = [
                len(output.token_ids) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "num_output_tokens", np.asarray(num_output_tokens, dtype=np.uint32)
                )
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)


class EmbedRequest(RequestBase):
    def __init__(
        self, request, executor_callback: Callable, output_dtype: np.dtype, logger, tokenizer=None
    ):
        super().__init__(request, executor_callback, output_dtype, logger)
        self.tokenizer = tokenizer

    def _get_input_tensors(self):
        embedding_request = pb_utils.get_input_tensor_by_name(
            self.triton_request, "embedding_request"
        ).as_numpy()[0]
        embedding_request = json.loads(embedding_request.decode("utf-8"))
        
        # Get modality (default to "text")
        modality = embedding_request.get("modality", "text")
        
        # prompt/input
        input_data = embedding_request["input"]
        
        if modality == "image":
            # For image modality, return raw conversations for async processing in execute()
            conversations = input_data if isinstance(input_data, list) else [input_data]
            prompt = {"modality": "image", "conversations": conversations}
        else:
            # Process as text (default behavior)
            prompt = input_data
            if isinstance(prompt, str):
                pass  # Single string - use as is
            elif isinstance(prompt, list):
                if len(prompt) > 0 and isinstance(prompt[0], int):
                    # Single list of token IDs
                    prompt = TokensPrompt(prompt_token_ids=prompt)
                elif len(prompt) > 0 and isinstance(prompt[0], str):
                    # Batch of strings - mark for parallel processing
                    prompt = {"modality": "text_batch", "texts": prompt}
                else:
                    # Empty list or unknown type - use as is
                    pass

        # pooling_params
        pooling_params = self._to_pooling_params(embedding_request)

        # additional outputs
        additional_outputs = {
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(self.triton_request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, pooling_params, additional_outputs

    async def execute(self):
        (
            prompt,
            pooling_params,
            self.additional_outputs,
        ) = self._get_input_tensors()

        # Helper function for processing single item in batch
        async def process_single_item(item):
            outputs = []
            unique_id = random_uuid()
            async for response in self.executor_callback(item, pooling_params, unique_id):
                outputs.append(response)
            return outputs

        # Handle image modality - process conversations asynchronously
        if isinstance(prompt, dict) and prompt.get("modality") == "image":
            conversations = prompt["conversations"]
            
            if not self.tokenizer:
                raise ValueError("Tokenizer is required for multimodal embeddings")
            
            # First pass: extract all data (prompt texts and image URLs)
            conversation_data = []
            for conversation in conversations:
                # Apply chat_template to get formatted prompt string with image placeholders
                prompt_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Extract image URL from conversation
                image_url = None
                for message in conversation:
                    if message.get("role") == "user":
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image_url":
                                    image_url = item.get("image_url", {}).get("url")
                                    break
                
                if not image_url:
                    raise ValueError("No image_url found in user message")
                
                conversation_data.append({
                    "prompt_text": prompt_text,
                    "image_url": image_url
                })
            
            # Fetch all images in parallel using asyncio.to_thread
            async def fetch_one_image(url):
                return await asyncio.to_thread(fetch_image, url)
            
            try:
                images = await asyncio.gather(*[fetch_one_image(data["image_url"]) for data in conversation_data])
            except Exception as e:
                raise ValueError(f"Failed to fetch images: {str(e)}")
            
            # Combine prompts with fetched images
            prompts = []
            for data, image_media in zip(conversation_data, images):
                prompts.append({
                    "prompt": data["prompt_text"],
                    "multi_modal_data": {"image": image_media},
                })
            
            # For batch processing, we need to call encode for each item separately
            # vLLM will batch them internally
            if len(prompts) > 1:
                # Process all prompts in parallel with asyncio.gather
                all_results = await asyncio.gather(*[process_single_item(p) for p in prompts])
                
                # Yield results in order
                for result_list in all_results:
                    for response in result_list:
                        yield response
                return
            else:
                # Single item, use original flow
                prompt = prompts[0]
        
        # Handle text batch - process each text separately
        elif isinstance(prompt, dict) and prompt.get("modality") == "text_batch":
            texts = prompt["texts"]
            
            # Process all texts in parallel with asyncio.gather
            all_results = await asyncio.gather(*[process_single_item(t) for t in texts])
            
            # Yield results in order
            for result_list in all_results:
                for response in result_list:
                    yield response
            return

        # Create PoolingParams for embeddings
        response_iterator = self.executor_callback(prompt, pooling_params, self.id)

        # Yield each response from the async iterator
        async for response in response_iterator:
            yield response

    def _to_pooling_params(self, embedding_request: dict):
        pooling_params_dict = embedding_request.get("pooling_params", {})
        
        # Extract dimensions if present
        dims = pooling_params_dict.get("dimensions", [None])[0] if "dimensions" in pooling_params_dict else None
        
        # Create PoolingParams once with all parameters
        if dims is not None:
            return PoolingParams(dimensions=dims, task="embed")
        return PoolingParams(task="embed")

    def create_response(self, request_output: PoolingRequestOutput[EmbeddingOutput]):
        output_tensors = []
        request_output = EmbeddingRequestOutput.from_base(request_output)

        # Extract embedding list from output
        embedding: list[float] = request_output.outputs.embedding
        output_tensors.append(
            pb_utils.Tensor(
                "text_output",
                np.asarray([json.dumps(embedding)], dtype=self.output_dtype),
            )
        )

        # num_input_tokens
        if self.additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # For embeddings, num_output_tokens is 0 (no generation happened)
        if self.additional_outputs["return_num_output_tokens"]:
            output_tensors.append(
                pb_utils.Tensor("num_output_tokens", np.asarray(0, dtype=np.uint32))
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)
