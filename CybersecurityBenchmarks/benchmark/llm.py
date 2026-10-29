# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Any

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore

try:
    from typing_extensions import override
except ImportError:
    # Fallback for environments without typing_extensions
    def override(func):
        return func

from .benchmark_utils import audio_to_b64, image_to_b64
from .cache_handler import CacheHandler


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 環境変数で上書き可（長時間スリープのリトライループを避ける: CREATE_EVAL_LLM_MAX_RETRIES=5 等）
NUM_LLM_RETRIES = _env_int("CREATE_EVAL_LLM_MAX_RETRIES", 100)
MAX_RETRY_TIMEOUT = _env_int("CREATE_EVAL_LLM_MAX_SLEEP_SEC", 600)


def _is_non_retriable_device_error(exception: BaseException) -> bool:
    """
    CUDA / デバイス不正はリトライしても直らない。600 秒 × 多数回の無駄待機を防ぐ。
    Flash/cuDNN SDPA の mha_graph 系も同様（リトライでは直らず、外側の Judge タイムアウトだけが伸びる）。
    """
    msg = f"{type(exception).__name__} {exception}".lower()
    if "cuda" in msg:
        return True
    if "cublas" in msg or "cudnn" in msg or "illegal memory access" in msg:
        return True
    if "cudaerror" in msg:
        return True
    if "mha_graph" in msg or "variant_pack" in msg:
        return True
    return False

MAX_TOKENS = 1000

LOG: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


class RetriesExceededException(Exception):
    def __init__(
        self, message: str = "Maximum number of retries has been exceeded"
    ) -> None:
        self.message = message
        super().__init__(self.message)


class UnretriableQueryException(Exception):
    pass


class LLM(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_handler: CacheHandler | None = None,
    ) -> None:
        if model not in self.valid_models():
            LOG.warning(
                f"{model} is not in the valid model list for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key
        # 正しく引数で受け取った cache_handler を設定する
        self.cache_handler: CacheHandler | None = cache_handler

    @abstractmethod
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts and return the response.

        Args:
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts as well as system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM.
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        # guided_decode_json_schema を子呼び出しに伝搬する
        return self.query(system_prompt + "\n" + prompt, guided_decode_json_schema)

    def supports_parallel_inference(self) -> bool:
        """ThreadPool から同一インスタンスへ同時推論できるか。ローカル HF は False。"""
        return True

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[str, bool, Optional[str]],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                if _is_non_retriable_device_error(exception):
                    LOG.error(
                        f"LLM Query failed with non-retriable device error (not retrying): {exception}"
                    )
                    raise
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def _chat_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[List[str], bool, Optional[str]],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                if _is_non_retriable_device_error(exception):
                    LOG.error(
                        f"LLM Chat failed with non-retriable device error (not retrying): {exception}"
                    )
                    raise
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Chat query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(self.model, None, prompt)
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query, prompt, guided_decode_json_schema
        )

        if self.cache_handler:
            self.cache_handler.save_response(self.model, None, prompt, response)

        return response

    def chat_with_retries(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, None, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat, prompt_with_history, guided_decode_json_schema
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, None, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, prompt
            )
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query_with_system_prompt,
            system_prompt,
            prompt,
            guided_decode_json_schema,
        )

        if self.cache_handler:
            self.cache_handler.save_response(
                self.model, system_prompt, prompt, response
            )

        return response

    def chat_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, system_prompt, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat_with_system_prompt,
            system_prompt,
            prompt_with_history,
            guided_decode_json_schema,
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, system_prompt, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt, system prompt, and image and return the response.
        """
        LOG.error(
            "[ERROR] Please override this method in a subclass to use this method."
        )
        return "Function not implemented."

    def query_with_system_prompt_and_image_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt_and_image,
            system_prompt,
            prompt,
            image_path,
            guided_decode_json_schema,
        )

    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM and return the response. This method
        supports multimodal queries with text, images and audio, and usage of a system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_path (Optional[str]): The path to the image to send to the LLM.
            audio_path (Optional[str]): The path to the audio to send to the LLM.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        raise NotImplementedError(
            "Please override this method in a subclass to use this method."
        )

    def query_multimodal_with_retries(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        num_retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        backoff_factor: float = 0.5,
    ) -> Optional[str]:
        """
        Abstract method to query an LLM and return the response, wrapped in retries.
        This method supports multimodal queries and system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_path (Optional[str]): The path to the image to send to the LLM.
            audio_path (Optional[str]): The path to the audio to send to the LLM.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        cache_key = json.dumps(
            {
                "text_prompt": text_prompt,
                "image_path": image_path,
                "audio_path": audio_path,
            }
        )
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, cache_key
            )
            if cached_response:
                return cached_response

        last_exception = None
        for retry in range(num_retries):
            try:
                response = self.query_multimodal(
                    system_prompt=system_prompt,
                    text_prompt=text_prompt,
                    image_path=image_path,
                    audio_path=audio_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if self.cache_handler:
                    self.cache_handler.save_response(
                        self.model, system_prompt, cache_key, response
                    )
                return response
            except NotImplementedError as exception:
                LOG.error(
                    f"LLM Query failed as the method was not implemented: {exception}"
                )
                raise exception
            except UnretriableQueryException as exception:
                LOG.error(f"LLM Query failed due to unretriable error: {exception}")
                return None
            except Exception as exception:
                last_exception = exception
                if _is_non_retriable_device_error(exception):
                    LOG.error(
                        f"LLM multimodal query failed with non-retriable device error (not retrying): {exception}"
                    )
                    return None
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        LOG.error(f"Unable to query LLM after {num_retries} retries: {last_exception}")
        return None

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []


class OPENAI(LLM):
    """Accessing OPENAI"""

    def __init__(self, model: str, api_key: str) -> None:
        if not OPENAI_AVAILABLE or openai is None:
            raise ImportError("openai package is required for OPENAI provider. Install it with: pip install openai")
        super().__init__(model, api_key)
        self.client = openai.OpenAI(api_key=api_key)  # noqa

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        # Best-level effort to suppress openai log-spew.
        # Likely not work well in multi-threaded environment.
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"]

    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        image = image_to_b64(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64, {image}"},
                        },
                    ],
                },
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message_content = []
        if text_prompt:
            user_message_content.append({"type": "text", "text": text_prompt})

        if image_path:
            image_data = image_to_b64(image_path)
            user_message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64, {image_data}"},
                }
            )

        if audio_path:
            user_message_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_to_b64(str(audio_path)),
                        "format": "wav",
                    },
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message_content,
            }
        )

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        return response.choices[0].message.content


class ANYSCALE(LLM):
    """Accessing ANYSCALE"""

    def __init__(self, model: str, api_key: str) -> None:
        if not OPENAI_AVAILABLE or openai is None:
            raise ImportError("openai package is required for ANYSCALE provider. Install it with: pip install openai")
        super().__init__(model, api_key)
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1", api_key=api_key
        )

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        # Best-level effort to suppress openai log-spew.
        # Likely not work well in multi-threaded environment.
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
            "codellama/CodeLlama-70b-Instruct-hf",
        ]


class TOGETHER(LLM):
    """Accessing TOGETHER"""

    def __init__(self, model: str, api_key: str) -> None:
        if not OPENAI_AVAILABLE or openai is None:
            raise ImportError("openai package is required for TOGETHER provider. Install it with: pip install openai")
        super().__init__(model, api_key)

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.75,
            top_p=1,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query(
        self, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.75,
            top_p=1,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.75,
            top_p=1,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-v0.1",
            "lmsys/vicuna-7b-v1.5",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
            "togethercomputer/falcon-40b",
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
        ]

#################
# ollama (2025-06 Added by Han)
#################
import requests
import base64

DEFAULT_TIMEOUT = 600
OLLAMA_API_URL = "http://localhost:11434/v1"

class OllamaLLM(LLM):
    """Ollama を使用してローカル LLM を動作させるクラス"""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_API_URL,
        timeout: int = DEFAULT_TIMEOUT,
        default_max_tokens: int = MAX_TOKENS,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_top_p: float = DEFAULT_TOP_P,
    ) -> None:
        """
        OllamaLLM 初期化
        - model: Ollama 上の model id (例: "llama3-instruct-backdoor" または "llama3-instruct-backdoor:latest")
        - base_url: Ollama API ベース URL
        - timeout: HTTP タイムアウト(秒)
        - default_*: _send_chat_request のデフォルト生成パラメータ
        """
        super().__init__(model)
        self.base_url = base_url
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        # オプション格納場所（外部から一時的に上書き可能）
        self.options: Dict[str, Union[int, float]] = {}

    def _send_chat_request(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        """Ollama API に POST リクエストを送信してレスポンスを取得（汎用）"""
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        payload["max_tokens"] = int(max_tokens or self.options.get("max_tokens") or self.default_max_tokens)
        payload["temperature"] = float(temperature or self.options.get("temperature") or self.default_temperature)
        payload["top_p"] = float(top_p or self.options.get("top_p") or self.default_top_p)
        if guided_decode_json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            LOG.exception("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama API request failed: {e}") from e

        try:
            data = resp.json()
            # 安全に choices -> message -> content を抜き出す
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Ollama returned no choices: {data}")
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            # content が複合型（list/object）になることがあるので文字列化して返す
            if isinstance(content, (list, dict)):
                return json.dumps(content, ensure_ascii=False)
            return str(content)
        except Exception as e:
            LOG.exception("Failed to parse Ollama response: %s", e)
            raise RuntimeError(f"Failed to parse Ollama response: {e}") from e

    def query(self, prompt: str, guided_decode_json_schema: Optional[str] = None) -> str:
        """1ターンのユーザー入力のみのプロンプトを処理"""
        messages = [{"role": "user", "content": prompt}]
        return self._send_chat_request(messages, guided_decode_json_schema=guided_decode_json_schema)

    def query_with_system_prompt(
        self, system_prompt: str, prompt: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """システムプロンプト＋ユーザープロンプトの単一ターン入力を処理"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._send_chat_request(messages, guided_decode_json_schema=guided_decode_json_schema)

    def chat(
        self, prompt_with_history: List[str], guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """チャット履歴を含めた会話（user/assistantの交互）を処理"""
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        return self._send_chat_request(messages, guided_decode_json_schema=guided_decode_json_schema)

    def chat_with_system_prompt(
        self, system_prompt: str, prompt_with_history: List[str], guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """チャット履歴＋システムプロンプトの組み合わせ"""
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        return self._send_chat_request(messages, guided_decode_json_schema=guided_decode_json_schema)

    def query_with_system_prompt_and_image(
        self, system_prompt: str, prompt: str, image_path: str, guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """画像とテキストを組み合わせたシステム＋ユーザープロンプト"""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ]

        return self._send_chat_request(messages, guided_decode_json_schema=guided_decode_json_schema)

    def valid_models(self) -> List[str]:
        """Ollama で有効なモデル名の一覧を返す（動的取得）。"""
        try:
            resp = requests.get(f"{OLLAMA_API_URL}/models", timeout=5)
            resp.raise_for_status()
            data = resp.json().get("data", []) or []
            ids: List[str] = []
            for m in data:
                mid = m.get("id")
                if not mid:
                    continue
                # model:tag と model の両方を許容
                ids.append(mid)
                if ":" in mid:
                    ids.append(mid.split(":", 1)[0])
            # 重複を排除して返す
            return list(dict.fromkeys(ids))
        except Exception as e:
            LOG.warning("Could not fetch Ollama models: %s", e)
            # フォールバックの静的リスト（既存の値）
            return [
                "llama3.2:1b",
                "gemma2:27b",
                "mistral:7b-instruct",
                "phi3:mini-128k-instruct",
            ]


#################
# Hugging Face (2025-06 Added by Han)
#################
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        AutoProcessor,
        Gemma3ForConditionalGeneration,
    )
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    AutoProcessor = None  # type: ignore
    Gemma3ForConditionalGeneration = None  # type: ignore

try:
    from peft import AutoPeftModelForCausalLM
    PEFT_AVAILABLE = True
except ImportError:
    AutoPeftModelForCausalLM = None  # type: ignore
    PEFT_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore
    HF_HUB_AVAILABLE = False

# 再現性確保: 推論・Judge の同一入力での結果を一致させる（RQ2 改善プラン §1）
REPRODUCIBILITY_SEED = 42

# 429 Too Many Requests 時のリトライ（Hugging Face Hub レート制限対策）
HF_429_RETRIES = 3
HF_429_BACKOFF_SECONDS = [60, 120, 180]


def _is_429(e: BaseException) -> bool:
    """例外が Hugging Face の 429 レート制限に起因するか判定する。"""
    msg = str(e).lower()
    if "429" in msg or "too many requests" in msg:
        return True
    exc = e
    for _ in range(5):
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        if exc is None:
            break
        if "429" in str(exc).lower() or "too many requests" in str(exc).lower():
            return True
    return False


def _stable_hf_attention_env_enabled() -> bool:
    """CREATE_EVAL_HF_STABLE_ATTENTION=1 で Flash/mem-efficient SDPA を切り math/eager 寄りにする。"""
    raw = os.environ.get("CREATE_EVAL_HF_STABLE_ATTENTION", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _hf_local_files_only_enabled() -> bool:
    """
    CREATE_EVAL_HF_LOCAL_FILES_ONLY=1 なら Hugging Face への問い合わせを避け、
    ローカル cache のみで tokenizer / model / processor を解決する。
    """
    raw = os.environ.get("CREATE_EVAL_HF_LOCAL_FILES_ONLY", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _apply_stable_hf_sdp_backends(torch_mod: Any) -> None:
    """cudnn mha_graph 経路を避ける（環境によっては from_pretrained 前に呼ぶ必要がある）。"""
    if not hasattr(torch_mod.backends, "cuda"):
        return
    try:
        torch_mod.backends.cuda.enable_flash_sdp(False)
        torch_mod.backends.cuda.enable_mem_efficient_sdp(False)
        torch_mod.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _resolve_hf_device_map(torch_mod: Any, *, model_id: str = "") -> str:
    """
    Resolve device_map for HF loading.
    Priority:
      1) CREATE_EVAL_HF_DEVICE_MAP (explicit override)
      2) PLaMo -> cuda:0 (auto split breaks lm_head tie + parallel generate)
      3) CUDA available -> auto
      4) otherwise -> cpu (safe on macOS/MPS)
    """
    forced = os.environ.get("CREATE_EVAL_HF_DEVICE_MAP", "").strip()
    if forced:
        return forced
    if "plamo" in model_id.lower():
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
    try:
        if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            return "auto"
    except Exception:
        pass
    return "cpu"


def _load_from_hf_with_retry(loader_name: str, load_fn, *args, **kwargs):
    """from_pretrained 系を 429 時にリトライして実行する。"""
    last_exc = None
    for attempt in range(HF_429_RETRIES):
        try:
            return load_fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < HF_429_RETRIES - 1 and _is_429(e):
                wait = HF_429_BACKOFF_SECONDS[attempt]
                LOG.warning(
                    "Hugging Face 429 (rate limit) loading %s, retry in %ds (attempt %d/%d)",
                    loader_name, wait, attempt + 1, HF_429_RETRIES,
                )
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore


def _looks_like_peft_adapter_load_error(exc: BaseException) -> bool:
    """AutoModel 経路で失敗した際、PEFT adapter repo の可能性が高い例外か判定する。"""
    msg = str(exc).lower()
    return (
        "unrecognized model" in msg
        or "model_type" in msg
        or "adapter_config.json" in msg
    )


def _resolve_peft_base_model_name(
    repo_id: str,
    *,
    local_files_only: bool,
) -> str | None:
    """
    PEFT adapter repo なら adapter_config.json から base_model_name_or_path を返す。
    local_files_only=True のときは cache 上にある adapter_config.json のみを使う。
    """
    if not HF_HUB_AVAILABLE or hf_hub_download is None:
        return None
    try:
        cfg_path = hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            local_files_only=local_files_only,
        )
    except Exception:
        return None
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    base_model = str(cfg.get("base_model_name_or_path") or "").strip()
    return base_model or None


def _warn_plamo_transformers_version() -> None:
    """Model card requires transformers<=4.57.1; newer versions need inference workarounds."""
    try:
        import transformers as _tf

        ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    except Exception:
        return
    if ver > (4, 57):
        LOG.warning(
            "PLaMo 3: transformers %s is outside the model-card range (4.56–4.57.1). "
            "Inference uses use_cache=False on v5; for production consider vLLM>=0.12 or transformers 4.57.",
            _tf.__version__,
        )


def _plamo3cache_getitem(self: Any, layer_idx: int) -> tuple[Any, Any]:
    """Transformers v5 DynamicCache dropped ``__getitem__``; PLaMo finalize still uses it."""
    layer = self.layers[layer_idx]
    return layer.keys, layer.values


def _patch_plamo_modeling_for_transformers_v5(
    model_id: str,
    *,
    local_files_only: bool,
) -> None:
    """
    PLaMo 3 on Transformers v5:
    - list ``_tied_weights_keys`` crashes ``post_init``
    - dict mapping + ``device_map`` leaves ``lm_head`` on meta (accelerate dispatch bug)
    - ``Plamo3Cache.finalize`` indexes cache with ``self[layer]`` (needs ``__getitem__``)
    Use an empty dict here, then tie ``lm_head`` to ``embed_tokens`` after load.
    """
    if "plamo" not in model_id.lower():
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(
            "modeling_plamo.Plamo3ForCausalLM",
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        cls._tied_weights_keys = {}
        cache_cls = get_class_from_dynamic_module(
            "modeling_plamo.Plamo3Cache",
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        cache_cls.__getitem__ = _plamo3cache_getitem
    except Exception as e:
        LOG.warning("PLaMo transformers-v5 modeling patch skipped for %s: %s", model_id, e)


def _tie_plamo_lm_head_to_embeddings(model_instance: Any) -> None:
    """Checkpoint stores only ``embed_tokens``; share weights with ``lm_head`` after load."""
    try:
        input_emb = model_instance.get_input_embeddings()
        output_emb = model_instance.get_output_embeddings()
        if input_emb is None or output_emb is None:
            return
        output_emb.weight = input_emb.weight
        LOG.info("PLaMo: tied lm_head.weight to embed_tokens.weight (transformers v5 workaround)")
    except Exception as e:
        LOG.warning("PLaMo lm_head tie failed: %s", e)


def _plamo_messages_to_completion_prompt(messages: List[Dict[str, str]]) -> str:
    from .plamo_subprocess import plamo_messages_to_completion_prompt

    return plamo_messages_to_completion_prompt(messages)


def _resolve_local_hf_snapshot_path(
    repo_id: str,
    *,
    local_files_only: bool,
) -> str | None:
    """snapshot_download で repo のローカル snapshot ディレクトリを返す。"""
    if not HF_HUB_AVAILABLE or snapshot_download is None:
        return None
    try:
        return str(snapshot_download(repo_id=repo_id, local_files_only=local_files_only))
    except Exception:
        return None


class HuggingFaceLLM(LLM):
    """
    HuggingFaceベースのローカルLLM推論クラス（Llama3, Qwen, Deepseek, Gemma に対応）。
    - trust_remote_code 有効
    - 4bit量子化（任意）
    - モデル種別に応じて tokenizer / processor を自動選択
    - 再現性: 各 _generate_text 呼び出し前に REPRODUCIBILITY_SEED でシード固定
    """

    def supports_parallel_inference(self) -> bool:
        return False

    def __init__(self, model: str, use_4bit: bool = True) -> None:
        if not HUGGINGFACE_AVAILABLE or torch is None:
            raise ImportError("torch and transformers packages are required for HuggingFace provider. Install them with: pip install torch transformers")
        _is_plamo_model = "plamo" in model.lower()
        # 再現性: モデルロード前からシード・cudnn を固定（subprocess 2 回実行で同一結果にする）
        torch.manual_seed(REPRODUCIBILITY_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(REPRODUCIBILITY_SEED)
        if not _is_plamo_model:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        else:
            LOG.info(
                "PLaMo: skipping cudnn deterministic / use_deterministic_algorithms "
                "(breaks logits with custom attention)"
            )
        self._use_stable_attention = _stable_hf_attention_env_enabled() and not _is_plamo_model
        self._local_files_only = _hf_local_files_only_enabled()
        if self._use_stable_attention:
            _apply_stable_hf_sdp_backends(torch)
            LOG.info(
                "CREATE_EVAL_HF_STABLE_ATTENTION: using math SDPA backends + eager attention where supported"
            )
        elif _is_plamo_model and _stable_hf_attention_env_enabled():
            LOG.info(
                "PLaMo: CREATE_EVAL_HF_STABLE_ATTENTION ignored (incompatible with PLaMo 3 SDPA path)"
            )
        if self._local_files_only:
            LOG.info("CREATE_EVAL_HF_LOCAL_FILES_ONLY: loading Hugging Face assets from local cache only")
        super().__init__(model)
        self.model_name = model
        LOG.info(f"🚀 Loading model: {model} (4bit: {use_4bit})")

        # gemma-3 は processor + Gemma3ForConditionalGeneration 経路、
        # gemma-4 は通常の AutoTokenizer/AutoModelForCausalLM 経路を使う。
        model_l = model.lower()
        self.is_gemma3 = ("gemma-3" in model_l) or ("gemma3" in model_l)
        self.is_gemma4 = ("gemma-4" in model_l) or ("gemma4" in model_l)
        self.is_gemma = self.is_gemma3
        self.is_qwen = "qwen" in model.lower()
        self.is_deepseek = "deepseek" in model.lower()
        self.is_plamo = "plamo" in model.lower()
        if self.is_plamo:
            _warn_plamo_transformers_version()
            _patch_plamo_modeling_for_transformers_v5(
                model,
                local_files_only=self._local_files_only,
            )

        # tokenizer or processor 読み込み（テンプレートや特殊トークン定義に必須）。429 時はリトライする。
        try:
            if self.is_gemma3:
                self.processor = _load_from_hf_with_retry(
                    f"processor({model})",
                    lambda: AutoProcessor.from_pretrained(
                        model,
                        trust_remote_code=True,
                        local_files_only=self._local_files_only,
                    ),
                )
                # transformers 版差分で AutoProcessor が tokenizer を内包しないケースがある
                # （processor 自体が GemmaTokenizer を返す）。両方を許容する。
                self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            else:
                try:
                    self.tokenizer = _load_from_hf_with_retry(
                        f"tokenizer({model})",
                        lambda: AutoTokenizer.from_pretrained(
                            model,
                            trust_remote_code=True,
                            local_files_only=self._local_files_only,
                        ),
                    )
                except Exception:
                    snapshot_path = _resolve_local_hf_snapshot_path(
                        model,
                        local_files_only=self._local_files_only,
                    )
                    if snapshot_path:
                        LOG.info(
                            "Retrying tokenizer load for %s via local snapshot path: %s",
                            model,
                            snapshot_path,
                        )
                        self.tokenizer = _load_from_hf_with_retry(
                            f"tokenizer({snapshot_path})",
                            lambda: AutoTokenizer.from_pretrained(
                                snapshot_path,
                                trust_remote_code=True,
                                local_files_only=True,
                            ),
                        )
                    else:
                        base_tokenizer_model = _resolve_peft_base_model_name(
                            model,
                            local_files_only=self._local_files_only,
                        )
                        if not base_tokenizer_model:
                            raise
                        LOG.info(
                            "Retrying tokenizer load for %s via PEFT base model tokenizer: %s",
                            model,
                            base_tokenizer_model,
                        )
                        self.tokenizer = _load_from_hf_with_retry(
                            f"tokenizer({base_tokenizer_model})",
                            lambda: AutoTokenizer.from_pretrained(
                                base_tokenizer_model,
                                trust_remote_code=True,
                                local_files_only=self._local_files_only,
                            ),
                        )
                # eos_token が存在すれば pad_token を安全に設定する（存在しない場合は設定しない）
                if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            LOG.exception("Failed to load tokenizer/processor for %s", model)
            raise RuntimeError(f"Failed to load tokenizer/processor for {model}: {e}") from e

        # 量子化設定（nf4 は正規分布重み向けで精度が良い。未対応環境では例外で fallback）
        quant_config = None
        if use_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                LOG.warning("Could not create BitsAndBytesConfig (bitsandbytes may be unavailable): %s. Falling back to no-quant.", e)
                quant_config = None

        # モデル読み込み。429 時はリトライする。
        _attn_kw: dict[str, Any] = {}
        if self._use_stable_attention:
            _attn_kw["attn_implementation"] = "eager"
        _device_map = _resolve_hf_device_map(torch, model_id=model)
        _load_dtype = torch.bfloat16 if self.is_plamo else torch.float16

        def _load_hf_weights(attn_extra: dict[str, Any]) -> Any:
            if self.is_gemma3:
                return _load_from_hf_with_retry(
                    f"model Gemma({model})",
                    lambda: Gemma3ForConditionalGeneration.from_pretrained(
                        model,
                        device_map=_device_map,
                        trust_remote_code=True,
                        local_files_only=self._local_files_only,
                        dtype=torch.bfloat16,  # torch_dtypeからdtypeに変更（非推奨警告回避）
                        quantization_config=quant_config,
                        **attn_extra,
                    ).eval(),
                )
            try:
                return _load_from_hf_with_retry(
                    f"model({model})",
                    lambda: AutoModelForCausalLM.from_pretrained(
                        model,
                        device_map=_device_map,
                        dtype=_load_dtype,
                        trust_remote_code=True,
                        local_files_only=self._local_files_only,
                        quantization_config=quant_config,
                        **attn_extra,
                    ).eval(),
                )
            except Exception as e:
                if not _looks_like_peft_adapter_load_error(e):
                    raise
                if not PEFT_AVAILABLE or AutoPeftModelForCausalLM is None:
                    raise RuntimeError(
                        f"{model} looks like a PEFT adapter repo, but peft is not installed"
                    ) from e
                LOG.info("Retrying %s as PEFT adapter model", model)
                return _load_from_hf_with_retry(
                    f"peft model({model})",
                    lambda: AutoPeftModelForCausalLM.from_pretrained(
                        model,
                        device_map=_device_map,
                        dtype=_load_dtype,
                        trust_remote_code=True,
                        local_files_only=self._local_files_only,
                        quantization_config=quant_config,
                        **attn_extra,
                    ).eval(),
                )

        try:
            self.model_instance = _load_hf_weights(_attn_kw)
        except (TypeError, ValueError) as e:
            if self._use_stable_attention and _attn_kw:
                LOG.warning(
                    "from_pretrained rejected attention kwargs (%s); retrying without attn_implementation "
                    "(math SDPA backends remain disabled)",
                    e,
                )
                try:
                    self.model_instance = _load_hf_weights({})
                except Exception as e2:
                    LOG.exception("Failed to load model weights for %s", model)
                    raise RuntimeError(f"Failed to load model instance for {model}: {e2}") from e2
            else:
                LOG.exception("Failed to load model weights for %s", model)
                raise RuntimeError(f"Failed to load model instance for {model}: {e}") from e
        except Exception as e:
            LOG.exception("Failed to load model weights for %s", model)
            raise RuntimeError(f"Failed to load model instance for {model}: {e}") from e

        if self.is_plamo:
            _tie_plamo_lm_head_to_embeddings(self.model_instance)

        if torch is not None and torch.cuda.is_available():
            try:
                alloc_gib = torch.cuda.memory_allocated() / (1024**3)
                rsv_gib = torch.cuda.memory_reserved() / (1024**3)
                LOG.info(
                    "VRAM after HuggingFaceLLM load: allocated=%.2f GiB reserved=%.2f GiB (model=%s, 4bit=%s)",
                    alloc_gib,
                    rsv_gib,
                    model,
                    use_4bit,
                )
            except Exception:
                pass

        # オプション: prefill hidden 保存（run.py --save-prefill-hidden-dir）。スレッドごとに prompt_id/pass_id を保持。
        self._prefill_hidden_tls = threading.local()
        self.options: Dict[str, Any] = {}

        # 推論設定（Gemmaは eos/pad は processor側で管理）
        # num_beams=1: 貪欲デコード。ビーム>1 は浮動小数誤差で枝選択がセッション間でずれることがある（再現性 README 参照）。
        self.generate_config = {
            "max_new_tokens": 256,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3,
        }

        if not self.is_gemma:
            self.generate_config.update({
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            })
        # PLaMo 3 base: completion model (not instruct-tuned). Sampling per model card.
        # On transformers v5, Plamo3Cache + use_cache=True is broken; use_cache=False is required.
        if self.is_plamo:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = self.tokenizer.convert_tokens_to_ids("<|plamo:pad|>")
            self.generate_config = {
                "max_new_tokens": 128,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": 1.0,
                "num_beams": 1,
                "use_cache": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_id,
            }
        self._sda_applied = False

    def _get_transformer_core(self) -> Any:
        """
        HF 本体の transformer core（embed_tokens / layers を持つモジュール）を返す。
        通常モデルだけでなく、PeftModel/LoraModel の wrapper 越しも許容する。
        """
        queue = [self.model_instance]
        seen: set[int] = set()
        while queue:
            cur = queue.pop(0)
            if cur is None or id(cur) in seen:
                continue
            seen.add(id(cur))
            if hasattr(cur, "embed_tokens") and hasattr(cur, "layers"):
                return cur
            queue.append(getattr(cur, "model", None))
            queue.append(getattr(cur, "base_model", None))
        raise RuntimeError(
            "Unsupported HF model structure: missing transformer core with embed_tokens/layers"
        )

    def _sda_align_rank1_to_weight(
        self,
        weight: Any,
        b: Any,
        a: Any,
    ) -> tuple[Any, Any]:
        """
        保存された lora_B / lora_A の長さを weight の (out_dim, in_dim) に合わせる。
        抽出元と転移先で語彙サイズのみ異なる（例: embed 128256 vs 128258）場合、
        B は不足分を 0 埋め・超過分は切り捨て、A も同様に in_dim へ合わせる。
        """
        assert torch is not None
        out_dim, in_dim = weight.shape
        b1 = b.reshape(-1).detach().cpu().float()
        a1 = a.reshape(-1).detach().cpu().float()
        if b1.numel() != out_dim:
            LOG.warning(
                "SDA lora_B numel=%s != weight out_dim=%s; padding/truncating (vocab/out_features mismatch on transfer)",
                b1.numel(),
                out_dim,
            )
            if b1.numel() < out_dim:
                b1 = torch.cat(
                    [b1, torch.zeros(out_dim - b1.numel(), dtype=b1.dtype)],
                    dim=0,
                )
            else:
                b1 = b1[:out_dim].contiguous()
        if a1.numel() != in_dim:
            LOG.warning(
                "SDA lora_A numel=%s != weight in_dim=%s; padding/truncating",
                a1.numel(),
                in_dim,
            )
            if a1.numel() < in_dim:
                a1 = torch.cat(
                    [a1, torch.zeros(in_dim - a1.numel(), dtype=a1.dtype)],
                    dim=0,
                )
            else:
                a1 = a1[:in_dim].contiguous()
        return b1, a1

    def _sda_add_rank1_outer_to_weight(
        self,
        weight: Any,
        b: Any,
        a: Any,
        chunk_rows: int,
    ) -> None:
        b, a = self._sda_align_rank1_to_weight(weight, b, a)
        out_dim, in_dim = weight.shape
        bv = b.reshape(out_dim).to(device=weight.device, dtype=weight.dtype)
        av = a.reshape(in_dim).to(device=weight.device, dtype=weight.dtype)
        if chunk_rows <= 0:
            weight.data.add_(bv.unsqueeze(1) * av.unsqueeze(0))
            return
        for start in range(0, out_dim, chunk_rows):
            end = min(out_dim, start + chunk_rows)
            weight.data[start:end].add_(bv[start:end].unsqueeze(1) * av.unsqueeze(0))

    def apply_sda_adapter(
        self,
        adapter_dir: str | Path,
        *,
        chunk_rows_embed: int = 1024,
        chunk_rows_linear: int = 512,
    ) -> None:
        """
        SDA adapter（custom format: adapter_config.json + adapter_weights.pt）を in-place で適用する。
        デフォルト無効。run.py から opt-in の場合のみ呼び出す。
        """
        if self._sda_applied:
            return
        adapter_dir = Path(adapter_dir).resolve()
        cfg_path = adapter_dir / "adapter_config.json"
        w_path = adapter_dir / "adapter_weights.pt"
        if not cfg_path.exists() or not w_path.exists():
            raise FileNotFoundError(f"SDA adapter files not found in {adapter_dir}")
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        schema_version = int(config.get("schema_version", 1))
        state = torch.load(w_path, map_location="cpu", weights_only=True)
        core = self._get_transformer_core()

        if schema_version == 1:
            layer_index = int(config["layer_index"])
            kb = f"base_model.model.model.layers.{layer_index}.mlp.down_proj.lora_B.default"
            ka = f"base_model.model.model.layers.{layer_index}.mlp.down_proj.lora_A.default"
            if kb not in state or ka not in state:
                raise KeyError(f"Missing SDA keys for schema v1: {kb}, {ka}")
            self._sda_add_rank1_outer_to_weight(
                core.layers[layer_index].mlp.down_proj.weight, state[kb], state[ka], chunk_rows_linear
            )
        elif schema_version == 2:
            targets = set(config.get("targets") or [])
            apply_layers = list(config.get("apply_layers") or [])
            if "embed_tokens" in targets:
                kb = "embed_tokens.lora_B.default"
                ka = "embed_tokens.lora_A.default"
                if kb not in state or ka not in state:
                    raise KeyError("Missing embed_tokens SDA keys for schema v2")
                self._sda_add_rank1_outer_to_weight(core.embed_tokens.weight, state[kb], state[ka], chunk_rows_embed)
            for l in apply_layers:
                l = int(l)
                if "self_attn.o_proj" in targets:
                    kb = f"layers.{l}.self_attn.o_proj.lora_B.default"
                    ka = f"layers.{l}.self_attn.o_proj.lora_A.default"
                    if kb not in state or ka not in state:
                        raise KeyError(f"Missing o_proj SDA keys for layer {l}")
                    self._sda_add_rank1_outer_to_weight(
                        core.layers[l].self_attn.o_proj.weight, state[kb], state[ka], chunk_rows_linear
                    )
                if "mlp.down_proj" in targets:
                    kb = f"layers.{l}.mlp.down_proj.lora_B.default"
                    ka = f"layers.{l}.mlp.down_proj.lora_A.default"
                    if kb not in state or ka not in state:
                        raise KeyError(f"Missing down_proj SDA keys for layer {l}")
                    self._sda_add_rank1_outer_to_weight(
                        core.layers[l].mlp.down_proj.weight, state[kb], state[ka], chunk_rows_linear
                    )
        else:
            raise ValueError(f"Unsupported SDA schema_version={schema_version}")
        self._sda_applied = True
        self.options["sda_adapter_path"] = str(adapter_dir)
        LOG.info("Applied SDA adapter: %s", adapter_dir)

    def set_prefill_hidden_context(self, prompt_id: int, pass_id: int) -> None:
        """query_llm からのみ使用。並列スレッドでも衝突しないよう TLS に prompt_id/pass_id を保存。"""
        self._prefill_hidden_tls.ids = (int(prompt_id), int(pass_id))

    def clear_prefill_hidden_context(self) -> None:
        if hasattr(self._prefill_hidden_tls, "ids"):
            del self._prefill_hidden_tls.ids

    def _maybe_attach_prefill_hook(
        self,
        layer_index: int,
        captured: List[Any],
    ) -> Any:
        """非 Gemma: prefill 時の最初の forward で 1 回だけ hidden を取るフックを登録。handle を返す。"""
        model = self.model_instance
        core = self._get_transformer_core()
        if layer_index == 0:
            hook_target = core.embed_tokens
        else:
            n_layers = len(core.layers)
            if not (1 <= layer_index <= n_layers):
                raise ValueError(
                    f"prefill_hidden_layer must be in [0, {n_layers}] for this model, got {layer_index}"
                )
            hook_target = core.layers[layer_index - 1]

        def hook(module: Any, inp: Any, out: Any) -> None:
            if captured:
                return
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h[0, -1, :].detach().float().cpu())

        return hook_target.register_forward_hook(hook)

    def _maybe_save_prefill_hidden_npz(
        self,
        captured: List[Any],
        layer_index: int,
    ) -> None:
        opt = self.options
        out_dir = opt.get("prefill_hidden_dir")
        if not out_dir or not captured:
            return
        try:
            import numpy as np
        except ImportError:
            LOG.warning("numpy not available; skip prefill hidden save")
            return
        ids = getattr(self._prefill_hidden_tls, "ids", None)
        if ids is None:
            LOG.warning("prefill hidden: missing prompt_id/pass_id context; skip save")
            return
        prompt_id, pass_id = ids
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        vec = captured[0].numpy().astype(np.float32)
        fn = path / f"prefill_pid{prompt_id}_pass{pass_id}.npz"
        np.savez_compressed(
            fn,
            hidden=vec,
            layer=np.int32(layer_index),
            prompt_id=np.int32(prompt_id),
            pass_id=np.int32(pass_id),
            model=np.array(self.model_name),
        )
        LOG.debug("Saved prefill hidden to %s", fn)

    def _generate_text(self, messages: List[Dict[str, str]]) -> str:
        """プロンプトを構築し，応答を生成（モデルに応じて分岐）。再現性のため毎回シード固定。"""
        torch.manual_seed(REPRODUCIBILITY_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(REPRODUCIBILITY_SEED)
        if not self.is_plamo and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.is_gemma:
            if self.options.get("prefill_hidden_dir"):
                LOG.warning(
                    "save-prefill-hidden-dir is not implemented for Gemma models; ignoring"
                )
            # system / user を別々に保持（Gemmaの仕様に従う）
            new_messages = []
            for m in messages:
                role = m["role"]
                content = m["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                new_messages.append({"role": role, "content": content})

            # print("==== new_messages ====")
            # print(json.dumps(new_messages, indent=2, ensure_ascii=False))

            # プロンプト構築 & トークナイズ
            inputs = self.processor.apply_chat_template(
                new_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model_instance.device, dtype=torch.bfloat16)

            # print("==== Prompt for generation ====")
            # print("Prompt text:", self.tokenizer.decode(inputs["input_ids"][0]))
        elif self.is_plamo:
            prompt = _plamo_messages_to_completion_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model_instance.device) for k, v in inputs.items()}
        else:
            kwargs = {}
            if self.is_qwen:
                kwargs["enable_thinking"] = False  # Qwen のみ <think> 抑制

            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **kwargs
                )
            else:
                prompt = ""
                for m in messages:
                    prompt += f"{m['role']}:\n{m['content']}\n"
                prompt += "assistant:\n"

            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.model_instance.device) for k, v in inputs.items()}

            # print("==== Prompt for generation ====")
            # print(prompt)

        self.model_instance.eval()
        save_layer = self.options.get("prefill_hidden_layer")
        save_dir = self.options.get("prefill_hidden_dir")
        captured: List[Any] = []
        handle = None
        if (
            not self.is_gemma
            and save_dir is not None
            and save_layer is not None
        ):
            try:
                handle = self._maybe_attach_prefill_hook(int(save_layer), captured)
            except Exception as e:
                LOG.warning("prefill hidden hook failed: %s", e)

        with torch.no_grad():
            output_ids = self.model_instance.generate(**inputs, **self.generate_config)

        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
            try:
                self._maybe_save_prefill_hidden_npz(captured, int(save_layer))
            except Exception as e:
                LOG.warning("prefill hidden save failed: %s", e)

        # 応答取り出し
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]

        if self.is_gemma:
            response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if self.is_deepseek and "</think>" in response:
                response = response.split("</think>")[-1].strip()

        # print("==== Response ====")
        # print(response)
        return response

    # ===== ユーザー用の明示API =====

    def query(self, prompt: str, guided_decode_json_schema: Optional[str] = None) -> str:
        """system prompt を含まない単発プロンプト（userのみ）"""
        messages = [{"role": "user", "content": prompt}]
        return self._generate_text(messages)

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """system prompt を含む単発プロンプト"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._generate_text(messages)

    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """system prompt なしのチャット履歴形式"""
        messages = []
        for i, turn in enumerate(prompt_with_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        return self._generate_text(messages)

    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None
    ) -> str:
        """system prompt を含む履歴付きチャット"""
        messages = [{"role": "system", "content": system_prompt}]
        for i, turn in enumerate(prompt_with_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        return self._generate_text(messages)

    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None
    ) -> str:
        raise NotImplementedError("HF LLM does not support multimodal in this setup")

    def valid_models(self) -> List[str]:
        return [
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3.5-27B",
            "google/gemma-3-4b-it",
        ]


#################
# create LLM instance (2025-06 Updated by Han)
#################
SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"

def create(
    identifier: str,
    use_system_prompt: bool = False,
    sda_adapter_path: str | None = None,
) -> LLM:
    """
    Takes a string of the format <PROVIDER>::<MODEL>(::<API KEY>)? and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.
    """
    api_key: str | None = None
    split = identifier.split("::")
    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    provider = provider.upper()

    if provider == "OPENAI":
        if api_key is None:
            raise ValueError("OPENAI requires an API key.")
        return OPENAI(name, api_key)

    if provider == "ANYSCALE":
        if api_key is None:
            raise ValueError("ANYSCALE requires an API key.")
        return ANYSCALE(name, api_key)

    if provider == "TOGETHER":
        if api_key is None:
            raise ValueError("TOGETHER requires an API key.")
        return TOGETHER(name, api_key)

    if provider == "OLLAMA":
        return OllamaLLM(model=name)

    if provider == "HF":
        use_4bit = True
    elif provider in ("HF16", "HF_FP32", "HF_NO4BIT"):
        use_4bit = False
    else:
        use_4bit = False

    if provider in ("HF", "HF16", "HF_FP32", "HF_NO4BIT"):
        if "plamo" in name.lower():
            from .plamo_subprocess import (
                running_on_plamo_transformers,
                try_create_plamo_subprocess_llm,
            )

            plamo_llm = try_create_plamo_subprocess_llm(name, use_system_prompt=use_system_prompt)
            if plamo_llm is not None:
                if sda_adapter_path:
                    LOG.warning("SDA adapter is not supported for PLaMo subprocess LLM; ignoring")
                return plamo_llm
            if running_on_plamo_transformers():
                LOG.info("PLaMo: using in-process HuggingFaceLLM (transformers 4.57 venv).")
            else:
                LOG.warning(
                    "PLaMo: transformers 4.57 subprocess venv not available "
                    "(run: bash scripts/setup_plamo_venv.sh). "
                    "Falling back to in-process HuggingFaceLLM on transformers 5.x."
                )
        llm = HuggingFaceLLM(model=name, use_4bit=use_4bit)
        llm.options["use_system_prompt"] = use_system_prompt
        if sda_adapter_path:
            llm.apply_sda_adapter(sda_adapter_path)
        return llm

    raise ValueError(f"Unknown provider: {provider}")
