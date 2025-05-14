"""
Usage:
```python
from llm_utils import OAI_LM
lm = OAI_LM(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
response = lm(prompt="Hello, how are you?")
# Print the response
print(response)

"""

import fcntl
import os
import random
import tempfile
from copy import deepcopy
import time
from typing import Any, List, Literal, Optional, TypedDict, Dict


import numpy as np
from loguru import logger
from pydantic import BaseModel
from speedy_utils import dump_json_or_pickle, identify_uuid, load_json_or_pickle

from llm_utils.chat_format import get_conversation_one_turn

# from llm_utils.chat_session import ChatSession
from .port_utils import (
    _clear_port_use,
    _atomic_save,
    _update_port_use,
    _pick_least_used_port,
)


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str | BaseModel


class OAI_LM:
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache: bool = True,
        callbacks: Optional[Any] = None,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        host="localhost",
        port=None,
        ports=None,
        api_key=None,
        **kwargs,
    ):
        # Lazy import dspy
        import dspy

        self.ports = ports
        self.host = host
        if ports is not None:
            port = ports[0]

        if port is not None:
            kwargs["base_url"] = f"http://{host}:{port}/v1"
            self.base_url = kwargs["base_url"]

        if model is None:
            try:
                model = self.list_models(kwargs.get("base_url"))[0]
                model = f"openai/{model}"
                logger.info(f"Using default model: {model}")
            except Exception as e:
                example_cmd = (
                    "LM.start_server('unsloth/gemma-3-1b-it')\n"
                    "# Or manually run: svllm serve --model unsloth/gemma-3-1b-it --gpus 0 -hp localhost:9150"
                )
                logger.error(
                    f"Failed to initialize dspy.LM: {e}\n"
                    f"Make sure your model server is running and accessible.\n"
                    f"Example to start a server:\n{example_cmd}"
                )
        assert model is not None, "Model name must be provided"

        if not model.startswith("openai/"):
            model = f"openai/{model}"

        self._dspy_lm: dspy.LM = dspy.LM(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,  # disable cache handling by default and implement custom cache handling
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            launch_kwargs=launch_kwargs,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "abc"),
            **kwargs,
        )
        # Store the kwargs for later use
        self.kwargs: Dict[str, Any] = self._dspy_lm.kwargs
        self.model = self._dspy_lm.model

        self.do_cache = cache

    @property
    def last_message(self):
        return self._dspy_lm.history[-1]["response"].model_dump()["choices"][0][
            "message"
        ]

    def __call__(
        self,
        prompt=None,
        messages=None,
        response_format: Optional[type[BaseModel]] = None,
        cache=None,
        retry_count=0,
        port=None,
        error: Optional[Exception] = None,
        use_loadbalance=None,
        must_load_cache=False,
        max_tokens=None,
        num_retries=10,
        **kwargs,
    ) -> str | BaseModel:
        """Main method to call the language model."""
        # Check retry limit
        if retry_count > num_retries and error:
            logger.error(f"Retry limit exceeded, error: {error}, {self.base_url=}")
            raise error
        if retry_count > 0:
            logger.warning(
                f"Retrying {retry_count} times, error: {error}, {self.base_url=}"
            )

        # Prepare parameters
        kwargs = kwargs or self.kwargs
        cache = cache or self.do_cache
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Validate response format if provided
        if response_format:
            self._validate_response_format(response_format)

        # Try to get from cache first
        cache_id = None
        result = None
        if cache:
            cache_id = self._generate_cache_id(
                prompt, messages, response_format, kwargs
            )
            result = self.load_cache(cache_id)

        # Make API call if not in cache or must_load_cache is set and failed
        if not result:
            if must_load_cache:
                raise ValueError(
                    "Expected to load from cache but got None, maybe previous call failed so it didn't save to cache"
                )

            # Select a port for load balancing if needed
            api_port = self._select_port(port, use_loadbalance)
            if api_port:
                # Avoid mutating a dictionary with strict type expectations
                assert isinstance(kwargs, dict), "kwargs must be a dictionary"
                kwargs["base_url"] = f"http://{self.host}:{api_port}/v1"

            # Make the actual API call with error handling
            try:
                result = self._make_api_call(prompt, messages, kwargs, response_format)
            except Exception as e:
                # Handle different types of errors with retries
                return self._handle_api_error(
                    e,
                    prompt,
                    messages,
                    response_format,
                    cache,
                    retry_count,
                    api_port,
                    kwargs,
                    use_loadbalance,
                )
            finally:
                # Update port usage stats if load balancing was used
                if api_port and use_loadbalance:
                    _update_port_use(api_port, -1)

        # Save result to cache if enabled
        if self.do_cache and cache_id:
            self.dump_cache(cache_id, result)

        # Format the response if needed
        if response_format:
            result = self._format_response(
                result, response_format, prompt, messages, cache, retry_count, kwargs
            )
            return result
        assert isinstance(
            result, str
        ), f"Expected result to be a string, got {type(result)}"
        return result

    def _validate_response_format(self, response_format):
        """Validate that the response format is a pydantic model."""
        assert issubclass(
            response_format, BaseModel
        ), f"response_format must be a pydantic model, {type(response_format)} provided"

    def _generate_cache_id(self, prompt, messages, response_format, kwargs):
        """Generate a unique ID for caching based on input parameters."""
        _kwargs = {**self.kwargs, **kwargs}
        s = str(
            [
                prompt,
                messages,
                (response_format.model_json_schema() if response_format else None),
                _kwargs["temperature"],
                _kwargs["max_tokens"],
                self.model,
            ]
        )
        return identify_uuid(s)

    def _select_port(self, port, use_loadbalance):
        """Select an appropriate port for the API call."""
        if self.ports and not port:
            if use_loadbalance:
                return self.get_least_used_port()
            else:
                return random.choice(self.ports)
        return port

    def _make_api_call(self, prompt, messages, kwargs, response_format):
        """Make the actual API call to the language model."""
        result = self._dspy_lm(
            prompt=prompt,
            messages=messages,
            **kwargs,
            response_format=response_format,
        )
        if kwargs.get("n", 1) == 1:
            result = result[0]
        return result

    def _handle_api_error(
        self,
        error,
        prompt,
        messages,
        response_format,
        cache,
        retry_count,
        port,
        kwargs,
        use_loadbalance,
    ):
        """Handle different types of API errors with appropriate retry logic."""
        import litellm

        if isinstance(error, litellm.exceptions.ContextWindowExceededError):
            logger.error(f"Context window exceeded: {error}")
            raise error

        elif isinstance(
            error, (litellm.exceptions.APIError, litellm.exceptions.Timeout)
        ):
            t = 3
            base_url = kwargs.get("base_url", self.base_url)
            if retry_count > 0 and isinstance(error, litellm.exceptions.APIError):
                logger.warning(
                    f"[{base_url=}] API error: {str(error)[:100]}, will sleep for {t}s and retry"
                )
            elif retry_count > 0 and isinstance(error, litellm.exceptions.Timeout):
                logger.warning(
                    f"Timeout error: {str(error)[:100]}, will sleep for {t} seconds and retry"
                )
            time.sleep(t)
            return self.__call__(
                prompt=prompt,
                messages=messages,
                response_format=response_format,
                cache=cache,
                retry_count=retry_count + 1,
                port=port,
                error=error,
                use_loadbalance=use_loadbalance,
                **kwargs,
            )

        else:
            logger.error(f"Error: {error}")
            import traceback

            traceback.print_exc()
            raise error

    def _format_response(
        self, result, response_format, prompt, messages, cache, retry_count, kwargs
    ):
        """Format the response according to the specified response format."""
        import json_repair

        try:
            parsed_result = json_repair.loads(result)
            # Ensure parsed_result is a dict with string keys
            if not isinstance(parsed_result, dict):
                raise ValueError(f"Expected dict, got {type(parsed_result)}")
            # Convert any non-string keys to strings
            parsed_dict = {str(k): v for k, v in parsed_result.items()}
            return response_format(**parsed_dict)
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}, result: {result}")
            return self.__call__(
                prompt=prompt,
                messages=messages,
                response_format=response_format,
                cache=cache,
                retry_count=retry_count + 1,
                error=e,
                **kwargs,
            )

    @property
    def last_think(self):
        # just for fun
        return (
            self._dspy_lm.history[-1]["response"].choices[0].message.reasoning_content
        )

    def clear_port_use(self):
        _clear_port_use(self.ports)

    def get_least_used_port(self):
        if not self.ports:

            raise ValueError(
                "No ports available for load balancing. Please provide a list of ports."
            )
        least_used_port = _pick_least_used_port(self.ports)
        port = least_used_port
        return port

    def dump_cache(self, id, result):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)

            dump_json_or_pickle(result, cache_file)
        except Exception as e:
            logger.warning(f"Cache dump failed: {e}")

    def load_cache(self, id):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)
            if not os.path.exists(cache_file):
                return
            return load_json_or_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def list_models(self, base_url=None):
        import openai

        base_url = base_url or self.kwargs["base_url"]
        client = openai.OpenAI(
            base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "abc")
        )
        page = client.models.list()
        return [d.id for d in page.data]

    @property
    def client(self):
        import openai

        return openai.OpenAI(
            base_url=self.kwargs["base_url"], api_key=os.getenv("OPENAI_API_KEY", "abc")
        )

    @classmethod
    def get_deepseek_chat(cls, api_key=None, max_tokens=2000, **kwargs):
        return OAI_LM(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def get_deepseek_reasoner(cls, api_key=None, max_tokens=2000, **kwargs):
        return OAI_LM(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            max_tokens=max_tokens,
            **kwargs,
        )

    def inspect_history(self):
        """
        Inspect the history of the language model.
        """
        return self._dspy_lm.inspect_history()
