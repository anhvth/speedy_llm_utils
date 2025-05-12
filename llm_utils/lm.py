import fcntl
import os
import random
import tempfile
from copy import deepcopy
import time
from typing import Any, List, Literal, Optional, TypedDict


import numpy as np
from loguru import logger
from pydantic import BaseModel
from speedy_utils import dump_json_or_pickle, identify_uuid, load_json_or_pickle


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str | BaseModel


class ChatSession:
    def __init__(
        self,
        lm: "OAI_LM",
        system_prompt: str = None,
        history: List[Message] = [],
        callback=None,
        response_format=None,
        multi_turn_agent=False,
    ):
        self.lm = deepcopy(lm)
        self.multi_turn = multi_turn_agent
        self.history = history
        self.callback = callback
        self.response_format = response_format
        if system_prompt:
            system_prompt = {
                "role": "system",
                "content": system_prompt,
            }
            self.history.insert(0, system_prompt)

    def __len__(self):
        return len(self.history)

    def __call__(
        self, text, response_format=None, display=False, max_prev_turns=3, **kwargs
    ) -> str | BaseModel:
        response_format = response_format or self.response_format
        if self.multi_turn:
            self.history.append({"role": "user", "content": text})
        output = self.lm(
            messages=self.parse_history(), response_format=response_format, **kwargs
        )
        # output could be a string or a pydantic model
        if isinstance(output, BaseModel):
            if self.multi_turn:
                self.history.append({"role": "assistant", "content": output})
        else:
            assert response_format is None
            if self.multi_turn:
                self.history.append({"role": "assistant", "content": output})
        if display:
            self.inspect_history(max_prev_turns=max_prev_turns)

        if self.callback:
            self.callback(self, output)
        return output

    def send_message(self, text, **kwargs):
        """
        Wrapper around __call__ method for sending messages.
        This maintains compatibility with the test suite.
        """
        return self.__call__(text, **kwargs)

    def parse_history(self, indent=None):
        parsed_history = []
        for m in self.history:
            if isinstance(m["content"], str):
                parsed_history.append(m)
            elif isinstance(m["content"], BaseModel):
                parsed_history.append(
                    {
                        "role": m["role"],
                        "content": m["content"].model_dump_json(indent=indent),
                    }
                )
            else:
                raise ValueError(f"Unexpected content type: {type(m['content'])}")
        return parsed_history

    def inspect_history(self, max_prev_turns=3):
        from llm_utils import display_chat_messages_as_html

        h = self.parse_history(indent=2)
        try:
            from IPython.display import clear_output

            clear_output()
            display_chat_messages_as_html(h[-max_prev_turns * 2 :])
        except:
            pass




def _clear_port_use(ports):
    """
    Clear the usage counters for all ports.
    """
    for port in ports:
        file_counter = f"/tmp/port_use_counter_{port}.npy"
        if os.path.exists(file_counter):
            os.remove(file_counter)


def _atomic_save(array: np.ndarray, filename: str):
    """
    Write `array` to `filename` with an atomic rename to avoid partial writes.
    """
    # The temp file must be on the same filesystem as `filename` to ensure
    # that os.replace() is truly atomic.
    tmp_dir = os.path.dirname(filename) or "."
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
        np.save(tmp, array)
        temp_name = tmp.name

    # Atomically rename the temp file to the final name.
    # On POSIX systems, os.replace is an atomic operation.
    os.replace(temp_name, filename)


def _update_port_use(port: int, increment: int):
    """
    Update the usage counter for a given port, safely with an exclusive lock
    and atomic writes to avoid file corruption.
    """
    file_counter = f"/tmp/port_use_counter_{port}.npy"
    file_counter_lock = f"/tmp/port_use_counter_{port}.lock"

    with open(file_counter_lock, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # If file exists, load it. Otherwise assume zero usage.
            if os.path.exists(file_counter):
                try:
                    counter = np.load(file_counter)
                except Exception as e:
                    # If we fail to load (e.g. file corrupted), start from zero
                    logger.warning(f"Corrupted usage file {file_counter}: {e}")
                    counter = np.array([0])
            else:
                counter = np.array([0])

            # Increment usage and atomically overwrite the old file
            counter[0] += increment
            _atomic_save(counter, file_counter)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pick_least_used_port(ports: List[int]) -> int:
    """
    Pick the least-used port among the provided list, safely under a global lock
    so that no two processes pick a port at the same time.
    """
    global_lock_file = "/tmp/ports.lock"

    with open(global_lock_file, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            port_use = {}
            # Read usage for each port
            for port in ports:
                file_counter = f"/tmp/port_use_counter_{port}.npy"
                if os.path.exists(file_counter):
                    try:
                        counter = np.load(file_counter)
                    except Exception as e:
                        # If the file is corrupted, reset usage to 0
                        logger.warning(f"Corrupted usage file {file_counter}: {e}")
                        counter = np.array([0])
                else:
                    counter = np.array([0])
                port_use[port] = counter[0]

            logger.debug(f"Port use: {port_use}")

            # Pick the least-used port
            lsp = min(port_use, key=port_use.get)

            # Increment usage of that port
            _update_port_use(lsp, 1)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return lsp


class OAI_LM:
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str = None,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
<<<<<<< HEAD
        max_tokens: int = 2048,
=======
        max_tokens: int = 2000,
>>>>>>> main
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
        self.kwargs = self._dspy_lm.kwargs
        self.model = self._dspy_lm.model

        self.do_cache = cache

    @property
    def last_message(self):
<<<<<<< HEAD
        return self.history[-1]["response"].model_dump()["choices"][0]["message"]
=======
        return self._dspy_lm.history[-1]["response"].model_dump()["choices"][0][
            "message"
        ]
>>>>>>> main

    def __call__(
        self,
        prompt=None,
        messages=None,
        response_format: BaseModel = None,
        cache=None,
        retry_count=0,
        port=None,
        error=None,
        use_loadbalance=None,
        must_load_cache=False,
        max_tokens=None,
        num_retries=10,
        **kwargs,
    ) -> str | BaseModel:
<<<<<<< HEAD
        if retry_count > self.kwargs.get("num_retries", 3):
=======
        if retry_count > num_retries:
            # raise ValueError("Retry limit exceeded")
>>>>>>> main
            logger.error(f"Retry limit exceeded, error: {error}, {self.base_url=}")
            raise error
        if retry_count > 0:
            logger.warning(
                f"Retrying {retry_count} times, error: {error}, {self.base_url=}"
            )
        # have multiple ports, and port is not specified
        if not kwargs:
            kwargs = self.kwargs
        id = None
        cache = cache or self.do_cache
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            assert issubclass(
                response_format, BaseModel
            ), f"response_format must be a pydantic model, {type(response_format)} provided"
        result = None
        if cache:
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
            id = identify_uuid(s)
            result = self.load_cache(id)
        if not result:
            import litellm

            if self.ports and not port:
                if use_loadbalance:

                    port = self.get_least_used_port()
                else:
                    port = random.choice(self.ports)

            if port:
                kwargs["base_url"] = f"http://{self.host}:{port}/v1"

            try:
                if must_load_cache:
                    raise ValueError(
                        "Expected to load from cache but got None, maybe previous call failed so it didn't save to cache"
                    )
<<<<<<< HEAD
                result = super().__call__(
=======
                # Use the _dspy_lm instance instead of super()
                result = self._dspy_lm(
>>>>>>> main
                    prompt=prompt,
                    messages=messages,
                    **kwargs,
                    response_format=response_format,
                )
                if kwargs.get("n", 1) == 1:
                    result = result[0]
            except litellm.exceptions.ContextWindowExceededError as e:
                logger.error(f"Context window exceeded: {e}")
            except litellm.exceptions.APIError as e:
                t = 3
                base_url = kwargs["base_url"]
                if retry_count > 0:
                    logger.warning(
                        f"[{base_url=}] API error: {str(e)[:100]}, will sleep for {t}s and retry"
                    )
                time.sleep(t)
                return self.__call__(
                    prompt=prompt,
                    messages=messages,
                    response_format=response_format,
                    cache=cache,
                    retry_count=retry_count + 1,
                    port=port,
                    error=e,
                    **kwargs,
                )
            except litellm.exceptions.Timeout as e:
                t = 3
                if retry_count > 0:
                    logger.warning(
                        f"Timeout error: {str(e)[:100]}, will sleep for {t} seconds and retry"
                    )
                time.sleep(t)
                return self.__call__(
                    prompt=prompt,
                    messages=messages,
                    response_format=response_format,
                    cache=cache,
                    retry_count=retry_count + 1,
                    port=port,
                    error=e,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback

                traceback.print_exc()
                raise
            finally:
                if port and use_loadbalance:
                    _update_port_use(port, -1)

        if self.do_cache:
            self.dump_cache(id, result)
        if response_format:
            import json_repair

            try:
                return response_format(**json_repair.loads(result))
            except Exception as e:
                # try again
                return self.__call__(
                    prompt=prompt,
                    messages=messages,
                    response_format=response_format,
                    cache=cache,
                    retry_count=retry_count + 1,
                    error=e,
                    **kwargs,
                )
        return result

    def clear_port_use(self):
        _clear_port_use(self.ports)

    def get_least_used_port(self):
        least_used_port = _pick_least_used_port(self.ports)
        port = least_used_port
        return port

    def get_session(
        self,
        system_prompt,
        history: List[Message] = None,
        callback=None,
        response_format=None,
        **kwargs,
    ) -> ChatSession:
        if history is None:
            history = []
        else:
            history = deepcopy(history)
        return ChatSession(
            self,
            system_prompt=system_prompt,
            history=history,
            callback=callback,
            response_format=response_format,
            **kwargs,
        )

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

<<<<<<< HEAD
=======
    def __getattr__(self, name):
        """
        Delegate any attributes not found in OAI_LM to the underlying dspy.LM instance.
        This makes sure any dspy.LM methods not explicitly defined in OAI_LM are still accessible.
        """
        # Check __dict__ directly to avoid recursion via hasattr
        if "_dspy_lm" in self.__dict__ and hasattr(self._dspy_lm, name):
            return getattr(self._dspy_lm, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

>>>>>>> main
    @classmethod
    def get_deepseek_chat(self, api_key=None, max_tokens=2000, **kwargs):
        return OAI_LM(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def get_deepseek_reasoner(self, api_key=None, max_tokens=2000, **kwargs):
        return OAI_LM(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            max_tokens=max_tokens,
            **kwargs,
        )
<<<<<<< HEAD
=======

    @staticmethod
    def start_server(model_name, gpus="4567", port=9150, eager=True):
        cmd = f"svllm serve --model {model_name} --gpus {gpus} -hp localhost:{port}"
        if eager:
            cmd += " --eager"
        session_name = f"vllm_{port}"
        is_session_exists = os.system(f"tmux has-session -t {session_name}")
        logger.info(f"Starting server with command: {cmd}")
        if is_session_exists == 0:
            logger.warning(
                f"Session {session_name} exists, please kill it before running the script"
            )
            # as user if they want to kill the session
            user_input = input(
                f"Session {session_name} exists, do you want to kill it? (y/n): "
            )
            if user_input.lower() == "y":
                os.system(f"tmux kill-session -t {session_name}")
                logger.info(f"Session {session_name} killed")
        os.system(cmd)
        # return subprocess.Popen(shlex.split(cmd))

    # set get_agent is get_session
    get_agent = get_session


LM = OAI_LM
>>>>>>> main
