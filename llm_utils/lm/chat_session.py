from copy import deepcopy
from typing import Any, List, Literal, TypedDict
from pydantic import BaseModel
from llm_utils.chat_format import get_conversation_one_turn
from .oai_lm import OAI_LM
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
    ):
        self.lm = deepcopy(lm)
        self.system_prompt = system_prompt
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
        messages = get_conversation_one_turn(
            system_msg=self.system_prompt,
            user_msg=text,
        )
        output = self.lm(
            messages=messages, response_format=response_format, **kwargs
        )
        # output could be a string or a pydantic model
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


