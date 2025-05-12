from typing import Any
from pydantic import BaseModel
from .oai_lm import OAI_LM
from typing import List, Union, Tuple

def build_pydantic_response_model_from_string(keys: List[Union[str, Tuple[str, str]]]) -> type[BaseModel]:
    """
    Dynamically build a Pydantic response model from a list of keys or (key, type) tuples.

    Args:
        keys (List[Union[str, Tuple[str, str]]]): A list of field names as strings or tuples of (field_name, field_type).

    Returns:
        type[BaseModel]: A dynamically created Pydantic model class.
    """
    fields = {}
    for key in keys:
        if isinstance(key, str):
            fields[key] = (str, ...)  # Default to string type
        elif isinstance(key, tuple) and len(key) == 2:
            field_name, field_type = key
            fields[field_name] = (eval(field_type), ...)  # Dynamically evaluate type
        else:
            raise ValueError("Keys must be a list of strings or (field_name, field_type) tuples.")

    return type("DynamicResponseModel", (BaseModel,), fields)

class LMAgent:
    def __init__(
        self,
        lm: OAI_LM,
        system_prompt: str = None,
        response_format: type[BaseModel] | List[str] | List[Tuple[str, str]] | str = None,
    ):
        """
        Initialize the LMAgent with a language model, system prompt, and response format.
        """
        assert isinstance(lm, OAI_LM), "model must be an instance of OAI_LM"
        self.lm = lm
            
        self.system_prompt = system_prompt
        # Process response_format if it's a list or string
        if isinstance(response_format, list):
            self.response_format = build_pydantic_response_model_from_string(response_format)
        elif isinstance(response_format, str):
            # Assume it's a single field name
            self.response_format = build_pydantic_response_model_from_string([response_format])
        else:
            self.response_format = response_format
        self.base_messages = [{'role': 'system', 'content': system_prompt}] if system_prompt else []

    def __call__(
        self, 
        prompt: str,
    ):
        messages = self.base_messages + [{'role': 'user', 'content': prompt}]
        return self.lm(messages=messages, response_format=self.response_format)

    def inspect_history(self):
        """
        Return the history of interactions with the language model.
        """
        if hasattr(self.lm, 'inspect_history'):
            return self.lm.inspect_history()
        else:
            return None