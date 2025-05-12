from llm_utils import *
from speedy_utils import *
from pydantic import BaseModel
from typing import Dict, Set, Tuple, Union, Any, Type
from tabulate import tabulate


class LMClassifier(OAI_LM):
    """Pydantic wrapper for language models that handles structured data conversion."""

    def __init__(
        self,
        system_prompt: str,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel],
        model: str,
        port: int = None,
        example_input: str = None,
        example_output: str = None,
        **kwargs,
    ):
        super().__init__(model=model, port=port, **kwargs)
        self.system_prompt = system_prompt
        self.input_model = input_model
        self.output_model = output_model
        self.example_input = example_input
        self.example_output = example_output

    def __repr__(self) -> str:
        """Generate a detailed string representation of the LMPydantic instance."""

        # Trim system prompt if needed
        max_prompt_length = 1000
        system_prompt_display = (
            self.system_prompt[:max_prompt_length] + "..."
            if len(self.system_prompt) > max_prompt_length
            else self.system_prompt
        )

        # Get input and output field names
        input_fields = list(self.input_model.__annotations__.keys())
        output_fields = list(self.output_model.__annotations__.keys())

        # Create table data
        table_data = [
            ["System Prompt", system_prompt_display],
            ["Example Input", self.example_input],
            ["Example Output", self.example_output],
            ["Input Model", f"fields: {', '.join(input_fields)}"],
            ["Output Model", f"fields: {', '.join(output_fields)}"],
            ["Language Model", str(self.model)],
        ]

        # Generate table
        return tabulate(table_data, tablefmt="grid")

    def __call__(
        self, input_data: Union[Dict[str, Any], str], strict=True
    ) -> Union[Dict[str, Any], str]:
        if strict:
            if isinstance(input_data, str):
                input_data = jloads(input_data)
            assert isinstance(input_data, dict), "Input data must be a dictionary."
            input_instance = self.input_model(**input_data)
            input_json = jdumps(input_instance.model_dump())
            output_model = self.output_model
        else:
            assert isinstance(input_data, str)
            # assume the input is a JSON string already and do not check
            input_json = input_data
            output_model = None
        msgs = get_conversation_one_turn(
            system_msg=self.system_prompt,
            user_msg=input_json,
        )

        # Using the parent class's __call__ method with proper model validation
        return super().__call__(messages=msgs, response_format=output_model)

    @classmethod
    def from_sharegpt_file(
        cls, file_path: str, model: str = None, port: int = 8165
    ) -> "LMClassifier":
        """Create an LLM_Classifier instance from a ShareGPT file."""
        return initialize_lm_from_sample(file_path, model=model, port=port)


def _build_pydantic_models(
    input_fields: Set[str], output_fields: Set[str]
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """Build Pydantic models based on field names."""
    input_annotations = {field: str for field in input_fields}
    InputModel = type(
        "InputModel", (BaseModel,), {"__annotations__": input_annotations}
    )

    output_annotations = {field: str for field in output_fields}
    OutputModel = type(
        "OutputModel", (BaseModel,), {"__annotations__": output_annotations}
    )

    return InputModel, OutputModel


def initialize_lm_from_sample(
    sample_path: str, model: str = None, port: int = 8165
) -> LMClassifier:
    """Create an LMPydantic instance from a sample conversation file.

    Args:
        sample_path: Path to the sample conversation file
        model: Model name to use, defaults to "gpt-3.5-turbo" if not specified
        port: Port for the API server

    Returns:
        An initialized LLM_Classifier instance
    """
    samples = load_by_ext(sample_path)
    messages = samples[0]["messages"]

    # Extract system prompt
    system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else None
    assert system_prompt is not None, "System prompt not found in the first message."

    # Extract input/output structure
    user_data = jloads(messages[1]["content"])
    input_fields = user_data.keys()

    assistant_data = jloads(messages[2]["content"])
    output_fields = assistant_data.keys()

    # Use a default model if none is provided
    if model is None:
        model = "gpt-3.5-turbo"

    # Build models and LM instance
    InputModel, OutputModel = _build_pydantic_models(input_fields, output_fields)

    return LMClassifier(
        system_prompt,
        InputModel,
        OutputModel,
        model=model,
        port=port,
        example_input=messages[1]["content"],
        example_output=messages[2]["content"],
    )
