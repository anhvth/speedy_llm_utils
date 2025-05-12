# LLM Utils

A Python library providing utility functions for working with Large Language Models (LLMs).

## Overview

LLM Utils is a collection of utility functions and classes designed to simplify working with language models, particularly for chat applications, message formatting, and model interactions. It provides tools for transforming message formats, managing chat sessions, grouping messages by length, and interacting with various LLM providers.

## Features

- **Message Formatting**: Transform between different chat message formats (ChatML, ShareGPT)
- **Chat Sessions**: Manage conversation history and interactions with LLMs
- **Message Grouping**: Efficiently batch messages based on length for optimal processing
- **Model Integration**: Interface with various LLM backends including OpenAI models
- **Visualization**: Display chat messages and conversations as HTML in Jupyter notebooks
- **Command Line Tools**: VLLM serving and load balancing utilities

## Installation

```bash
# Install from PyPI
pip install speedy_llm_utils

# Install from source
pip install -e .

# Or using Poetry
poetry install
```

Note: While the package is named `speedy_llm_utils` on PyPI, you still import it as `llm_utils` in your code:

```python
import llm_utils
```

## Dependencies

- Python ≥ 3.9
- numpy
- pandas
- google-generativeai
- transformers
- dspy
- langchain
- langchain-core
- langchain-openai
- json-repair

## Usage Examples

### Transforming Message Formats

```python
from llm_utils import transform_messages, transform_messages_to_chatml

# Transform messages between formats
chatml_messages = transform_messages_to_chatml(messages)
```

### Working with Chat Sessions

```python
from llm_utils import OAI_LM

# Initialize LLM instance
lm = OAI_LM(model_name="gpt-3.5-turbo")

# Create a chat session
session = lm.create_chat_session(
    system_prompt="You are a helpful assistant."
)

# Generate a response
response = session.send_message("Hello, how can you help me today?")
print(response)
```

### Message Grouping

```python
from llm_utils import split_indices_by_length, group_messages_by_len

# Group messages by length for efficient processing
batches = split_indices_by_length(
    lengths=[len(msg) for msg in messages],
    batch_size_by_mean_length=1000,
    random_seed=42,
    verbose=True,
    shuffle=True
)
```

### Visualizing Conversations

```python
from llm_utils import display_chat_messages_as_html, display_conversations

# Display conversations in a Jupyter notebook
display_conversations(conversations)
```

## Command Line Tools

LLM Utils provides command-line utilities:

- `svllm`: Start a VLLM server
- `svllm-lb`: Run a VLLM load balancer

```bash
# Start a VLLM server
svllm --model <model_name> --port <port>

# Run a load balancer
svllm-lb --servers <server1,server2> --port <port>
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your license information here]

## Author

- Anh Vo
