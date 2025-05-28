# MCP (Model Context Protocol) Dataset Fine-tuning Toolkit

A comprehensive toolkit for creating datasets, training, and evaluating fine-tuned models for the Model Context Protocol.

## Overview

This toolkit provides end-to-end functionality for fine-tuning large language models on MCP-style instructions using PEFT/QLoRA methods. The system enables efficient fine-tuning on consumer hardware (e.g., RTX 3090) by leveraging parameter-efficient methods.

## Features

- **Dataset Generation**: Create supervised instruction-completion pairs for MCP tasks
- **Example Templating**: Generate templates for various MCP servers (Filesystem, Git, GitHub, Database, etc.)
- **Validation Tools**: Ensure dataset quality and formatting consistency
- **Training Integration**: Fine-tune models using HuggingFace PEFT/QLoRA
- **Evaluation Framework**: Compare fine-tuned models against baselines on MCP tasks

## Architecture

The toolkit uses:
- **UV** for Python environment management
- **PEFT/QLoRA** for parameter-efficient fine-tuning
- **HuggingFace Transformers** for model management
- **Justfile** for workflow orchestration
- **Pytest** for component testing

## Getting Started

```bash
# Set up the environment with UV
just setup

# Generate MCP example templates
just generate-examples

# Generate dataset from examples
just generate-dataset

# Validate the dataset
just validate-dataset

# Run all tests
just test-all

# Run the entire pipeline
just complete-workflow
```

## Technical Details

### Dataset Format

The dataset consists of instruction-completion pairs following MCP conventions:

```json
{
  "examples": [
    {
      "instruction": "Find commit history of main.py file from the last week",
      "completion": "I'll find the commit history...\n\n<use_mcp_tool>\n<server_name>git</server_name>...",
      "metadata": {
        "server": "git",
        "tool": "log",
        "complexity": "simple"
      }
    }
  ]
}
```

### Fine-tuning Approach

- **Loss Masking**: Using DataCollatorForCompletionOnlyLM to mask prompt tokens (setting to -100)
- **Parameter Efficiency**: LoRA adapters focus training on a small subset of model parameters
- **Hardware Requirements**: Optimized for consumer GPUs (e.g., RTX 3090)
- **Integration**: Supports Accelerate for multi-GPU or distributed training

### MCP Server Categories

The toolkit supports examples from multiple MCP servers:

1. **Filesystem**: File operations, directory management, path manipulation
2. **Git**: Repository operations, commit history, diffs, branch management
3. **GitHub**: Issues, PRs, repository exploration, code search
4. **Database**: Query execution, schema exploration
5. **Custom Tools**: Weather data, calculator functions, image processing

## Contributing

See the [TODO.md](TODO.md) file for upcoming features and improvements.

## License

[MIT License](LICENSE)
