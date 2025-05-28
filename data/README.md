# MCP Dataset

This directory contains the MCP (Model Context Protocol) instruction-completion dataset for fine-tuning language models.

## Dataset Structure

The dataset is a JSON file with the following structure:

```json
{
  "examples": [
    {
      "instruction": "User instruction text",
      "completion": "Assistant completion with MCP tool calls",
      "metadata": {
        "server": "mcp_server_type",
        "complexity": "simple|multi_step|complex",
        "tool": "primary_tool_used"
      }
    },
    ...
  ]
}
```

## Dataset Statistics

- Total examples: 20
- Simple examples: 10 (50%)
- Multi-step examples: 5 (25%)
- Complex examples: 5 (25%)

### Server Type Distribution

- git: 4 examples (20%)
- github: 4 examples (20%)
- filesystem: 4 examples (20%)
- database: 4 examples (20%)
- weather: 2 examples (10%)
- calculator: 1 example (5%)
- image_processor: 1 example (5%)

## Using This Dataset

This dataset is designed for fine-tuning language models to work with the Model Context Protocol (MCP). When fine-tuning, you should:

1. Use loss masking to focus learning only on the completion (assistant response) tokens
2. Use HuggingFace's `DataCollatorForCompletionOnlyLM` to properly handle the training
3. Set prompt tokens to -100 to be ignored during training

## Generating More Examples

To expand this dataset:

1. Add more example templates to the `src/mcp_examples/` directory
2. Run the dataset generator script:

```bash
python src/generate_dataset.py --num-examples 100 --output-file expanded_dataset.json