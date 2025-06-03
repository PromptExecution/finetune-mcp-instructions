#!/bin/bash
# Script to evaluate and compare base model vs fine-tuned model on MCP tasks
# Usage: ./scripts/evaluate_models.sh BASE_MODEL FINETUNED_MODEL [TEST_DATA]

set -e

# Default values
BASE_MODEL=${1:-"mistralai/devstral"}
FINETUNED_MODEL=${2:-"finetune_output/final"}
TEST_DATA=${3:-"data/eval_samples.json"}
OUTPUT_DIR="evaluation_results"

echo "Evaluating models on MCP tasks"
echo "Base model: $BASE_MODEL"
echo "Fine-tuned model: $FINETUNED_MODEL"
echo "Test data: $TEST_DATA"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if test data exists, if not, create sample evaluation data
if [ ! -f "$TEST_DATA" ]; then
  echo "Test data file not found, creating sample evaluation data..."

  mkdir -p "$(dirname "$TEST_DATA")"

  # Create a sample evaluation dataset
  cat > "$TEST_DATA" << EOL
{
  "eval_samples": [
    {
      "instruction": "List all files in the src directory and check if there's a Python file",
      "reference": "<use_mcp_tool>\\n<server_name>filesystem</server_name>\\n<tool_name>list_files</tool_name>\\n<arguments>\\n{\\n  \\"path\\": \\"src\\"\\n}\\n</arguments>\\n</use_mcp_tool>",
      "server": "filesystem",
      "complexity": "simple"
    },
    {
      "instruction": "Show me the commit history of the README.md file from the last month",
      "reference": "<use_mcp_tool>\\n<server_name>git</server_name>\\n<tool_name>log</tool_name>\\n<arguments>\\n{\\n  \\"file\\": \\"README.md\\",\\n  \\"since\\": \\"1 month ago\\"\\n}\\n</arguments>\\n</use_mcp_tool>",
      "server": "git",
      "complexity": "simple"
    },
    {
      "instruction": "Find all issues with the label 'bug' in the repository, then show details of the most recent one",
      "reference": "<use_mcp_tool>\\n<server_name>github</server_name>\\n<tool_name>search_issues</tool_name>\\n<arguments>\\n{\\n  \\"query\\": \\"label:bug\\"\\n}\\n</arguments>\\n</use_mcp_tool>\\n\\n<use_mcp_tool>\\n<server_name>github</server_name>\\n<tool_name>get_issue</tool_name>\\n<arguments>\\n{\\n  \\"number\\": 42\\n}\\n</arguments>\\n</use_mcp_tool>",
      "server": "github",
      "complexity": "multi_step"
    },
    {
      "instruction": "Find all JavaScript files in the src directory containing the text 'useEffect', then display the content of the first matched file",
      "reference": "<use_mcp_tool>\\n<server_name>filesystem</server_name>\\n<tool_name>search_files</tool_name>\\n<arguments>\\n{\\n  \\"path\\": \\"src\\",\\n  \\"pattern\\": \\"*.js\\",\\n  \\"content_regex\\": \\"useEffect\\"\\n}\\n</arguments>\\n</use_mcp_tool>\\n\\n<use_mcp_tool>\\n<server_name>filesystem</server_name>\\n<tool_name>read_file</tool_name>\\n<arguments>\\n{\\n  \\"path\\": \\"src/components/App.js\\"\\n}\\n</arguments>\\n</use_mcp_tool>",
      "server": "filesystem",
      "complexity": "multi_step"
    },
    {
      "instruction": "Show all customers who have placed orders worth more than $1000 in total, and list their most expensive order",
      "reference": "<use_mcp_tool>\\n<server_name>database</server_name>\\n<tool_name>execute_query</tool_name>\\n<arguments>\\n{\\n  \\"query\\": \\"SELECT c.customer_id, c.name, SUM(o.total) as total_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name HAVING SUM(o.total) > 1000 ORDER BY total_value DESC\\"\\n}\\n</arguments>\\n</use_mcp_tool>\\n\\n<use_mcp_tool>\\n<server_name>database</server_name>\\n<tool_name>execute_query</tool_name>\\n<arguments>\\n{\\n  \\"query\\": \\"SELECT * FROM orders WHERE customer_id = 42 ORDER BY total DESC LIMIT 1\\"\\n}\\n</arguments>\\n</use_mcp_tool>",
      "server": "database",
      "complexity": "complex"
    }
  ]
}
EOL

  echo "Created sample evaluation data at: $TEST_DATA"
fi

# Path to the standalone Python evaluation script
PYTHON_EVAL_SCRIPT="scripts/run_model_evaluation.py"

# Install required Python packages for evaluation
echo "Installing required packages for evaluation..."
uv pip install matplotlib numpy tqdm sentencepiece transformers torch accelerate bitsandbytes peft

# Run evaluation
echo "Running evaluation..."
python3 "$PYTHON_EVAL_SCRIPT" --base-model "$BASE_MODEL" --finetuned-model "$FINETUNED_MODEL" --eval-data "$TEST_DATA" --output-dir "$OUTPUT_DIR"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
else
  echo "Error: Evaluation failed"
  exit 1
fi