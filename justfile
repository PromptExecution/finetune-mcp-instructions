# MCP Dataset Generator Justfile
# This file contains recipes for common tasks using uv instead of direct Python

# Set the default Python interpreter to use uv
python := "uv run python"

# Default recipe to show available commands
default:
    @just --list

# Setup the virtual environment with all dependencies
setup:
    @echo "Setting up virtual environment with uv"
    uv venv
    uv pip install -r requirements.txt
    @echo "Installing Hugging Face CLI"
    uv pip install --upgrade huggingface_hub
    @echo "Run 'huggingface-cli login' to authenticate with Hugging Face"

# Generate example templates for all MCP server types
generate-examples:
    @echo "Generating MCP example templates"
    {{python}} src/generate_examples.py

# Generate dataset from examples
generate-dataset num_examples="50" output_file="mcp_dataset.json":
    @echo "Generating dataset with {{num_examples}} examples"
    {{python}} src/generate_dataset.py --num-examples {{num_examples}} --output-file {{output_file}}

# Validate dataset examples
validate-dataset dataset_file="data/mcp_dataset.json":
    @echo "Validating dataset examples in {{dataset_file}}"
    {{python}} src/validate_dataset.py {{dataset_file}}

# Test all subsystems and create GitHub issues for failures
test-all:
    @echo "Testing all subsystems"
    just test-template-helpers
    just test-examples-generator
    just test-dataset-generator
    @echo "All tests completed"

# Test template helpers
test-template-helpers:
    @echo "Testing template helpers"
    {{python}} src/run_tests.py --subsystem template_helpers

# Test examples generator
test-examples-generator:
    @echo "Testing examples generator"
    {{python}} src/run_tests.py --subsystem examples_generator

# Test dataset generator
test-dataset-generator:
    @echo "Testing dataset generator"
    {{python}} src/run_tests.py --subsystem dataset_generator

# Create a GitHub issue for a failure
create-issue title body:
    @echo "Creating GitHub issue: {{title}}"
    gh issue create --title "{{title}}" --body "{{body}}" || echo "Failed to create GitHub issue"

# Run tests with verbose output and GitHub token
test detail subsystem:
    @echo "Testing {{subsystem}} with detailed output"
    GITHUB_TOKEN=$(gh auth token) {{python}} src/run_tests.py --subsystem {{subsystem}} --github-token $(gh auth token)

# Run a complete workflow: generate examples, dataset, and validate
complete-workflow num_examples="50":
    @echo "Running complete workflow"
    just generate-examples
    just generate-dataset {{num_examples}}
    just validate-dataset

# Create the test directory structure if it doesn't exist
create-test-structure:
    @echo "Creating test directory structure"
    mkdir -p src/tests
    touch src/tests/__init__.py

# Authenticate with Hugging Face
hf-login:
    @echo "Authenticating with Hugging Face"
    huggingface-cli login

# Download Devstral base model
download-devstral target_dir="models/devstral":
    @echo "Downloading Devstral base model to {{target_dir}}"
    mkdir -p {{target_dir}}
    huggingface-cli download devstral/Devstral --local-dir {{target_dir}}

# Download a specific model from Hugging Face
download-model model_name target_dir:
    @echo "Downloading {{model_name}} to {{target_dir}}"
    mkdir -p {{target_dir}}
    huggingface-cli download {{model_name}} --local-dir {{target_dir}}

# Fine-tune model with PEFT/LoRA
finetune-model model_dir="models/devstral" dataset_file="data/mcp_dataset.json" output_dir="models/mcp-finetuned":
    @echo "Fine-tuning model from {{model_dir}} with dataset {{dataset_file}}"
    mkdir -p {{output_dir}}
    {{python}} src/train.py --model-dir {{model_dir}} --dataset {{dataset_file}} --output-dir {{output_dir}}

# Evaluate model performance
evaluate-model model_dir="models/mcp-finetuned" eval_dataset="data/mcp_eval.json":
    @echo "Evaluating model at {{model_dir}} with dataset {{eval_dataset}}"
    {{python}} src/evaluate.py --model-dir {{model_dir}} --dataset {{eval_dataset}}

# Compare base and fine-tuned models
compare-models base_model="models/devstral" finetuned_model="models/mcp-finetuned" eval_dataset="data/mcp_eval.json":
    @echo "Comparing models on MCP tasks"
    {{python}} src/compare.py --base-model {{base_model}} --finetuned-model {{finetuned_model}} --dataset {{eval_dataset}}

# Verify GPU availability
check-gpu:
    @echo "Checking GPU availability"
    {{python}} -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB' if torch.cuda.is_available() else 'No GPU')"

# Complete training workflow
training-workflow dataset_file="data/mcp_dataset.json":
    @echo "Running complete training workflow"
    just check-gpu
    just download-devstral
    just finetune-model models/devstral {{dataset_file}} models/mcp-finetuned
    just evaluate-model models/mcp-finetuned data/mcp_eval.json