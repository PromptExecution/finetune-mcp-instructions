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