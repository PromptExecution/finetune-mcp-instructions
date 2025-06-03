
# Justfile for finetune-mcp-instructions

# Validate the MCP dataset
validate-dataset data_path:
    python src/validate_dataset.py --data-path {{data_path}}

validate-example:
    just validate-dataset data/mcp_dataset.json

# Preprocess the MCP dataset
preprocess-dataset input_file output_file:
    @echo "Preprocessing dataset from {{input_file}} to {{output_file}}"
    python src/preprocess_dataset.py --input-file "{{input_file}}" --output-file "{{output_file}}"


# Run the entire fine-tuning pipeline
run-finetuning num_examples:
    @just generate-dataset {{num_examples}} ./mcp_dataset.json
    @just preprocess-dataset ./mcp_dataset.json ./preprocessed_mcp_dataset.json
    @just prepare-training ./finetune_output
    @just finetune mistralai/Devstral-Small-2505 ./data/preprocessed_mcp_dataset.json ./finetune_output
    @just evaluate mistralai/Devstral-Small-2505 finetune_output ./data/eval_samples.json

# Default recipe to show help
default:
    @just --list

# Generate the MCP dataset
generate-dataset num_examples output_file:
    python src/generate_dataset.py --num-examples {{num_examples}} --output-file {{output_file}}

generate-dataset-example:
    just generate-dataset 50 mcp_dataset.json

# Download model from Hugging Face
download-model model_name:
    ./scripts/download_model.sh {{model_name}}

download-example:
    just download-model mistralai/Devstral-Small-2505

# Prepare training script for fine-tuning
prepare-training output_dir:
    ./scripts/prepare_training.sh {{output_dir}}

# Run fine-tuning on the model
# Remove the old finetune command

finetune-example:
    just finetune mistralai/Devstral-Small-2505 data/preprocessed_mcp_dataset.json finetune_output

# Serve model using VLLM
serve-model model_path port:
    ./scripts/serve_model.sh {{model_path}} {{port}}

serve-model-example:
    just serve-model finetune_output 8000

# Run evaluation comparing base model vs fine-tuned model
evaluate base_model finetuned_model test_data:
    uv run -- ./scripts/evaluate_models.sh {{base_model}} {{finetuned_model}} {{test_data}}

evaluate-examples:
    just evaluate mistralai/Devstral-Small-2505 finetune_output data/eval_samples.json
    # Evaluate the usage of the fine-tuned model

# Generate the MCP dataset for fine-tuning
generate-mcp-dataset num_examples output_file:
    # Example: just generate-mcp-dataset num_examples=50 output_file=mcp_dataset.json
    python src/generate_dataset.py --num-examples {{num_examples}} --output-file {{output_file}}

# Fine-tune the model with MCP knowledge
finetune-mcp output_dir model_name data_path:
    ./scripts/internal_finetune_mcp_logic.sh "{{output_dir}}" "{{model_name}}" "{{data_path}}"


# Example command to run the entire process for Devstral
example-devstral:
    @just generate-dataset 50 mcp_dataset.json
    @just preprocess-dataset ./data/mcp_dataset.json ./data/preprocessed_mcp_dataset.json
    @just prepare-training finetune_output
    @just finetune mistralai/Devstral-Small-2505 ./data/preprocessed_mcp_dataset.json ./finetune_output
    @just evaluate mistralai/Devstral-Small-2505 ./finetune_output ./data/eval_samples.json
    @echo run the model with: just serve-model finetune_output 8000


# Command to run the fine-tuning UV tool
# Command to run the fine-tuning process using uv
finetune model_name data_path output_dir:
    bash -c ". .venv/bin/activate && uv finetune_model.py {{model_name}} {{data_path}} {{output_dir}}"
