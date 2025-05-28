# Justfile Updates for Hugging Face Integration

The following changes should be made to the justfile to support Hugging Face CLI integration and model fine-tuning workflows.

## 1. Updated Setup Recipe

Add Hugging Face CLI installation to the setup process:

```justfile
# Setup the virtual environment with all dependencies
setup:
    @echo "Setting up virtual environment with uv"
    uv venv
    uv pip install -r requirements.txt
    @echo "Installing Hugging Face CLI"
    uv pip install --upgrade huggingface_hub
    @echo "Run 'huggingface-cli login' to authenticate with Hugging Face"
```

## 2. New Hugging Face Authentication Recipe

Add a specific recipe for Hugging Face authentication:

```justfile
# Authenticate with Hugging Face
hf-login:
    @echo "Authenticating with Hugging Face"
    huggingface-cli login
```

## 3. Model Download Recipes

Add recipes for downloading base models:

```justfile
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
```

## 4. Training Recipes

Add recipes for model fine-tuning:

```justfile
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
```

## 5. GPU Verification Recipe

Add a recipe to verify GPU availability:

```justfile
# Verify GPU availability
check-gpu:
    @echo "Checking GPU availability"
    {{python}} -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB' if torch.cuda.is_available() else 'No GPU')"
```

## 6. Combined Training Workflow Recipe

Add a recipe for the complete training workflow:

```justfile
# Complete training workflow
training-workflow dataset_file="data/mcp_dataset.json":
    @echo "Running complete training workflow"
    just check-gpu
    just download-devstral
    just finetune-model models/devstral {{dataset_file}} models/mcp-finetuned
    just evaluate-model models/mcp-finetuned data/mcp_eval.json
```

These updates will enable the next phase of the project, which focuses on model training and evaluation.