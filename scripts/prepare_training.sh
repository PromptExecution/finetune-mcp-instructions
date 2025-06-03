#!/bin/bash
# Script to prepare the training environment for fine-tuning
# Usage: ./scripts/prepare_training.sh [OUTPUT_DIR] [MODEL_NAME_FOR_CONFIG] [DATA_PATH_FOR_CONFIG]

set -e

# Default values
# Raw arguments that might come from 'just' like key=value
_raw_output_dir_arg=${1}
_raw_model_name_arg=${2}
_raw_data_path_arg=${3}

# Strip 'key=' prefix. If no prefix, original value is retained.
# If the arg was not 'key=value' but just 'value', this will result in 'value'.
# If the arg was 'key=', this will result in an empty string.
OUTPUT_DIR_ARG=$(echo "${_raw_output_dir_arg}" | sed -e 's/^[^=]*=//' -e 's/"//g')
MODEL_NAME_ARG=$(echo "${_raw_model_name_arg}" | sed -e 's/^[^=]*=//' -e 's/"//g')
DATA_PATH_ARG=$(echo "${_raw_data_path_arg}" | sed -e 's/^[^=]*=//' -e 's/"//g')

# If the original raw argument was identical to the stripped argument,
# it means there was no "key=" prefix.
# If they are different, AND the stripped argument is now empty,
# it means the input was "key=". In this case, we want to trigger the default.
# So, if $OUTPUT_DIR_ARG is empty here, the default on the next line will apply.
# This logic is implicitly handled by the :- default mechanism if the variable is empty.

OUTPUT_DIR=${OUTPUT_DIR_ARG:-"finetune_output"}
MODEL_NAME_FOR_CONFIG=${MODEL_NAME_ARG:-"mistralai/Devstral-Small-2505"}
DATA_PATH_FOR_CONFIG=${DATA_PATH_ARG:-"data/preprocessed_mcp_dataset.json"}

echo "Preparing training environment"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/configs"

# Check for required Python packages
echo "Checking required Python packages..."
REQUIRED_PACKAGES=(
  "torch"
  "transformers"
  "peft"
  "accelerate"
  "datasets"
  "trl"
)

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
  python -c "import $package" >/dev/null 2>&1 || MISSING_PACKAGES+=("$package")
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
  echo "Missing required Python packages: ${MISSING_PACKAGES[*]}"
  echo "Please install them using: pip install ${MISSING_PACKAGES[*]}"
  echo "Would you like to install them now? (y/n)"
  read -r INSTALL
  if [[ "$INSTALL" == "y" || "$INSTALL" == "Y" ]]; then
    echo "Attempting to install missing packages..."
    uv pip install "${MISSING_PACKAGES[@]}" || { echo "Failed to install packages. Please install them manually."; exit 1; }
  else
    echo "Warning: Missing packages may cause training to fail"
  fi
fi

# Create a default training configuration file
CONFIG_FILE="$OUTPUT_DIR/configs/training_config.json"
echo "Creating default training configuration..."
cat > "$CONFIG_FILE" << EOL
{
  "model_config": {
    "model_name_or_path": "$MODEL_NAME_FOR_CONFIG",
    "trust_remote_code": true
  },
  "training_config": {
    "output_dir": "$OUTPUT_DIR",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "fp16": true,
    "logging_steps": 10,
    "optim": "paged_adamw_8bit",
    "save_strategy": "steps",
    "save_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "max_grad_norm": 0.3
  },
  "peft_config": {
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "r": 8,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "data_config": {
    "train_file": "$DATA_PATH_FOR_CONFIG",
    "max_seq_length": 2048,
    "preprocessing_num_workers": 4
  }
}
EOL

echo "Created training configuration at: $CONFIG_FILE"
echo "Environment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review and modify the training configuration if needed"
echo "2. Run the fine-tuning script with: just finetune MODEL_NAME DATA_PATH $OUTPUT_DIR"