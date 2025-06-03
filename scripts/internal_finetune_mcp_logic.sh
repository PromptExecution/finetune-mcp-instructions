#!/bin/bash -e

# Script to handle the internal logic of the finetune-mcp just recipe

# Raw arguments from just (could be key=value or just value)
# Ensure correct mapping of positional arguments from just "$@"
# $1 should be output_dir=..., $2 model_name=..., $3 data_path=...
_raw_output_dir_arg="$1"
_raw_model_name_arg="$2"
_raw_data_path_arg="$3"

# Clean the arguments
output_dir_cleaned_sh=`echo "$_raw_output_dir_arg" | sed -e 's/^[^=]*=//' -e 's/"//g'`
model_name_cleaned_sh=`echo "$_raw_model_name_arg" | sed -e 's/^[^=]*=//' -e 's/"//g'`
data_path_cleaned_sh=`echo "$_raw_data_path_arg"  | sed -e 's/^[^=]*=//' -e 's/"//g'`

# Echo for debugging (optional, can be removed)
echo "--- Internal Script Start ---"
echo "Raw output dir arg: $_raw_output_dir_arg"
echo "Cleaned output dir: $output_dir_cleaned_sh"
echo "Raw model name arg: $_raw_model_name_arg"
echo "Cleaned model name: $model_name_cleaned_sh"
echo "Raw data path arg: $_raw_data_path_arg"
echo "Cleaned data path: $data_path_cleaned_sh"
echo "---------------------------"

echo "Target configuration file: $output_dir_cleaned_sh/configs/training_config.json"
echo "Always running preparation script to ensure fresh config: ./scripts/prepare_training.sh $output_dir_cleaned_sh $model_name_cleaned_sh $data_path_cleaned_sh";
./scripts/prepare_training.sh "$output_dir_cleaned_sh" "$model_name_cleaned_sh" "$data_path_cleaned_sh";
echo "Running fine-tuning script: scripts/run_mcp_finetune.py"
uv run python scripts/run_mcp_finetune.py --config "$output_dir_cleaned_sh/configs/training_config.json"

echo "--- Internal Script End ---"