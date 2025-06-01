#!/bin/bash
# Script to download models from Hugging Face
# Usage: ./scripts/download_model.sh MODEL_NAME [OUTPUT_DIR]

set -e

# Default values
MODEL_NAME=$1
OUTPUT_DIR=${2:-"models"}

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME is required"
  echo "Usage: $0 MODEL_NAME [OUTPUT_DIR]"
  echo "Example: $0 mistralai/Mistral-7B-v0.1 ./models"
  exit 1
fi

echo "Downloading model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download model using huggingface-cli
echo "Starting download using huggingface-cli..."
huggingface-cli download --resume-download "$MODEL_NAME" --local-dir "$OUTPUT_DIR/$MODEL_NAME"

# Check if download was successful
if [ $? -eq 0 ]; then
  echo "Model downloaded successfully to: $OUTPUT_DIR/$MODEL_NAME"
else
  echo "Error: Failed to download model"
  exit 1
fi

# List downloaded files
echo "Downloaded files:"
ls -la "$OUTPUT_DIR/$MODEL_NAME"

echo "Download complete!"