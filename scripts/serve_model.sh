#!/bin/bash
# Script to serve a model using VLLM
# Usage: ./scripts/serve_model.sh MODEL_PATH [PORT]

set -e

# Default values
MODEL_PATH=${1:-"finetune_output/final"}
PORT=${2:-"8000"}

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path does not exist: $MODEL_PATH"
  exit 1
fi

echo "Starting VLLM server for model: $MODEL_PATH"
echo "Port: $PORT"

# Check if VLLM is installed
if ! command -v python -c "import vllm" &> /dev/null; then
  echo "VLLM is not installed. Would you like to install it now? (y/n)"
  read -r INSTALL
  if [[ "$INSTALL" == "y" || "$INSTALL" == "Y" ]]; then
    pip install vllm
  else
    echo "Error: VLLM is required to serve the model"
    exit 1
  fi
fi

# Create a config file for the server
CONFIG_DIR="$(dirname "$MODEL_PATH")/configs"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/vllm_config.json"

cat > "$CONFIG_FILE" << EOL
{
  "model": "${MODEL_PATH}",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 8192,
  "dtype": "bfloat16",
  "seed": 42
}
EOL

echo "Created VLLM configuration at: $CONFIG_FILE"

# Create a Python script for the VLLM server with MCP-specific handlers
SERVER_SCRIPT="$CONFIG_DIR/vllm_server.py"

cat > "$SERVER_SCRIPT" << 'EOL'
#!/usr/bin/env python3
"""
VLLM Server for MCP-compatible model serving.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union, Any

import torch
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request, Response, status, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    """Input model for generation requests."""
    prompt: str
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False

class MCPRequest(BaseModel):
    """Input model for MCP-specific requests."""
    instruction: str
    metadata: Optional[Dict[str, Any]] = None
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

def create_app(model_config_path: str):
    """Create the FastAPI application with the given model configuration."""
    # Load model configuration
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Initialize VLLM model
    logger.info(f"Initializing VLLM with model: {model_config['model']}")
    llm = LLM(
        model=model_config["model"],
        tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9),
        max_model_len=model_config.get("max_model_len", 8192),
        dtype=model_config.get("dtype", "auto"),
        seed=model_config.get("seed", 42),
        trust_remote_code=model_config.get("trust_remote_code", True)
    )

    # Initialize FastAPI
    app = FastAPI(title="VLLM MCP Server")

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "model": model_config["model"]}

    @app.post("/generate")
    async def generate(request: GenerationRequest):
        """Generate text based on the provided prompt."""
        logger.info(f"Received generation request with prompt length: {len(request.prompt)}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )

        # Generate response
        outputs = llm.generate(request.prompt, sampling_params)

        # Extract generated text
        generated_text = outputs[0].outputs[0].text

        return {
            "generated_text": generated_text,
            "tokens_generated": len(outputs[0].outputs[0].token_ids)
        }

    @app.post("/mcp/generate")
    async def mcp_generate(request: MCPRequest):
        """Generate MCP-compatible responses for instructions."""
        logger.info(f"Received MCP request: {request.instruction[:50]}...")

        # Format prompt for MCP-style responses
        formatted_prompt = f"<|user|>\n{request.instruction}\n<|assistant|>"

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # Generate response
        outputs = llm.generate(formatted_prompt, sampling_params)

        # Extract generated text
        generated_text = outputs[0].outputs[0].text

        return {
            "instruction": request.instruction,
            "completion": generated_text,
            "metadata": request.metadata or {}
        }

    @app.get("/info")
    async def model_info():
        """Return information about the loaded model."""
        model_path = model_config["model"]

        return {
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "config": model_config
        }

    return app

def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="Run VLLM server")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    args = parser.parse_args()

    # Create and run the server
    app = create_app(args.config)

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
EOL

chmod +x "$SERVER_SCRIPT"

# Start the server
echo "Starting VLLM server..."
python "$SERVER_SCRIPT" --config "$CONFIG_FILE" --port "$PORT"