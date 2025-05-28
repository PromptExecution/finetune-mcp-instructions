#!/usr/bin/env python3
"""
GPU Availability Checker for MCP Fine-tuning

This script checks CUDA availability, version, and GPU memory.
It's used to verify the environment is correctly set up for training.
"""

import sys
import torch
import logging
import importlib.metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_torch_version():
    """Check PyTorch version."""
    torch_version = torch.__version__
    logger.info(f"PyTorch version: {torch_version}")
    return torch_version

def check_cuda_availability():
    """Check CUDA availability."""
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        logger.info(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        logger.info(f"Available GPU devices: {device_count}")

        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  Compute capability: {props.major}.{props.minor}")
    else:
        logger.warning("No CUDA-capable GPU found. Training will be slow on CPU only.")

    return cuda_available

def check_required_libs():
    """Check required libraries for training."""
    required_libs = {
        "transformers": "Hugging Face Transformers",
        "peft": "Parameter-Efficient Fine-Tuning",
        "bitsandbytes": "Quantization",
        "accelerate": "Distributed training",
        "datasets": "Dataset handling"
    }

    logger.info("Checking required libraries:")
    all_available = True

    for lib, description in required_libs.items():
        try:
            version = importlib.metadata.version(lib)
            logger.info(f"✓ {lib} (v{version}): {description}")
        except importlib.metadata.PackageNotFoundError:
            logger.error(f"✗ {lib}: Not installed")
            all_available = False

    return all_available

def check_gpu_memory_sufficient(min_gb=24):
    """Check if GPU has sufficient memory for training."""
    if not torch.cuda.is_available():
        logger.warning(f"Cannot check GPU memory as CUDA is not available")
        return False

    device = torch.device("cuda:0")
    memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    if memory_gb >= min_gb:
        logger.info(f"✓ GPU memory is sufficient: {memory_gb:.2f} GB (minimum required: {min_gb} GB)")
        return True
    else:
        logger.warning(f"✗ GPU memory may be insufficient: {memory_gb:.2f} GB (minimum recommended: {min_gb} GB)")
        return False

def main():
    """Check GPU availability and related requirements."""
    logger.info("Checking GPU availability for MCP model fine-tuning")

    # Check PyTorch version
    check_torch_version()

    # Check CUDA availability
    cuda_available = check_cuda_availability()

    # Check required libraries
    libs_available = check_required_libs()

    # Check GPU memory
    if cuda_available:
        memory_sufficient = check_gpu_memory_sufficient(24)  # RTX 3090 has 24GB
    else:
        memory_sufficient = False

    # Report overall status
    if cuda_available and libs_available and memory_sufficient:
        logger.info("✅ Environment is ready for fine-tuning")
        return 0
    else:
        if not cuda_available:
            logger.warning("⚠️ CUDA not available - training will be very slow")
        if not libs_available:
            logger.warning("⚠️ Missing required libraries - please install them")
        if not memory_sufficient and cuda_available:
            logger.warning("⚠️ GPU memory may be insufficient for optimal performance")

        logger.warning("Environment check completed with warnings")
        return 1

if __name__ == "__main__":
    sys.exit(main())