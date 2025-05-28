# MCP Dataset Generator Justfile
# This file contains recipes for common tasks using uv instead of direct Python

# Set the default Python interpreter to use uv
python := "uv run python"

# Configuration variables - centralized for easier management
base_model_dir := "models/devstral"
output_model_dir := "models/mcp-finetuned"
dataset_dir := "data"
dataset_file := dataset_dir + "/mcp_dataset.json"
eval_dataset := dataset_dir + "/mcp_eval.json"
tensorboard_dir := "runs/tensorboard"

# Default recipe to show available commands with descriptions
default:
    @echo "MCP Fine-tuning Toolkit"
    @echo "======================="
    @just --list

# Setup the virtual environment with all dependencies
setup:
    @echo "Setting up virtual environment with uv"
    uv venv
    uv pip install -r requirements.txt
    @echo "Installing Hugging Face CLI"
    uv tool install --upgrade huggingface_hub[cli]
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

# Fine-tune model with PEFT/LoRA with configurable parameters
# Uses multiple chained commands for better workflow
finetune-model model_dir=base_model_dir dataset_file=dataset_file output_dir=output_model_dir lora_r="16" lora_alpha="32" learning_rate="2e-4" epochs="3" batch_size="1" grad_accum="8":
    #!/usr/bin/env bash
    # Tutorial: This recipe performs LoRA fine-tuning on the base model with the following steps:
    # 1. Check if output directory exists and create it if needed
    # 2. Verify if model exists before starting the expensive fine-tuning process
    # 3. Run the fine-tuning with carefully selected hyperparameters
    # 4. Save training logs to a date-stamped file for tracking progress

    @echo "Step 1/4: Preparing output directory {{output_dir}}"
    mkdir -p {{output_dir}}

    @echo "Step 2/4: Verifying base model availability in {{model_dir}}"
    if [ ! -d "{{model_dir}}" ]; then
        echo "Error: Base model directory not found. Please run 'just download-devstral' first."
        exit 1
    fi

    @echo "Step 3/4: Fine-tuning model with the following configuration:"
    @echo "  - Base model: {{model_dir}}"
    @echo "  - Dataset: {{dataset_file}}"
    @echo "  - LoRA rank: {{lora_r}}"
    @echo "  - Learning rate: {{learning_rate}}"
    @echo "  - Epochs: {{epochs}}"
    @echo "  - Batch size: {{batch_size}} with gradient accumulation: {{grad_accum}}"

    # Create logs directory if it doesn't exist
    mkdir -p logs

    @echo "Step 4/4: Starting training process (logs will be saved to logs/training_$(date +%Y%m%d_%H%M%S).log)"
    {{python}} src/train.py \
        --model-dir {{model_dir}} \
        --dataset {{dataset_file}} \
        --output-dir {{output_dir}} \
        --lora-rank {{lora_r}} \
        --lora-alpha {{lora_alpha}} \
        --learning-rate {{learning_rate}} \
        --num-epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --gradient-accumulation {{grad_accum}} \
        --tensorboard-dir {{tensorboard_dir}} \
        --save-steps 100 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Evaluate model performance with detailed metrics
# Chain commands to provide rich evaluation output
evaluate-model model_dir=output_model_dir eval_dataset=eval_dataset output_file="":
    #!/usr/bin/env bash
    # Tutorial: This recipe evaluates the fine-tuned model by:
    # 1. Verifying the model and evaluation dataset exist
    # 2. Running evaluation with detailed metrics
    # 3. Saving results to a file if requested

    @echo "Step 1/3: Verifying model and dataset availability"
    if [ ! -d "{{model_dir}}" ]; then
        echo "Error: Model directory not found at {{model_dir}}"
        exit 1
    fi

    if [ ! -f "{{eval_dataset}}" ]; then
        echo "Error: Evaluation dataset not found at {{eval_dataset}}"
        exit 1
    fi

    @echo "Step 2/3: Evaluating model at {{model_dir}} with dataset {{eval_dataset}}"

    # Determine the command based on whether an output file was specified
    @echo "Step 3/3: Running evaluation and collecting metrics"
    if [ -n "{{output_file}}" ]; then
        mkdir -p $(dirname {{output_file}})
        {{python}} src/evaluate.py \
            --model-dir {{model_dir}} \
            --dataset {{eval_dataset}} \
            --output-file {{output_file}} \
            --detailed-metrics \
            --format-report
    else
        {{python}} src/evaluate.py \
            --model-dir {{model_dir}} \
            --dataset {{eval_dataset}} \
            --detailed-metrics
    fi

# Compare base and fine-tuned models with statistical significance testing
compare-models base_model=base_model_dir finetuned_model=output_model_dir eval_dataset=eval_dataset output_format="markdown" significance_level="0.05":
    #!/usr/bin/env bash
    # Tutorial: This recipe performs a thorough comparison between models:
    # 1. Verifies both models exist
    # 2. Runs comparative evaluation on same dataset
    # 3. Produces detailed statistics with significance tests
    # 4. Outputs results in chosen format (markdown, json, csv)

    @echo "Step 1/3: Verifying model directories"
    if [ ! -d "{{base_model}}" ] || [ ! -d "{{finetuned_model}}" ]; then
        echo "Error: One or both model directories not found."
        echo "Base model: {{base_model}}"
        echo "Fine-tuned model: {{finetuned_model}}"
        exit 1
    fi

    @echo "Step 2/3: Preparing comparison environment"
    mkdir -p reports
    output_file="reports/model_comparison_$(date +%Y%m%d_%H%M%S).{{output_format}}"

    @echo "Step 3/3: Running comparative analysis with statistical significance tests"
    {{python}} src/compare.py \
        --base-model {{base_model}} \
        --finetuned-model {{finetuned_model}} \
        --dataset {{eval_dataset}} \
        --output-file ${output_file} \
        --output-format {{output_format}} \
        --significance-level {{significance_level}} \
        --show-examples \
        --detailed-metrics

    @echo "Comparison report saved to ${output_file}"

# Verify GPU availability with extended diagnostics
check-gpu memory_threshold="10":
    #!/usr/bin/env bash
    # Tutorial: This recipe performs comprehensive GPU checks:
    # 1. Verifies CUDA availability through PyTorch
    # 2. Runs extended diagnostics to check memory and compute capability
    # 3. Validates if GPU meets minimum requirements for training

    @echo "Step 1/3: Checking basic GPU availability"
    {{python}} src/check_gpu.py --verbose

    @echo "Step 2/3: Running extended diagnostics"
    {{python}} -c "import torch; print(f'\nCUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB' if torch.cuda.is_available() else 'No GPU')"

    @echo "Step 3/3: Validating GPU meets minimum requirements"
    {{python}} -c "import torch; memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0; print(f'\nGPU has {memory:.2f}GB RAM'); threshold = {{memory_threshold}}; print(f'Minimum required: {threshold}GB RAM'); print(f'Status: {\"✅ Sufficient\" if memory >= threshold else \"❌ Insufficient\"} memory for training')"

# Complete training workflow with chained commands and error handling
training-workflow dataset_file=dataset_file lora_r="16" learning_rate="2e-4" epochs="3":
    #!/usr/bin/env bash
    # Tutorial: This recipe orchestrates the complete fine-tuning pipeline:
    # 1. Validates GPU capabilities and availability
    # 2. Downloads or verifies the base model
    # 3. Fine-tunes with optimized hyperparameters
    # 4. Evaluates the resulting model
    # 5. Generates comparison reports against baseline

    @echo "======== MCP FINE-TUNING WORKFLOW ========"

    @echo "\n[Phase 1/5] Verifying hardware capabilities"
    just check-gpu 10 || { echo "Error: GPU requirements not met. Aborting workflow"; exit 1; }

    @echo "\n[Phase 2/5] Preparing base model"
    just download-devstral || { echo "Error: Failed to download base model. Aborting workflow"; exit 1; }

    @echo "\n[Phase 3/5] Fine-tuning model with LoRA (rank={{lora_r}}, lr={{learning_rate}}, epochs={{epochs}})"
    just finetune-model {{base_model_dir}} {{dataset_file}} {{output_model_dir}} {{lora_r}} "32" {{learning_rate}} {{epochs}} || { echo "Error: Fine-tuning failed. Aborting workflow"; exit 1; }

    @echo "\n[Phase 4/5] Evaluating fine-tuned model"
    evaluation_file="reports/evaluation_$(date +%Y%m%d_%H%M%S).md"
    just evaluate-model {{output_model_dir}} {{eval_dataset}} ${evaluation_file} || { echo "Warning: Evaluation encountered issues"; }

    @echo "\n[Phase 5/5] Generating comparison report against baseline"
    just compare-models {{base_model_dir}} {{output_model_dir}} {{eval_dataset}} "markdown" || { echo "Warning: Comparison encountered issues"; }

    @echo "\n✅ Training workflow completed successfully!"
    @echo "Tensorboard logs available. Run: tensorboard --logdir={{tensorboard_dir}}"
# Advanced fine-tuning commands for weight optimization

# Hyperparameter tuning for optimizing LoRA configuration
tune-hyperparams model_dir=base_model_dir dataset_file=dataset_file output_dir="models/tuning" trials="5":
    #!/usr/bin/env bash
    # Tutorial: This recipe performs hyperparameter tuning to find optimal LoRA settings:
    # 1. Creates a tuning directory structure
    # 2. Tests different hyperparameter combinations (rank, alpha, learning rate)
    # 3. Records results and identifies best configuration
    # 4. Generates a report of findings

    @echo "Step 1/4: Setting up hyperparameter tuning environment"
    mkdir -p {{output_dir}}/results

    @echo "Step 2/4: Preparing hyperparameter search space"
    ranks=(8 16 32)
    alphas=(16 32 64)
    learning_rates=(1e-4 2e-4 5e-4)

    @echo "Step 3/4: Running {{trials}} tuning trials with different configurations"
    trial=1
    results_file="{{output_dir}}/tuning_results.csv"
    echo "trial,rank,alpha,learning_rate,eval_loss,accuracy" > $results_file

    # Use a subset of data for faster tuning
    subset_data="{{output_dir}}/tuning_subset.json"
    {{python}} -c "import json; data = json.load(open('{{dataset_file}}')); json.dump({'examples': data['examples'][:100]}, open('$subset_data', 'w'))"

    # Run multiple trials with different hyperparameter combinations
    for rank in ${ranks[@]}; do
        for alpha in ${alphas[@]}; do
            for lr in ${learning_rates[@]}; do
                if [ $trial -le {{trials}} ]; then
                    echo "\nTrial $trial/${{trials}}: rank=$rank, alpha=$alpha, learning_rate=$lr"
                    trial_dir="{{output_dir}}/trial_${trial}"
                    mkdir -p $trial_dir

                    # Run a quick training pass with these parameters
                    {{python}} src/train.py \
                        --model-dir {{model_dir}} \
                        --dataset $subset_data \
                        --output-dir $trial_dir \
                        --lora-rank $rank \
                        --lora-alpha $alpha \
                        --learning-rate $lr \
                        --num-epochs 1 \
                        --batch-size 1 \
                        --quick-eval \
                        --save-steps 0 > "$trial_dir/log.txt"

                    # Extract metrics from log
                    eval_loss=$(grep "eval_loss" "$trial_dir/log.txt" | tail -1 | sed 's/.*eval_loss: \([0-9.]*\).*/\1/')
                    accuracy=$(grep "accuracy" "$trial_dir/log.txt" | tail -1 | sed 's/.*accuracy: \([0-9.]*\).*/\1/')

                    # Record results
                    echo "$trial,$rank,$alpha,$lr,$eval_loss,$accuracy" >> $results_file

                    trial=$((trial + 1))
                fi
            done
        done
    done

    @echo "Step 4/4: Analyzing results and identifying best configuration"
    {{python}} -c "import pandas as pd; data = pd.read_csv('$results_file'); best = data.sort_values('eval_loss').iloc[0]; print(f'\nBest configuration:\n  Rank: {best.rank}\n  Alpha: {best.alpha}\n  Learning rate: {best.learning_rate}\n  Eval loss: {best.eval_loss:.4f}')"

    @echo "\nComplete results saved to $results_file"
    @echo "To use the best configuration, run: just finetune-model {{model_dir}} {{dataset_file}} {{output_model_dir}} [rank] [alpha] [learning_rate]"

# Export fine-tuned model in various formats (merged, quantized)
export-model model_dir=output_model_dir output_path="exports/model" format="merged" quantize="4bit":
    #!/usr/bin/env bash
    # Tutorial: This recipe exports the fine-tuned model in different formats:
    # 1. Verifies the fine-tuned model exists
    # 2. Merges LoRA weights with base model if requested
    # 3. Applies quantization for smaller model size if requested
    # 4. Creates HuggingFace-compatible model directory with proper structure

    @echo "Step 1/4: Verifying model at {{model_dir}}"
    if [ ! -d "{{model_dir}}" ]; then
        echo "Error: Model directory not found at {{model_dir}}"
        exit 1
    fi

    @echo "Step 2/4: Preparing export directory"
    mkdir -p $(dirname {{output_path}})

    @echo "Step 3/4: Exporting model in {{format}} format with {{quantize}} quantization"
    {{python}} -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load configuration and determine model type
model_dir = '{{model_dir}}'
is_peft = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
base_model_path = PeftConfig.from_pretrained(model_dir).base_model_name_or_path if is_peft else model_dir

print(f'Base model: {base_model_path}')
print(f'Export format: {{format}}')
print(f'Quantization: {{quantize}}')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Process based on export format
if '{{format}}' == 'merged' and is_peft:
    print('Merging LoRA weights with base model...')
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map='auto',
        trust_remote_code=True
    )

    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, model_dir)

    # Merge weights - this combines the LoRA adapters with the base model
    model = model.merge_and_unload()

    print('Model successfully merged')
else:
    # Just load the model directly
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        trust_remote_code=True
    )
    print('Model loaded (no merging required)')

# Apply quantization if requested
if '{{quantize}}' == '4bit':
    print('Applying 4-bit quantization...')
    from transformers import BitsAndBytesConfig
    import torch

    # Save in 4-bit format
    model.save_pretrained('{{output_path}}',
        max_shard_size='2GB',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4'
        )
    )
elif '{{quantize}}' == '8bit':
    print('Applying 8-bit quantization...')
    # Save in 8-bit format
    model.save_pretrained('{{output_path}}', max_shard_size='2GB', load_in_8bit=True)
else:
    # Save in full precision
    model.save_pretrained('{{output_path}}', max_shard_size='2GB')

# Save the tokenizer
tokenizer.save_pretrained('{{output_path}}')
    "

    @echo "Step 4/4: Creating model card and metadata"
    cat > "{{output_path}}/README.md" << 'EOF'
# MCP Fine-tuned Model

This model was fine-tuned on MCP (Model Context Protocol) tasks using PEFT/LoRA techniques.

## Model Details
- **Base Model**: Devstral
- **Fine-tuning Method**: LoRA
- **Export Format**: {{format}} {{quantize}}
- **Date Created**: $(date +"%Y-%m-%d")

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/model")
model = AutoModelForCausalLM.from_pretrained("path/to/model")

# Generate MCP response
response = model.generate(tokenizer.encode("Find all Python files", return_tensors="pt"), max_length=200)
print(tokenizer.decode(response[0]))
```
EOF

    @echo "\nModel successfully exported to {{output_path}}"
    @echo "The exported model can be used with HuggingFace Transformers library"

# Visual metrics monitoring with real-time dashboard for training
monitor-training run_id="latest":
    #!/usr/bin/env bash
    # Tutorial: This recipe provides visual monitoring for training:
    # 1. Finds the latest tensorboard logs or uses specified run ID
    # 2. Launches tensorboard server for visualization
    # 3. Displays URL for accessing the dashboard

    @echo "Step 1/3: Locating tensorboard logs"
    if [ "{{run_id}}" = "latest" ]; then
        # Find the latest run directory
        latest_dir=$(find {{tensorboard_dir}} -maxdepth 1 -type d -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d" " || echo "{{tensorboard_dir}}")
        log_dir=$latest_dir
    else
        # Use the specified run ID
        log_dir="{{tensorboard_dir}}/{{run_id}}"
    fi

    @echo "Step 2/3: Setting up dashboard for $log_dir"
    if [ ! -d "$log_dir" ]; then
        echo "Warning: Log directory not found. Using base tensorboard directory."
        log_dir="{{tensorboard_dir}}"
    fi

    @echo "Step 3/3: Launching Tensorboard server"
    @echo "Access dashboard at: http://localhost:6006"
    tensorboard --logdir=$log_dir

# Run systematic test suite to evaluate model quality across dimensions
evaluate-quality model_dir=output_model_dir eval_datasets=eval_dataset categories="all":
    #!/usr/bin/env bash
    # Tutorial: This recipe performs comprehensive quality evaluation:
    # 1. Tests the model across different evaluation dimensions
    # 2. Generates quality metrics for each MCP server type
    # 3. Creates visual reports of model performance

    @echo "Step 1/4: Preparing quality evaluation environment"
    mkdir -p reports/quality
    report_dir="reports/quality/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $report_dir

    @echo "Step 2/4: Checking model availability"
    if [ ! -d "{{model_dir}}" ]; then
        echo "Error: Model directory not found at {{model_dir}}"
        exit 1
    fi

    @echo "Step 3/4: Running quality evaluation across dimensions"

    # If categories is "all", test all server types
    if [ "{{categories}}" = "all" ]; then
        categories="filesystem git github database custom"
    else
        categories="{{categories}}"
    fi

    for category in $categories; do
        echo "\nEvaluating $category MCP tasks..."
        {{python}} src/evaluate.py \
            --model-dir {{model_dir}} \
            --dataset {{eval_datasets}} \
            --filter-category $category \
            --output-file "$report_dir/${category}_report.md" \
            --detailed-metrics \
            --format-report
    done

    @echo "Step 4/4: Generating consolidated quality report"
    {{python}} -c "
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gather metrics from individual reports
report_dir = '$report_dir'
metrics = {}

for filename in os.listdir(report_dir):
    if not filename.endswith('_report.md'):
        continue

    category = filename.split('_')[0]
    metrics[category] = {}

    with open(os.path.join(report_dir, filename), 'r') as f:
        content = f.read()

    # Extract accuracy
    accuracy_match = re.search(r'Task Success Rate: ([0-9.]+)%', content)
    if accuracy_match:
        metrics[category]['accuracy'] = float(accuracy_match.group(1))

    # Extract XML validity
    xml_match = re.search(r'XML Validity: ([0-9.]+)%', content)
    if xml_match:
        metrics[category]['xml_validity'] = float(xml_match.group(1))

    # Extract parameter correctness
    param_match = re.search(r'Parameter Correctness: ([0-9.]+)%', content)
    if param_match:
        metrics[category]['param_correctness'] = float(param_match.group(1))

# Create radar chart
if metrics:
    categories = list(metrics.keys())
    metrics_types = ['accuracy', 'xml_validity', 'param_correctness']

    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Number of categories
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialize plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Draw each metric
    colors = ['b', 'r', 'g']
    for i, metric in enumerate(metrics_types):
        values = [metrics[cat].get(metric, 0) for cat in categories]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, colors[i], linewidth=1, linestyle='solid', label=metric)
        ax.fill(angles, values, colors[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance by MCP Server Type')

    # Save the figure
    plt.savefig(os.path.join(report_dir, 'quality_radar.png'))

    # Create summary markdown
    with open(os.path.join(report_dir, 'quality_summary.md'), 'w') as f:
        f.write('# Model Quality Evaluation Summary\n\n')
        f.write(f'Date: {pd.Timestamp.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n')
        f.write('## Performance by Category\n\n')

        f.write('| Category | Task Success | XML Validity | Parameter Correctness |\n')
        f.write('|----------|-------------|-------------|---------------------|\n')

        for cat in categories:
            acc = metrics[cat].get('accuracy', 0)
            xml = metrics[cat].get('xml_validity', 0)
            param = metrics[cat].get('param_correctness', 0)
            f.write(f'| {cat} | {acc:.1f}% | {xml:.1f}% | {param:.1f}% |\n')

        f.write('\n\n![Performance Radar Chart](quality_radar.png)\n')

print(f'Quality evaluation complete. Summary available at {report_dir}/quality_summary.md')
    "

    @echo "\nQuality evaluation complete. Reports available in $report_dir"