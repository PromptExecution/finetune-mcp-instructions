# MCP Dataset Creation Plan

- **Status**: Dataset created and validated successfully. A total of 20 examples were generated, covering various server types and complexities.

## 1. Dataset Overview
- **Objective**: Create a supervised dataset of 50-100 high-quality instruction-completion pairs for fine-tuning LLMs on MCP-style tasks
- **Format**: JSON-based dataset following Hugging Face Dataset conventions
- **Structure**: Each example will contain:
  - Instruction (user query/command)
  - Completion (expected model response)
  - Metadata (source, MCP server type, tools used)

## 2. MCP Server Categories to Cover
We'll create examples based on these reference MCP servers:

1. **Filesystem Server**
   - File operations (read, write, search)
   - Directory operations (list, create)
   - Path manipulations

2. **Git Server**
   - Repository operations (clone, status)
   - Commit history analysis
   - Diff viewing and understanding
   - Branch management

3. **GitHub Server**
   - Issue tracking
   - Pull request interactions
   - Repository exploration
   - Code search

4. **Database Server**
   - Query execution
   - Schema exploration
   - Data visualization requests

5. **Custom Tool Servers**
   - Weather data retrieval
   - Calculator functions
   - Image manipulation tools

## 3. Data Collection Methodology

### 3.1 Source Material Analysis
- Extract patterns from MCP reference implementations on GitHub
- Analyze documentation of existing MCP servers
- Review common code-related queries

### 3.2 Example Creation Process
1. **Task Identification**: Identify common tasks performed with each MCP server
2. **Instruction Formulation**: Create natural language instructions requesting these tasks
3. **Completion Generation**: Develop proper completions that follow MCP formatting conventions
4. **Validation**: Review examples for accuracy and formatting consistency

### 3.3 Example Distribution
- **Simple queries**: 30% (basic tool usage with single operations)
- **Multi-step interactions**: 40% (tasks requiring 2-3 tool calls)
- **Complex scenarios**: 30% (tasks requiring multiple tools, reasoning, or custom processing)

## 4. Data Formatting Specifications

### 4.1 Instruction Format
- Natural language instructions referencing MCP servers and tools
- Range from simple to complex requests
- Include context and constraints as needed

### 4.2 Completion Format
- Structured responses showing proper MCP tool usage
- Clear delineation of reasoning and actions
- Proper error handling when appropriate

### 4.3 Schema (Example)
```json
{
  "examples": [
    {
      "instruction": "Using the Git MCP server, find the commit history of main.py file from the last week",
      "completion": "I'll use the Git MCP server to find the commit history of main.py from the last week.\n\n[MCP_TOOL_CALL format here showing proper XML structure]",
      "metadata": {
        "server": "git",
        "tool": "log",
        "complexity": "simple"
      }
    }
  ]
}
```

## 5. Implementation Plan

### 5.1 Environment Setup
- Create Python environment with necessary libraries
- Set up data processing scripts and validation tools
- Establish Git repositories for storing and versioning the dataset

### 5.2 Development Phases
1. **Research Phase** (Week 1)
   - Analyze MCP reference implementations
   - Document common patterns and use cases
   - Define dataset schema and validation criteria

2. **Example Creation** (Weeks 2-3)
   - Generate initial set of instructions (20-30 examples)
   - Review and refine for quality and diversity
   - Expand to full prototype dataset (50-100 examples)

3. **Dataset Processing & Validation** (Week 4)
   - Convert to appropriate format for fine-tuning (e.g., using `src/preprocess_dataset.py`)
   - Implement loss masking for completion-only learning (handled by TRL's `DataCollatorForCompletionOnlyLM`)
   - Validate token lengths and formatting consistency

4. **Model Fine-Tuning** (Week 5-6)
   - Prepare fine-tuning scripts and environment.
   - Execute fine-tuning runs on the Devstral model with the MCP dataset.
   - Monitor and log training.

5. **Performance Evaluation** (Week 7)
   - Define and prepare evaluation tasks/prompts.
   - Run base and fine-tuned models on evaluation tasks.
   - Collect performance data and model outputs.

6. **Results Analysis & Reporting** (Week 8)
   - Analyze quantitative and qualitative data.
   - Verify performance improvements.
   - Document findings and plan next steps or iterations.

## 6. Quality Assurance Measures

### 6.1 Data Validation
- Check for syntactic correctness of all examples
- Ensure instructions are clear and unambiguous
- Verify completions follow MCP formatting conventions

### 6.2 Diversity Checks
- Balance between different server types
- Include various complexity levels
- Cover different types of operations per server

### 6.3 Technical Validation
- Token count validation (stay within model context limits)
- Check for proper JSON formatting
- Verify all XML tags are properly closed

## 7. Dataset Format Considerations

### 7.1 Loss Masking Setup
- Implement token masking to focus learning on completions
- Use HuggingFace's DataCollatorForCompletionOnlyLM
- Set prompt tokens to -100 to be ignored during training

### 7.2 Tokenization Strategy
- Process all examples using the target model's tokenizer
- Handle special tokens appropriately
- Ensure consistent formatting

## 8. Delivery & Documentation

### 8.1 Final Deliverables
- JSON dataset file(s) with instruction-completion pairs
- Documentation explaining dataset structure and usage
- Sample code for loading and preprocessing the dataset
- Quality report and statistics

### 8.2 Initial Model Testing
- Perform test fine-tuning runs on small subsets
- Evaluate output quality on validation prompts
- Make iterative improvements based on initial results

## 9. Workflow Diagram

```mermaid
graph TD
    A[Analyze MCP Reference Servers] --> B[Define Example Categories]
    B --> C[Create Initial Examples]
    C --> D[Review & Refine Examples]
    D --> E[Expand to Full Dataset]
    E --> F[Format & Process Dataset]
    F --> G[Validate Dataset Quality]
    G --> H[Package for Fine-tuning]
    H --> I[Initial Dataset Testing & Refinement]
    I --> J[Prepare Fine-Tuning Scripts & Environment]
    J --> K[Execute Fine-Tuning Runs (Hugging Face / Axolotl)]
    K --> L[Define Evaluation Tasks & Metrics]
    L --> M[Run Base & Fine-Tuned Models on Evaluation Tasks]
    M --> N[Collect & Compare Performance Data]
    N --> O[Analyze Results & Verify MCP Improvement]
    O --> P[Document Findings & Iterate if Necessary]
```

## 10. Model Fine-Tuning

- **Objective**: Fine-tune the Devstral model using the prepared MCP dataset to enhance its capabilities on MCP-style tasks.
- **Primary Tooling**: Hugging Face Transformers library (specifically PEFT/QLoRA for efficient fine-tuning), and associated CLI tools. The existing `justfile` commands (`finetune-mcp`, `finetune`) will be leveraged.
- **Contingency Plan**: If significant issues arise with Hugging Face, pivot to [Axolotl](https://axolotl.ai/) as an alternative fine-tuning framework.
- **Steps**:
    1.  **Script Preparation**:
        -   Verify and adapt existing fine-tuning scripts (e.g., `scripts/finetune_model.sh`) for the Devstral model and the `mcp_dataset.json`.
        -   Ensure environment compatibility with Hugging Face libraries (PyTorch, Transformers, PEFT, Accelerate, TRL).
        -   **Status (01/06/2025)**: Fixed `NameError` for `DataCollatorForCompletionOnlyLM` and `TrainingArguments` import issues, and corrected dataset variable name in `finetune_output/run_finetune.py`.
    2.  **Execution**:
        -   Run fine-tuning using `just finetune-mcp model_name="mistralai/Devstral-Small-2505" data_path="data/preprocessed_mcp_dataset.json" output_dir="finetune_output"` (or similar, after preprocessing).
        -   Ensure the `preprocessed_mcp_dataset.json` is generated if not already present.
    3.  **Monitoring**:
        -   Track training loss, validation metrics (if applicable), and resource usage (GPU memory, time).
        -   Log training progress systematically.
    4.  **Artifact Storage**:
        -   Save the fine-tuned model weights, adapter configurations, and tokenizer files to a designated output directory (e.g., `finetune_output/devstral_mcp_tuned`).

## 11. Performance Evaluation

- **Objective**: Systematically compare the performance of the MCP fine-tuned Devstral model against the base Devstral model on a standardized set of MCP-related tasks.
- **Methodology**:
    1.  **Evaluation Dataset**:
        -   Utilize or expand the existing `data/eval_samples.json` to create a comprehensive set of evaluation prompts.
        -   Ensure prompts cover diverse MCP server interactions and complexity levels.
    2.  **Model Execution**:
        -   Set up inference pipelines for both the base model (`mistralai/Devstral-Small-2505`) and the fine-tuned model.
        -   Run both models on each evaluation prompt, collecting their full generated completions.
        -   The `just evaluate` command can be adapted for this: `just evaluate base_model_path finetuned_model_path data/eval_samples.json`.
    3.  **Data Collection**:
        -   Store model outputs in a structured format (e.g., JSON) for easy comparison and analysis. Include prompt, base model output, and fine-tuned model output.
- **Metrics**:
    -   **Task Completion Accuracy**:
        -   Does the model correctly identify and attempt to use the appropriate MCP tool(s)?
        -   Are the parameters for tool calls correct and relevant to the instruction?
        -   Binary (Success/Failure) or a graded score per task.
    -   **Response Quality**:
        -   Clarity and coherence of the model's reasoning (if any).
        -   Correctness of information provided or actions taken.
        -   Adherence to MCP formatting for tool calls.
    -   **Efficiency (Optional)**:
        -   Number of tool calls made to achieve the task (fewer might be better if effective).
        -   Latency of response.
    -   **Qualitative Review**:
        -   Human assessment of outputs for nuances not captured by automated metrics.
        -   Identification of common failure modes or strengths.

## 12. Results Analysis & Verification

- **Objective**: Analyze the collected evaluation data to quantitatively and qualitatively verify if the MCP instruction-tuned Devstral model demonstrates improved performance on MCP tasks compared to the base model.
- **Steps**:
    1.  **Quantitative Comparison**:
        -   Aggregate and compare the defined metrics (Task Completion Accuracy, etc.) for both models.
        -   Use statistical measures if applicable to determine significance of differences.
    2.  **Qualitative Analysis**:
        -   Review model outputs side-by-side for selected evaluation prompts.
        -   Identify patterns in errors, successful execution of complex instructions, and overall helpfulness.
    3.  **Identify Improvements & Regressions**:
        -   Pinpoint specific MCP tasks or server types where the fine-tuned model excels or lags.
        -   Note any unintended changes in behavior or new error types.
    4.  **Documentation**:
        -   Summarize findings in a report, including comparative metrics, qualitative observations, and examples.
        -   Conclude whether the fine-tuning was successful in improving MCP-specific performance.
    5.  **Iteration (If Necessary)**:
        -   Based on the analysis, decide if further iterations are needed (e.g., dataset augmentation, hyperparameter tuning, changes to fine-tuning strategy).