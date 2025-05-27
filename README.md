# finetune-mcp-instructions
a dataset &amp; tools for finetuning any instruction model on the nuances of mcp

Instruction-Completion Dataset – Assemble a supervised dataset of (instruction, completion) pairs reflecting the MCP-style tasks. This may include examples derived from your documentation, SDK usage examples, and code-related queries. For instance, given an MCP server (e.g. “Git” or “Filesystem”), one instruction might be “List all functions defined in this utils.py file” and the completion the expected list of functions. The Model Context Protocol (“MCP”) GitHub includes reference servers like Git, GitHub, Filesystem, etc., which show how to query code repositories
github.com. 

We can use those (and our own docs) to craft prompts guiding the model to follow MCP conventions (e.g. calling tools, reading files, etc.).
Formatting (MCP style) – Ensure the data mimics MCP prompt style. While standard instruction-tuning uses something like “Human: … Assistant: …”, MCP often involves structured JSON or step-by-step tool calls. We can flatten these into natural-text instructions (e.g. “Using the Git repository, find the commit history of file X”) with completions that would correspond to the MCP client’s expected JSON response. In training, each example should clearly delineate the “instruction” (what the LLM must do, possibly referencing a tool or code) and the “output” (the answer or next step).

Loss Masking (Completion-Only) – When training, we typically do not train on the user’s prompt text. Instead, only the “assistant” part should incur loss. Hugging Face’s TRL library provides DataCollatorForCompletionOnlyLM, which masks the prompt tokens so that only the model’s response is learned
discuss.huggingface.co. 

This matches how many instruction-tuning scripts work: set the labels of prompt tokens to -100 (ignore) and compute cross-entropy only on the answer tokens

discuss.huggingface.co
discuss.huggingface.co
. Doing so avoids the model wasting capacity on copying the prompt and focuses it on generating the correct completion.
Domain-Specific Tokens/Embeddings – To make the model code-aware, you could add special tokens (e.g. <func>, <class>, or tokens for common library names) to the tokenizer and fine-tune their embeddings. This effectively gives the model new “words” for code concepts. In practice, however, LoRA/QLoRA fine-tuning on code tasks often suffices without changing the vocab. If desired, you could train an auxiliary embedding matrix for domain terms, but in most pipelines the domain knowledge comes through the fine-tuning examples themselves. (Layering custom embedding modules or adapters for code is possible but experimental.)
Dataset Tools – Use standard libs (Hugging Face Datasets, Pandas) to load and preprocess data. Ensure all prompts and completions use the model’s tokenizer (e.g. Mistral’s Tekken tokenizer) and respect its max context (Devstral uses a 128k window
huggingface.co
, though practical fine-tuning may use much shorter spans). Keep the instruction format consistent. Include error checking/validation (token lengths, etc.) in preprocessing.
