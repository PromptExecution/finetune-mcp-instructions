#!/usr/bin/env python3
"""
Template Helper Utilities

Helper functions for working with MCP example templates.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# XML-style tag patterns
MCP_TAG_PATTERN = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
OPENING_TAG_PATTERN = re.compile(r'<(\w+)>')
CLOSING_TAG_PATTERN = re.compile(r'</(\w+)>')

def validate_mcp_format(text: str) -> Tuple[bool, str]:
    """
    Validate that text contains properly formatted MCP XML-style tags.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # First check if there are any tags at all
    if not OPENING_TAG_PATTERN.search(text) and not CLOSING_TAG_PATTERN.search(text):
        return False, "No valid XML-style tags found"

    # Check for balanced opening and closing tags
    tag_stack = []
    current_pos = 0

    while current_pos < len(text):
        open_tag_match = re.search(r'<(\w+)>', text[current_pos:])
        close_tag_match = re.search(r'</(\w+)>', text[current_pos:])

        if not open_tag_match and not close_tag_match:
            break

        if not open_tag_match and close_tag_match:
            # Only closing tags left
            tag_name = close_tag_match.group(1)
            if not tag_stack or tag_stack[-1] != tag_name:
                return False, f"Unmatched closing tag: {tag_name}"
            tag_stack.pop()
            current_pos += close_tag_match.end()
        elif not close_tag_match and open_tag_match:
            # Only opening tags left
            tag_name = open_tag_match.group(1)
            tag_stack.append(tag_name)
            current_pos += open_tag_match.end()
        elif open_tag_match and close_tag_match:
            # Both tags exist, use the nearest
            if open_tag_match.start() < close_tag_match.start():
                tag_name = open_tag_match.group(1)
                tag_stack.append(tag_name)
                current_pos += open_tag_match.end()
            else:
                tag_name = close_tag_match.group(1)
                if not tag_stack or tag_stack[-1] != tag_name:
                    return False, f"Unmatched closing tag: {tag_name}"
                tag_stack.pop()
                current_pos += close_tag_match.end()

    # Check if all tags were closed
    if tag_stack:
        return False, f"Unclosed tags: {', '.join(tag_stack)}"

    return True, ""

def create_example_template(
    instruction: str,
    completion: str,
    server: str,
    tool: Optional[str] = None,
    complexity: str = "simple",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an example template with the specified fields.

    Args:
        instruction: Instruction text
        completion: Completion text
        server: MCP server type
        tool: Tool name (optional)
        complexity: Complexity level (simple, multi_step, complex)
        additional_metadata: Additional metadata fields

    Returns:
        Example template dictionary
    """
    # Validate inputs
    if not instruction:
        raise ValueError("Instruction cannot be empty")
    if not completion:
        raise ValueError("Completion cannot be empty")
    if complexity not in ["simple", "multi_step", "complex"]:
        raise ValueError(f"Invalid complexity: {complexity}")

    # Create metadata
    metadata = {
        "server": server,
        "complexity": complexity
    }

    if tool:
        metadata["tool"] = tool

    # Add additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    # Create example
    example = {
        "instruction": instruction,
        "completion": completion,
        "metadata": metadata
    }

    # Validate MCP format in completion
    is_valid, error = validate_mcp_format(completion)
    if not is_valid:
        raise ValueError(f"Invalid MCP format in completion: {error}")

    return example

def save_example_templates(
    examples: List[Dict[str, Any]],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """
    Save example templates to a JSON file.

    Args:
        examples: List of example templates
        output_path: Path to save to
        overwrite: Whether to overwrite existing file
    """
    # Check if file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_path} already exists and overwrite=False")

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save examples
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Saved {len(examples)} examples to {output_path}")

def load_example_templates(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load example templates from a JSON file.

    Args:
        input_path: Path to load from

    Returns:
        List of example templates
    """
    # Check if file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    # Load examples
    with open(input_path, "r") as f:
        examples = json.load(f)

    # Validate examples
    if not isinstance(examples, list):
        raise ValueError(f"Expected a list of examples, got {type(examples)}")

    print(f"Loaded {len(examples)} examples from {input_path}")
    return examples

def generate_filesystem_examples() -> List[Dict[str, Any]]:
    """
    Generate examples for the Filesystem MCP server.

    Returns:
        List of example templates
    """
    examples = []

    # Simple example: Read file
    examples.append(create_example_template(
        instruction="Read the content of the file README.md",
        completion=(
            "I'll read the content of the README.md file using the Filesystem MCP server.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>read_file</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "README.md"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="filesystem",
        tool="read_file",
        complexity="simple"
    ))

    # Simple example: List directory contents
    examples.append(create_example_template(
        instruction="List all files in the current directory",
        completion=(
            "I'll list all files in the current directory using the Filesystem MCP server.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>list_files</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "."\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="filesystem",
        tool="list_files",
        complexity="simple"
    ))

    # Multi-step example: Search and read file
    examples.append(create_example_template(
        instruction="Find all Python files in the src directory and show the content of the first one",
        completion=(
            "I'll first search for Python files in the src directory, then read the content of the first one.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>search_files</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "src",\n'
            '  "pattern": "*.py"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll read the content of the first Python file found.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>read_file</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "src/example.py"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="filesystem",
        tool="search_files",
        complexity="multi_step"
    ))

    # Complex example: Process multiple files
    examples.append(create_example_template(
        instruction="Find all JSON files in the data directory, read their contents, and count how many have a 'status' field set to 'success'",
        completion=(
            "I'll search for JSON files, read each one, and count the files with 'status': 'success'.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>search_files</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "data",\n'
            '  "pattern": "*.json"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll read each file and count those with status: 'success'.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>read_file</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "data/file1.json"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "<use_mcp_tool>\n"
            "<server_name>filesystem</server_name>\n"
            "<tool_name>read_file</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "data/file2.json"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "After analyzing all files, I found 2 files with 'status': 'success'."
        ),
        server="filesystem",
        tool="search_files",
        complexity="complex"
    ))

    return examples

def generate_git_examples() -> List[Dict[str, Any]]:
    """
    Generate examples for the Git MCP server.

    Returns:
        List of example templates
    """
    examples = []

    # Simple example: Git status
    examples.append(create_example_template(
        instruction="Check the status of the Git repository",
        completion=(
            "I'll check the status of the Git repository.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>status</tool_name>\n"
            "<arguments>\n"
            "{}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="git",
        tool="status",
        complexity="simple"
    ))

    # Simple example: Git log
    examples.append(create_example_template(
        instruction="Show the commit history for the main.py file",
        completion=(
            "I'll show the commit history for the main.py file.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>log</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "file": "main.py"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="git",
        tool="log",
        complexity="simple"
    ))

    # Multi-step example: Git diff and blame
    examples.append(create_example_template(
        instruction="Show changes to the README.md file in the last commit, and who last modified the file",
        completion=(
            "I'll check the changes to README.md in the last commit, then find who last modified it.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>diff</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "file": "README.md",\n'
            '  "commit": "HEAD~1",\n'
            '  "compare_to": "HEAD"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll check who last modified the file.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>blame</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "file": "README.md"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="git",
        tool="diff",
        complexity="multi_step"
    ))

    # Complex example: Analyze branches and commits
    examples.append(create_example_template(
        instruction="List all branches, show commits unique to the 'feature-x' branch compared to 'main', and summarize the changes",
        completion=(
            "I'll analyze branches and compare feature-x to main.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>branch</tool_name>\n"
            "<arguments>\n"
            "{}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll find commits unique to feature-x compared to main.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>log</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "range": "main..feature-x"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Let me examine the changes in these commits.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>diff</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "commit": "main",\n'
            '  "compare_to": "feature-x"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Based on the analysis, feature-x adds 3 new features and modifies 2 existing files."
        ),
        server="git",
        tool="branch",
        complexity="complex"
    ))

    return examples

def generate_github_examples() -> List[Dict[str, Any]]:
    """
    Generate examples for the GitHub MCP server.

    Returns:
        List of example templates
    """
    examples = []

    # Simple example: List repository issues
    examples.append(create_example_template(
        instruction="List open issues in the repository",
        completion=(
            "I'll list the open issues in this repository.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>list_issues</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "state": "open"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="github",
        tool="list_issues",
        complexity="simple"
    ))

    # Simple example: Get repository info
    examples.append(create_example_template(
        instruction="Show information about this GitHub repository",
        completion=(
            "I'll show information about this GitHub repository.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>get_repo</tool_name>\n"
            "<arguments>\n"
            "{}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="github",
        tool="get_repo",
        complexity="simple"
    ))

    # Multi-step example: Search code and view file
    examples.append(create_example_template(
        instruction="Search for code that uses the 'useEffect' hook in the repository and show one of the files",
        completion=(
            "I'll search for code using the useEffect hook, then view one of the files.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>search_code</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "useEffect"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll view one of the files that contains the useEffect hook.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>get_content</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "path": "src/components/App.js"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="github",
        tool="search_code",
        complexity="multi_step"
    ))

    # Complex example: Analyze pull requests
    examples.append(create_example_template(
        instruction="Find all pull requests related to performance improvements, show their changes, and summarize the approaches used",
        completion=(
            "I'll analyze pull requests related to performance improvements.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>search_issues</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "is:pr performance"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll examine the changes in PR #42.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>get_pull_request</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "number": 42\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Let me look at the files changed in this PR.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>github</server_name>\n"
            "<tool_name>get_pull_request_files</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "number": 42\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Based on my analysis of the PRs, common performance improvement approaches include:\n"
            "1. Memoization of expensive components\n"
            "2. Optimized rendering algorithms\n"
            "3. Bundle size reduction\n"
            "4. Database query optimization"
        ),
        server="github",
        tool="search_issues",
        complexity="complex"
    ))

    return examples

if __name__ == "__main__":
    # This script can be run directly to generate example templates
    print("Generating example templates...")

    # Create output directories
    examples_dir = Path("./src/mcp_examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples for each server type
    for server_type, generator_func in {
        "filesystem": generate_filesystem_examples,
        "git": generate_git_examples,
        "github": generate_github_examples
    }.items():
        print(f"Generating examples for {server_type}...")
        examples = generator_func()

        # Save examples
        output_path = examples_dir / f"{server_type}/examples.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_example_templates(examples, output_path, overwrite=True)

    print("Done generating example templates.")