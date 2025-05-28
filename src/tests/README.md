# MCP Dataset Generator - Testing Framework

This directory contains tests for the MCP dataset generator components.

## Tests Overview

The testing framework is organized around the following subsystems:

1. **Template Helpers** (`test_template_helpers.py`): Tests for helper functions that create and validate MCP example templates.

2. **Examples Generator** (`test_generate_examples.py`): Tests for generating examples for different MCP server types.

3. **Dataset Generator** (`test_dataset_generator.py`): Tests for the main dataset generation functionality.

## Running Tests

Tests can be run using the `just` command-line utility, which uses `uv` under the hood for Python environment management.

To run all tests:

```bash
just test-all
```

To run tests for a specific subsystem:

```bash
just test-template-helpers
just test-examples-generator
just test-dataset-generator
```

For more detailed output and automatic GitHub issue creation on failure:

```bash
just test detail template_helpers
just test detail examples_generator
just test detail dataset_generator
```

## GitHub Issue Creation

The testing framework automatically creates GitHub issues for test failures. This is handled by `src/run_tests.py`, which:

1. Runs the specified tests using `uv`
2. Captures output and error information
3. Creates a GitHub issue if the test fails, including detailed information about the failure

For this to work, you should have the GitHub CLI (`gh`) installed and authenticated.

## Writing Additional Tests

When adding new functionality, be sure to add corresponding tests. Each test file should:

1. Import the necessary modules
2. Create test classes inheriting from `unittest.TestCase`
3. Define test methods that start with `test_`
4. Include appropriate assertions to verify functionality
5. Handle cleanup in `tearDown` methods if needed

## CI/CD Integration

These tests can be integrated into a CI/CD pipeline by running the `just test-all` command, which returns a non-zero exit code if any test fails.