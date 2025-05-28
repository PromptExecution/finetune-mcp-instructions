#!/usr/bin/env python3
"""
Unit tests for template helper utilities.
"""

import os
import json
import unittest
from pathlib import Path
import tempfile
import shutil

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.template_helpers import (
    validate_mcp_format,
    create_example_template,
    save_example_templates,
    load_example_templates
)

class TestTemplateMCPFormat(unittest.TestCase):
    """Test the MCP format validation function."""

    def test_valid_simple_format(self):
        """Test validation of valid, simple MCP format."""
        text = "This is a simple <tag>example</tag> of valid MCP format."
        is_valid, error = validate_mcp_format(text)
        self.assertTrue(is_valid, f"Validation failed: {error}")
        self.assertEqual(error, "")

    def test_valid_nested_format(self):
        """Test validation of valid, nested MCP format."""
        text = (
            "<outer>This is <inner>nested</inner> content</outer>"
        )
        is_valid, error = validate_mcp_format(text)
        self.assertTrue(is_valid, f"Validation failed: {error}")
        self.assertEqual(error, "")

    def test_valid_complex_format(self):
        """Test validation of valid, complex MCP format."""
        text = (
            "Before tags\n"
            "<use_mcp_tool>\n"
            "<server_name>git</server_name>\n"
            "<tool_name>status</tool_name>\n"
            "<arguments>\n"
            "{}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n"
            "After tags"
        )
        is_valid, error = validate_mcp_format(text)
        self.assertTrue(is_valid, f"Validation failed: {error}")
        self.assertEqual(error, "")

    def test_no_tags(self):
        """Test validation with no tags."""
        text = "This text has no tags at all."
        is_valid, error = validate_mcp_format(text)
        self.assertFalse(is_valid)
        self.assertEqual(error, "No valid XML-style tags found")

    def test_unmatched_closing_tag(self):
        """Test validation with unmatched closing tag."""
        text = "This has an </unmatched> closing tag."
        is_valid, error = validate_mcp_format(text)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Unmatched closing tag: unmatched")

    def test_unmatched_opening_tag(self):
        """Test validation with unmatched opening tag."""
        text = "This has an <unmatched> opening tag."
        is_valid, error = validate_mcp_format(text)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Unclosed tags: unmatched")

class TestCreateExampleTemplate(unittest.TestCase):
    """Test the create_example_template function."""

    def test_create_simple_template(self):
        """Test creating a simple example template."""
        instruction = "Test instruction"
        completion = "<tag>Test completion</tag>"
        server = "test_server"
        template = create_example_template(
            instruction=instruction,
            completion=completion,
            server=server
        )

        self.assertEqual(template["instruction"], instruction)
        self.assertEqual(template["completion"], completion)
        self.assertEqual(template["metadata"]["server"], server)
        self.assertEqual(template["metadata"]["complexity"], "simple")  # Default

    def test_create_complex_template(self):
        """Test creating a complex example template with all options."""
        instruction = "Complex test instruction"
        completion = "<tag>Complex test completion</tag>"
        server = "test_server"
        tool = "test_tool"
        complexity = "complex"
        additional_metadata = {"key1": "value1", "key2": 123}

        template = create_example_template(
            instruction=instruction,
            completion=completion,
            server=server,
            tool=tool,
            complexity=complexity,
            additional_metadata=additional_metadata
        )

        self.assertEqual(template["instruction"], instruction)
        self.assertEqual(template["completion"], completion)
        self.assertEqual(template["metadata"]["server"], server)
        self.assertEqual(template["metadata"]["tool"], tool)
        self.assertEqual(template["metadata"]["complexity"], complexity)
        self.assertEqual(template["metadata"]["key1"], "value1")
        self.assertEqual(template["metadata"]["key2"], 123)

    def test_invalid_parameters(self):
        """Test the function with invalid parameters."""
        # Test empty instruction
        with self.assertRaises(ValueError):
            create_example_template("", "<tag>completion</tag>", "server")

        # Test empty completion
        with self.assertRaises(ValueError):
            create_example_template("instruction", "", "server")

        # Test invalid complexity
        with self.assertRaises(ValueError):
            create_example_template("instruction", "<tag>completion</tag>", "server",
                                  complexity="invalid")

        # Test invalid MCP format
        with self.assertRaises(ValueError):
            create_example_template("instruction", "no tags here", "server")

class TestSaveLoadExamples(unittest.TestCase):
    """Test the save_example_templates and load_example_templates functions."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory after testing."""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_examples(self):
        """Test saving and then loading example templates."""
        # Create test examples
        examples = [
            create_example_template("Instruction 1", "<tag>Completion 1</tag>", "server1"),
            create_example_template("Instruction 2", "<tag>Completion 2</tag>", "server2", tool="tool2")
        ]

        # Define output path
        output_path = Path(self.temp_dir) / "test_examples.json"

        # Save examples
        save_example_templates(examples, output_path)

        # Ensure file exists
        self.assertTrue(output_path.exists())

        # Load examples
        loaded_examples = load_example_templates(output_path)

        # Verify loaded examples
        self.assertEqual(len(loaded_examples), len(examples))
        self.assertEqual(loaded_examples[0]["instruction"], "Instruction 1")
        self.assertEqual(loaded_examples[1]["instruction"], "Instruction 2")
        self.assertEqual(loaded_examples[0]["metadata"]["server"], "server1")
        self.assertEqual(loaded_examples[1]["metadata"]["server"], "server2")
        self.assertEqual(loaded_examples[1]["metadata"]["tool"], "tool2")

    def test_overwrite_parameter(self):
        """Test the overwrite parameter of save_example_templates."""
        examples = [
            create_example_template("Test", "<tag>Test</tag>", "server")
        ]

        output_path = Path(self.temp_dir) / "overwrite_test.json"

        # Save initial file
        save_example_templates(examples, output_path)

        # Try to save again without overwrite
        with self.assertRaises(FileExistsError):
            save_example_templates(examples, output_path, overwrite=False)

        # Save with overwrite
        save_example_templates(examples, output_path, overwrite=True)  # Should not raise

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.json"
        with self.assertRaises(FileNotFoundError):
            load_example_templates(nonexistent_path)

if __name__ == '__main__':
    unittest.main()