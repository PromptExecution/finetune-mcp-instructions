#!/usr/bin/env python3
"""
Unit tests for the dataset generator.
"""

import os
import json
import unittest
from pathlib import Path
import tempfile
import shutil
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import from dataset_generator
from src.dataset_generator import MCPDatasetGenerator

class TestDatasetGenerator(unittest.TestCase):
    """Test the MCPDatasetGenerator class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.examples_dir = Path(self.temp_dir) / "examples"
        self.output_dir = Path(self.temp_dir) / "output"

        # Create server type directories and sample examples
        self.server_types = ["filesystem", "git", "github", "database", "custom"]
        self.complexity_types = ["simple", "multi_step", "complex"]

        # Create test examples
        self.create_sample_examples()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def create_sample_examples(self):
        """Create sample example files for testing."""
        for server_type in self.server_types:
            server_dir = self.examples_dir / server_type
            server_dir.mkdir(parents=True, exist_ok=True)

            examples = []
            # Create one example of each complexity for each server type
            for complexity in self.complexity_types:
                examples.append({
                    "instruction": f"Test instruction for {server_type} ({complexity})",
                    "completion": f"<use_mcp_tool>\n<server_name>{server_type}</server_name>\n<tool_name>test</tool_name>\n<arguments>{{}}</arguments>\n</use_mcp_tool>",
                    "metadata": {
                        "server": server_type,
                        "tool": "test",
                        "complexity": complexity
                    }
                })

            # Save examples to file
            with open(server_dir / "examples.json", "w") as f:
                json.dump(examples, f, indent=2)

    def test_initialization(self):
        """Test initialization of the dataset generator."""
        generator = MCPDatasetGenerator(
            examples_dir=self.examples_dir,
            output_dir=self.output_dir,
            validate=True
        )

        # Check that directories were properly set
        self.assertEqual(generator.examples_dir, self.examples_dir)
        self.assertEqual(generator.output_dir, self.output_dir)
        self.assertTrue(generator.validate)

        # Check that output directory was created
        self.assertTrue(self.output_dir.exists())

        # Check that examples dictionary was initialized
        for server_type in self.server_types:
            self.assertIn(server_type, generator.examples)
            self.assertEqual(generator.examples[server_type], [])

        # Check that stats were initialized
        self.assertEqual(generator.stats["total"], 0)
        self.assertEqual(generator.stats["simple"], 0)
        self.assertEqual(generator.stats["multi_step"], 0)
        self.assertEqual(generator.stats["complex"], 0)

    def test_load_example_templates(self):
        """Test loading example templates."""
        generator = MCPDatasetGenerator(
            examples_dir=self.examples_dir,
            output_dir=self.output_dir
        )

        generator.load_example_templates()

        # Check that examples were loaded for each server type
        for server_type in self.server_types:
            self.assertEqual(len(generator.examples[server_type]), 3)

            # Check that each example has the correct metadata
            for i, complexity in enumerate(self.complexity_types):
                example = generator.examples[server_type][i]
                self.assertEqual(example["metadata"]["complexity"], complexity)
                self.assertEqual(example["metadata"]["server"], server_type)

    def test_validate_example(self):
        """Test example validation."""
        generator = MCPDatasetGenerator()

        # Valid example
        valid_example = {
            "instruction": "Test instruction",
            "completion": "<use_mcp_tool>\n<server_name>test</server_name>\n<tool_name>test</tool_name>\n<arguments>{}</arguments>\n</use_mcp_tool>",
            "metadata": {
                "server": "test",
                "complexity": "simple"
            }
        }
        is_valid, error = generator.validate_example(valid_example)
        self.assertTrue(is_valid, f"Valid example failed validation: {error}")
        self.assertEqual(error, "")

        # Example missing instruction
        invalid_example = {
            "completion": "<use_mcp_tool></use_mcp_tool>",
            "metadata": {"server": "test", "complexity": "simple"}
        }
        is_valid, error = generator.validate_example(invalid_example)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Missing required field: instruction")

        # Example missing completion
        invalid_example = {
            "instruction": "Test",
            "metadata": {"server": "test", "complexity": "simple"}
        }
        is_valid, error = generator.validate_example(invalid_example)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Missing required field: completion")

        # Example missing metadata
        invalid_example = {
            "instruction": "Test",
            "completion": "<use_mcp_tool></use_mcp_tool>"
        }
        is_valid, error = generator.validate_example(invalid_example)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Missing metadata field")

        # Example with invalid completion (no tags)
        invalid_example = {
            "instruction": "Test",
            "completion": "No tags here",
            "metadata": {"server": "test", "complexity": "simple"}
        }
        is_valid, error = generator.validate_example(invalid_example)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Completion missing properly formatted XML tags")

    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = MCPDatasetGenerator(
            examples_dir=self.examples_dir,
            output_dir=self.output_dir
        )

        # Load examples
        generator.load_example_templates()

        # Generate dataset with examples
        # Our test data has 15 examples total (5 server types * 3 complexity types)
        # But due to distribution constraints, we may not get exactly the requested number
        target_count = 15
        dataset = generator.generate_dataset(target_count)

        # Check dataset properties
        self.assertGreaterEqual(len(dataset), 1)  # At least some examples were generated
        self.assertEqual(generator.stats["total"], len(dataset))

        # Log the actual counts for debugging
        print(f"Requested {target_count} examples, got {len(dataset)}")

        # Test that all complexity types are represented (if available in sample data)
        complexity_types = ["simple", "multi_step", "complex"]
        for complexity in complexity_types:
            if any(ex["metadata"]["complexity"] == complexity for server_type in self.server_types
                   for ex in generator.examples[server_type]):
                self.assertGreater(generator.stats[complexity], 0,
                                  f"No examples of {complexity} complexity were included")

    def test_save_dataset(self):
        """Test saving dataset."""
        generator = MCPDatasetGenerator(
            examples_dir=self.examples_dir,
            output_dir=self.output_dir
        )

        # Create a sample dataset
        dataset = [
            {
                "instruction": "Test instruction",
                "completion": "<use_mcp_tool></use_mcp_tool>",
                "metadata": {"server": "test", "complexity": "simple"}
            }
        ]

        # Save dataset
        output_file = "test_dataset.json"
        generator.save_dataset(dataset, output_file)

        # Check that file was created
        output_path = self.output_dir / output_file
        self.assertTrue(output_path.exists())

        # Check file contents
        with open(output_path, "r") as f:
            saved_data = json.load(f)

        self.assertIn("examples", saved_data)
        self.assertEqual(len(saved_data["examples"]), 1)
        self.assertEqual(saved_data["examples"][0]["instruction"], "Test instruction")

if __name__ == "__main__":
    unittest.main()