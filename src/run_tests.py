#!/usr/bin/env python3
"""
Test runner script for running tests with uv and creating GitHub issues for failures.
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

class TestRunner:
    """Runner for tests with failure reporting via GitHub issues."""

    def __init__(self, github_token=None):
        """Initialize the test runner.

        Args:
            github_token: Optional GitHub token for authentication
        """
        self.github_token = github_token
        if not self.github_token:
            # Try to get from environment
            self.github_token = os.environ.get("GITHUB_TOKEN")

    def run_test(self, test_name, test_command):
        """Run a specific test and return result.

        Args:
            test_name: Name of the test
            test_command: Command to run the test

        Returns:
            tuple: (success, output, error)
        """
        print(f"Running test: {test_name}")
        print(f"Command: {test_command}")

        try:
            # Run the command
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                print(f"✅ Test {test_name} passed")
                return True, result.stdout, None
            else:
                print(f"❌ Test {test_name} failed (exit code: {result.returncode})")
                return False, result.stdout, result.stderr

        except Exception as e:
            print(f"❌ Error running test {test_name}: {str(e)}")
            return False, "", str(e)

    def create_github_issue(self, title, body):
        """Create a GitHub issue for a test failure.

        Args:
            title: Issue title
            body: Issue description

        Returns:
            bool: True if issue was created successfully
        """
        if not self.github_token:
            print("Warning: No GitHub token available. Cannot create issue.")
            return False

        # Create temporary file for issue body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        issue_body = f"{body}\n\nTimestamp: {timestamp}"

        issue_file = Path("./temp_issue.txt")
        try:
            with open(issue_file, "w") as f:
                f.write(issue_body)

            # Use gh CLI to create issue
            cmd = f'gh issue create --title "{title}" --body-file "{issue_file}"'
            if self.github_token:
                cmd = f'GH_TOKEN={self.github_token} {cmd}'

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                print(f"✅ Created GitHub issue: {title}")
                print(f"Issue URL: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Failed to create GitHub issue: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error creating GitHub issue: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            if issue_file.exists():
                issue_file.unlink()

    def run_subsystem_test(self, subsystem, test_file):
        """Run tests for a specific subsystem.

        Args:
            subsystem: Name of the subsystem
            test_file: Path to test file

        Returns:
            bool: True if test passed
        """
        # Build the command using uv
        command = f"uv run python -m pytest {test_file} -v"

        # Run the test
        success, stdout, stderr = self.run_test(subsystem, command)

        # Always print the output for debugging purposes
        print("\nTest output:")
        print(stdout)
        if stderr:
            print("\nError output:")
            print(stderr)

        # Handle test failure
        if not success:
            # Create descriptive issue title and body
            title = f"[TEST FAILURE] {subsystem} tests failed"
            body = f"### Test Failure: {subsystem}\n\n"
            body += "#### Command\n```\n" + command + "\n```\n\n"
            body += "#### Standard Output\n```\n" + stdout + "\n```\n\n"

            if stderr:
                body += "#### Error Output\n```\n" + stderr + "\n```\n\n"

            # Create GitHub issue
            self.create_github_issue(title, body)

        return success

def main():
    """Main function to run tests for specific subsystems."""
    parser = argparse.ArgumentParser(description="Run tests and create GitHub issues for failures")
    parser.add_argument("--subsystem", type=str, required=True,
                        help="Subsystem to test (template_helpers, examples_generator, dataset_generator)")
    parser.add_argument("--github-token", type=str,
                        help="GitHub token for authentication (optional)")

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner(github_token=args.github_token)

    # Map subsystems to test files
    subsystem_tests = {
        "template_helpers": "src/tests/test_template_helpers.py",
        "examples_generator": "src/tests/test_generate_examples.py",
        "dataset_generator": "src/tests/test_dataset_generator.py",
    }

    # Check if subsystem is valid
    if args.subsystem not in subsystem_tests:
        print(f"Error: Unknown subsystem '{args.subsystem}'")
        print(f"Available subsystems: {', '.join(subsystem_tests.keys())}")
        return 1

    # Run the test for the specified subsystem
    test_file = subsystem_tests[args.subsystem]
    success = runner.run_subsystem_test(args.subsystem, test_file)

    # Return exit code based on test result
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())