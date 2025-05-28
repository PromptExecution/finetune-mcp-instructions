#!/usr/bin/env python3
"""
Example Generator Script

This script generates initial example templates for different MCP servers
and saves them to the appropriate directories.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

# Import the helper functions
from utils.template_helpers import (
    generate_filesystem_examples,
    generate_git_examples,
    generate_github_examples,
    save_example_templates
)

# Define additional server types
def generate_database_examples():
    """Generate examples for the database MCP server."""
    from utils.template_helpers import create_example_template

    examples = []

    # Simple example: Execute a simple query
    examples.append(create_example_template(
        instruction="Show me all users in the database",
        completion=(
            "I'll execute a query to show all users in the database.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT * FROM users"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="database",
        tool="execute_query",
        complexity="simple"
    ))

    # Simple example: Get database schema
    examples.append(create_example_template(
        instruction="Show me the schema of the products table",
        completion=(
            "I'll show the schema of the products table.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>get_schema</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "table": "products"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="database",
        tool="get_schema",
        complexity="simple"
    ))

    # Multi-step example: Execute query and count results
    examples.append(create_example_template(
        instruction="Find all orders with a total greater than $100 and count them",
        completion=(
            "I'll find all orders with a total greater than $100 and then count them.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT * FROM orders WHERE total > 100"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll count these orders.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT COUNT(*) FROM orders WHERE total > 100"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="database",
        tool="execute_query",
        complexity="multi_step"
    ))

    # Complex example: Join tables and analyze results
    examples.append(create_example_template(
        instruction="Find the top 5 customers by total order value, including their contact information",
        completion=(
            "I'll join the customers and orders tables to find the top customers by order value.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT c.customer_id, c.name, c.email, SUM(o.total) as total_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name, c.email ORDER BY total_value DESC LIMIT 5"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll get more details about the top customer.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT * FROM customers WHERE customer_id = 42"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Let me also find the specific orders from this customer.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>database</server_name>\n"
            "<tool_name>execute_query</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "query": "SELECT * FROM orders WHERE customer_id = 42 ORDER BY total DESC"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Based on this analysis, customer #42 (Jane Smith) has the highest total order value at $15,420.75 across 28 orders."
        ),
        server="database",
        tool="execute_query",
        complexity="complex"
    ))

    return examples

def generate_custom_examples():
    """Generate examples for custom MCP servers."""
    from utils.template_helpers import create_example_template

    examples = []

    # Weather API example (simple)
    examples.append(create_example_template(
        instruction="What's the current weather in San Francisco?",
        completion=(
            "I'll check the current weather in San Francisco using the Weather MCP server.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>weather</server_name>\n"
            "<tool_name>get_current</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "location": "San Francisco, CA"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="weather",
        tool="get_current",
        complexity="simple"
    ))

    # Calculator example (simple)
    examples.append(create_example_template(
        instruction="Calculate the square root of 144",
        completion=(
            "I'll calculate the square root of 144.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>calculator</server_name>\n"
            "<tool_name>calculate</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "expression": "sqrt(144)"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>"
        ),
        server="calculator",
        tool="calculate",
        complexity="simple"
    ))

    # Multi-step weather forecast example
    examples.append(create_example_template(
        instruction="Should I plan an outdoor event this weekend in Boston?",
        completion=(
            "I'll check the weekend forecast for Boston to help you decide.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>weather</server_name>\n"
            "<tool_name>get_forecast</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "location": "Boston, MA",\n'
            '  "days": 5\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Let me also check if there's any precipitation expected.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>weather</server_name>\n"
            "<tool_name>get_precipitation</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "location": "Boston, MA",\n'
            '  "days": 2,\n'
            '  "start_day": "Saturday"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Based on the forecast, Saturday looks promising with sunny conditions and only a 10% chance of rain, but Sunday has a 60% chance of precipitation. I would recommend planning your outdoor event for Saturday."
        ),
        server="weather",
        tool="get_forecast",
        complexity="multi_step"
    ))

    # Complex image processing example
    examples.append(create_example_template(
        instruction="Analyze this image, enhance its contrast, and identify any text in it",
        completion=(
            "I'll analyze the image, enhance it, and extract any text.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>image_processor</server_name>\n"
            "<tool_name>analyze</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "image_path": "uploads/image1.jpg"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Now I'll enhance the contrast.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>image_processor</server_name>\n"
            "<tool_name>enhance</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "image_path": "uploads/image1.jpg",\n'
            '  "enhancement": "contrast",\n'
            '  "level": 1.5\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "Let me extract any text from the image.\n\n"
            "<use_mcp_tool>\n"
            "<server_name>image_processor</server_name>\n"
            "<tool_name>ocr</tool_name>\n"
            "<arguments>\n"
            "{\n"
            '  "image_path": "uploads/enhanced_image1.jpg"\n'
            "}\n"
            "</arguments>\n"
            "</use_mcp_tool>\n\n"
            "The image has been enhanced, and I've extracted the text 'Welcome to San Francisco' from the sign in the image."
        ),
        server="image_processor",
        tool="analyze",
        complexity="complex"
    ))

    return examples

def main():
    """Main function to generate example templates."""
    print("Generating MCP example templates...")

    # Base directory for examples
    examples_dir = Path("./src/mcp_examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save examples for each server type
    server_generators = {
        "filesystem": generate_filesystem_examples,
        "git": generate_git_examples,
        "github": generate_github_examples,
        "database": generate_database_examples,
        "custom": generate_custom_examples
    }

    for server_type, generator_func in server_generators.items():
        print(f"Generating examples for {server_type}...")
        examples = generator_func()

        # Create the server directory if it doesn't exist
        server_dir = examples_dir / server_type
        server_dir.mkdir(parents=True, exist_ok=True)

        # Save the examples
        output_path = server_dir / "examples.json"
        save_example_templates(examples, output_path, overwrite=True)

    print(f"Generated examples for {len(server_generators)} MCP servers")
    print("Done!")

if __name__ == "__main__":
    main()