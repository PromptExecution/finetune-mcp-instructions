import json
import argparse

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_dataset(input_file, output_file):
    logging.info(f"Processing input file: {input_file}")
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Example preprocessing: Ensure all examples have the required fields
    processed_data = []
    for example in data.get('examples', []):
        if 'instruction' in example and 'completion' in example:
            processed_data.append(example)

    # Save the processed dataset
    with open(output_file, 'w') as f:
        json.dump({'examples': processed_data}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess MCP dataset.')
    parser.add_argument('--input-file', required=True, help='Path to the input dataset file')
    parser.add_argument('--output-file', required=True, help='Path to save the processed dataset file')
    args = parser.parse_args()

    preprocess_dataset(args.input_file, args.output_file)