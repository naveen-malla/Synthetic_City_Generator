import pandas as pd
import json

def generate_jsonl(file_path, prompt_len, completion_len, include_prompt_in_completion=True, output_file="coordinate_data.jsonl"):
    """
    Reads coordinates from a CSV file and creates a JSONL file with non-overlapping prompt and completion pairs
    in the GPT-4o format. Optionally includes the prompt in the completion.
    
    Args:
    - file_path (str): Path to the CSV file with 'y_quant' and 'x_quant' columns.
    - prompt_len (int): Number of coordinate pairs in each prompt.
    - completion_len (int): Number of coordinate pairs in each completion.
    - include_prompt_in_completion (bool): Whether to include prompt pairs within the completion pairs.
    - output_file (str): Path to the output JSONL file.
    
    Output:
    - A JSONL file with 'messages' formatted prompt-completion pairs for GPT-4o fine-tuning.
    """
    # Load coordinates from CSV file
    coordinates_df = pd.read_csv(file_path)
    coordinates = list(zip(coordinates_df['y_quant'], coordinates_df['x_quant']))
    
    with open(output_file, "w") as f:
        index = 0
        # Generate prompt-completion pairs without overlap
        while index + prompt_len + completion_len <= len(coordinates):
            # Extract prompt segment
            prompt_pairs = ' '.join(f"{y} {x}" for y, x in coordinates[index:index + prompt_len])
            
            # Decide completion pairs based on the inclusion choice
            if include_prompt_in_completion:
                # Completion includes prompt pairs
                completion_pairs = ' '.join(f"{y} {x}" for y, x in coordinates[index:index + prompt_len + completion_len])
            else:
                # Completion does not include prompt pairs
                completion_pairs = ' '.join(f"{y} {x}" for y, x in coordinates[index + prompt_len:index + prompt_len + completion_len])
            
            # Write to JSONL file in 'messages' format
            json_line = {
                "messages": [
                    {"role": "system", "content": "This assistant helps you recall coordinate sequences exactly."},
                    {"role": "user", "content": prompt_pairs},
                    {"role": "assistant", "content": completion_pairs}
                ]
            }
            f.write(json.dumps(json_line) + "\n")
            
            # Move to the next segment without overlap
            index += prompt_len + completion_len

    print(f"JSONL file '{output_file}' created successfully with prompt-completion pairs in GPT-4o format.")

# Usage example
file_path = 'quantized_coordinates.csv'  # Path to the CSV file
prompt_len = 20       # Length of the prompt (number of coordinate pairs)
completion_len = 5   # Length of the completion (number of coordinate pairs)
include_prompt_in_completion = True  # Set to False if you don't want prompt pairs in the completion

# Generate JSONL file for larger sequence prediction
generate_jsonl(file_path, prompt_len, completion_len, include_prompt_in_completion)