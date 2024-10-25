import pandas as pd
import json

def generate_jsonl(file_path, prompt_len, completion_len, overlap, output_file="coordinate_data.jsonl"):
    """
    Reads coordinates from a CSV file and creates a JSONL file with prompt and completion pairs in the GPT-4o format.
    
    Args:
    - file_path (str): Path to the CSV file with 'y_quant' and 'x_quant' columns.
    - prompt_len (int): Number of coordinate pairs in each prompt.
    - completion_len (int): Number of coordinate pairs in each completion.
    - overlap (int): Number of overlapping pairs between consecutive prompts.
    - output_file (str): Path to the output JSONL file.
    
    Output:
    - A JSONL file with 'messages' formatted prompt-completion pairs for GPT-4o fine-tuning.
    """
    # Load coordinates from CSV file
    coordinates_df = pd.read_csv(file_path)
    coordinates = list(zip(coordinates_df['y_quant'], coordinates_df['x_quant']))
    
    with open(output_file, "w") as f:
        index = 0
        # Generate prompt-completion pairs with overlap
        while index + prompt_len + completion_len <= len(coordinates):
            # Extract prompt and completion segments
            prompt_pairs = ' '.join(f"{y} {x}" for y, x in coordinates[index:index + prompt_len])
            completion_pairs = ' '.join(f"{y} {x}" for y, x in coordinates[index:index + prompt_len + completion_len])
            
            # Write to JSONL file in 'messages' format
            json_line = {
                "messages": [
                    {"role": "system", "content": "This assistant helps you recall coordinate sequences exactly."},
                    {"role": "user", "content": prompt_pairs},
                    {"role": "assistant", "content": completion_pairs}
                ]
            }
            f.write(json.dumps(json_line) + "\n")
            
            # Move to the next segment with overlap
            index += prompt_len - overlap

    print(f"JSONL file '{output_file}' created successfully with prompt-completion pairs in GPT-4o format.")

# Usage example
file_path = 'quantized_coordinates.csv'  # Path to the CSV file
prompt_len = 10        # Length of the prompt (number of coordinate pairs)
completion_len = 40    # Length of the completion (number of coordinate pairs)
overlap = 5            # Overlap between consecutive pairs

# Generate JSONL file in the new format
generate_jsonl(file_path, prompt_len, completion_len, overlap)