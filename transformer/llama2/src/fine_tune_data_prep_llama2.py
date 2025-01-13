import os
import numpy as np
import json
from tqdm import tqdm

# Define variables for file paths and node range
COORD_FOLDER = 'data/coordinates/germany/center/transformed/train'
ADJ_FOLDER = 'data/adj_matrices/germany/center/train'
OUTPUT_FOLDER = 'transformer/llama2/dataset'
OUTPUT_FILE = 'train_10_50_nodes.json'
MIN_NODES = 10
MAX_NODES = 50

def get_matching_files(coord_folder, adj_folder):
    coord_files = set(f.replace('_coords.npy', '') for f in os.listdir(coord_folder) if f.endswith('_coords.npy'))
    adj_files = set(f.replace('_adj.npy', '') for f in os.listdir(adj_folder) if f.endswith('_adj.npy'))
    return list(coord_files.intersection(adj_files))

def load_and_process_coordinates(file_name):
    coord_file = os.path.join(COORD_FOLDER, f"{file_name}_coords.npy")
    structured_coords = np.load(coord_file)
    return np.column_stack((structured_coords['y'], structured_coords['x'])).astype(np.float32)

def check_node_count(file_name):
    adj_file = os.path.join(ADJ_FOLDER, f"{file_name}_adj.npy")
    adj_matrix = np.load(adj_file)
    return adj_matrix.shape[0]

def format_data_for_training(coordinates):
    total_pairs = coordinates.shape[0]
    input_pairs = int(total_pairs * 0.2)
    input_coords = coordinates[:input_pairs].flatten().tolist()
    output_coords = coordinates[input_pairs:].flatten().tolist()
    return {"input": input_coords, "output": output_coords}

def main():
    matching_files = get_matching_files(COORD_FOLDER, ADJ_FOLDER)
    dataset = []

    for file_name in tqdm(matching_files, desc="Processing files"):
        node_count = check_node_count(file_name)
        if MIN_NODES <= node_count <= MAX_NODES:
            coordinates = load_and_process_coordinates(file_name)
            formatted_data = format_data_for_training(coordinates)
            dataset.append(formatted_data)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    with open(output_path, 'w') as f:
        json.dump(dataset, f)

    print(f"Dataset saved to {output_path}")
    print(f"Total number of samples: {len(dataset)}")

if __name__ == "__main__":
    main()
