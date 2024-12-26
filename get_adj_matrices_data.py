import osmnx as ox
import networkx as nx
import os
import numpy as np
from pathlib import Path

# Create data directories
data_dirs = {
    'whole': Path('data/adj_matrices/whole/'),
    'center': Path('data/adj_matrices/center/')
}

# Create all directories
for dir_path in data_dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)

def process_city_whole(city_name, mode):
    print(f"Processing {city_name}...")
    try:
        # Check if file already exists
        clean_name = f"{city_name.split(',')[0].lower().replace(' ', '_')}"
        file_path = data_dirs[mode] / f"{clean_name}_adj.npy"
        
        if file_path.exists():
            print(f"File already exists for {city_name} ({mode}). Skipping download...")
            return np.load(file_path)
            
        # Download street network if file doesn't exist
        print(f"Downloading new data for {city_name}...")
        G = ox.graph_from_place(city_name, 
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        # Convert to undirected graph
        G = G.to_undirected()
        return process_graph(G, city_name, mode)
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def process_city_center(city_name, mode):
    print(f"Processing city center: {city_name}...")
    try:
        # Check if file already exists
        clean_name = f"{city_name.split(',')[0].lower().replace(' ', '_')}"
        file_path = data_dirs[mode] / f"{clean_name}_adj.npy"
        
        if file_path.exists():
            print(f"File already exists for {city_name} ({mode}). Skipping download...")
            return np.load(file_path)
            
        # Get city center coordinates and download if file doesn't exist
        print(f"Downloading new data for {city_name} center...")
        center_point = ox.geocoder.geocode(city_name)
        print(f"City center coordinates (lat, lon): {center_point}")
        
        # Download street network within 1km x 1km area
        G = ox.graph_from_point(center_point, 
                              dist=500,
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        # Convert to undirected graph
        G = G.to_undirected()
        return process_graph(G, city_name, mode)
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def process_graph(G, city_name, mode):
    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    dense_matrix = adj_matrix.todense()
    
    # Save matrix
    clean_name = f"{city_name.split(',')[0].lower().replace(' ', '_')}"
    np.save(data_dirs[mode] / f"{clean_name}_adj.npy", dense_matrix)
    
    print(f"Saved adjacency matrix for {city_name} ({mode})")
    print(f"Matrix shape: {dense_matrix.shape}")
    return dense_matrix

def main():
    # Load cities
    with open('cities_germany.txt', 'r') as f:
        cities = [line.strip() for line in f if line.strip()]
    
    # Get user choice
    while True:
        choice = input("Choose processing mode:\n1. Whole city\n2. City center\nEnter choice (1/2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Process cities based on choice
    mode = 'whole' if choice == '1' else 'center'
    process_func = process_city_whole if choice == '1' else process_city_center
    
    for city in cities:
        process_func(city, mode)

if __name__ == "__main__":
    main()