import osmnx as ox
import networkx as nx
import os
import numpy as np
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data/adj_matrices/')
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(exist_ok=True)

def process_city(city_name):
    print(f"Processing {city_name}...")
    try:
        # Download street network
        G = ox.graph_from_place(city_name, 
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        # Convert to undirected graph
        G = G.to_undirected()
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(G)
        dense_matrix = adj_matrix.todense()
        
        # Create clean filename
        clean_name = city_name.split(',')[0].lower().replace(' ', '_')
        
        # Save adjacency matrix
        file_path = data_dir / f"{clean_name}_adj.npy"
        np.save(file_path, dense_matrix)
        
        print(f"Saved adjacency matrix for {city_name}")
        print(f"Matrix shape: {dense_matrix.shape}")
        
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def main():
    # Read cities from file
    with open('cities.txt', 'r') as f:
        cities = [line.strip() for line in f if line.strip()]
    
    # Process each city
    for city in cities:
        process_city(city)

if __name__ == "__main__":
    main()