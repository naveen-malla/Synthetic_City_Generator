import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path
import requests

data_dirs = {
    'adj_matrices': {
        'world': {
            'whole': Path('data/adj_matrices/world/whole/'),
            'center': Path('data/adj_matrices/world/center/')
        }
    }
}

for category in data_dirs.values():
    for region in category.values():
        for mode in region.values():
            mode.mkdir(parents=True, exist_ok=True)

def process_city(city_name, country_code, mode):
    print(f"Processing {city_name}, {country_code}...")
    try:
        clean_name = f"{city_name.split('/')[0].strip().lower().replace(' ', '_')}_{country_code}"
        file_path = data_dirs['adj_matrices']['world'][mode] / f"{clean_name}_adj.npy"
        
        if file_path.exists():
            print(f"File already exists for {city_name}, {country_code} ({mode}). Skipping download...")
            return np.load(file_path)
        
        print(f"Downloading new data for {city_name}, {country_code}...")
        center_point = ox.geocode(f"{city_name}, {country_code}")
        if center_point is None:
            print(f"Could not geocode {city_name}, {country_code}")
            return None
        
        try:
            if mode == 'whole':
                G = ox.graph_from_place(f"{city_name}, {country_code}", 
                                        network_type='drive',
                                        custom_filter='["highway"~"primary|secondary|residential|motorway"]',
                                        timeout=300)
            else:  # center mode
                G = ox.graph_from_point(center_point, 
                                        dist=500,
                                        network_type='drive',
                                        custom_filter='["highway"~"primary|secondary|residential|motorway"]',
                                        timeout=300)
            
            # Skip if the graph is too large (e.g., more than 10000 nodes)
            if len(G.nodes) > 10000:
                print(f"Skipping {city_name}, {country_code} due to large size ({len(G.nodes)} nodes).")
                return None
            
            return process_graph(G, city_name, country_code, mode)
        except requests.exceptions.Timeout:
            print(f"Skipping {city_name}, {country_code} due to timeout.")
            return None
    except Exception as e:
        print(f"Error processing {city_name}, {country_code}: {e}")
        return None

def process_graph(G, city_name, country_code, mode):
    # Convert to undirected graph
    G = G.to_undirected()
    
    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    dense_matrix = adj_matrix.todense()
    
    clean_name = f"{city_name.lower().replace(' ', '_')}_{country_code}"
    np.save(data_dirs['adj_matrices']['world'][mode] / f"{clean_name}_adj.npy", dense_matrix)
    
    print(f"Saved adjacency matrix for {city_name}, {country_code} ({mode})")
    print(f"Matrix shape: {dense_matrix.shape}")
    return dense_matrix

def main():
    cities_folder = Path('cities')
    
    while True:
        choice = input("Choose processing mode:\n1. Whole city\n2. City center\nEnter choice (1/2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    mode = 'whole' if choice == '1' else 'center'
    
    for country_file in cities_folder.glob('*.txt'):
        country_code = country_file.stem.split('_')[-1]
        with open(country_file, 'r', encoding='utf-8') as f:
            cities = [line.strip() for line in f if line.strip()]
        
        for city in cities:
            process_city(city, country_code, mode)

if __name__ == "__main__":
    main()
