import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path

# Create data directories
data_dirs = {
    'whole': {
        'original': Path('data/coordinates/whole/original/'),
        'transformed': Path('data/coordinates/whole/transformed/')
    },
    'center': {
        'original': Path('data/coordinates/center/original/'),
        'transformed': Path('data/coordinates/center/transformed/')
    }
}

# Create all directories
for type_dir in data_dirs.values():
    for dir_path in type_dir.values():
        dir_path.mkdir(parents=True, exist_ok=True)

def normalize_coordinates(nodes):
    min_lat, max_lat = nodes['y'].min(), nodes['y'].max()
    min_lon, max_lon = nodes['x'].min(), nodes['x'].max()
    nodes['y_norm'] = (nodes['y'] - min_lat) / (max_lat - min_lat)
    nodes['x_norm'] = (nodes['x'] - min_lon) / (max_lon - min_lon)
    return nodes

def quantize_coordinates(nodes):
    nodes['y_quant'] = np.round(nodes['y_norm'] * 255).astype(int)
    nodes['x_quant'] = np.round(nodes['x_norm'] * 255).astype(int)
    return nodes

def process_city_whole(city_name, mode):
    """Process the entire city network"""
    print(f"Processing whole city: {city_name}...")
    try:
        G = ox.graph_from_place(city_name,
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        return process_graph(G, city_name, mode)
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def process_city_center(city_name, mode):
    """Process 1km x 1km area around city center"""
    print(f"Processing city center: {city_name}...")
    try:
        # Get city center coordinates
        center_point = ox.geocoder.geocode(city_name)
        print(f"City center coordinates (lat, lon): {center_point}")
        
        # Get 1km x 1km area (500m radius)
        G = ox.graph_from_point(center_point, 
                              dist=500,
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        return process_graph(G, city_name, mode)
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def process_graph(G, city_name, mode):
    """Common graph processing logic with mode-specific paths"""
    G.remove_edges_from(nx.selfloop_edges(G))
    
    nodes, _ = ox.graph_to_gdfs(G)
    nodes = nodes[['y', 'x']]
    
    nodes = normalize_coordinates(nodes)
    nodes = quantize_coordinates(nodes)
    
    clean_name = f"{city_name.split(',')[0].lower().replace(' ', '_')}"
    
    original_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y', 'x']].iterrows()],
                              dtype=[('osmid', np.int64), ('y', np.float64), ('x', np.float64)])
    
    transformed_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y_quant', 'x_quant']].iterrows()],
                                 dtype=[('osmid', np.int64), ('y', np.int64), ('x', np.int64)])
    
    # Save to mode-specific directories
    np.save(data_dirs[mode]['original'] / f"{clean_name}_coords.npy", original_coords)
    np.save(data_dirs[mode]['transformed'] / f"{clean_name}_coords.npy", transformed_coords)
    
    print(f"Saved coordinates for {city_name} ({mode})")
    print(f"Number of nodes: {len(original_coords)}")

def main():
    with open('cities_germany.txt', 'r') as f:
        cities = [line.strip() for line in f if line.strip()]
    
    while True:
        choice = input("Choose processing mode:\n1. Whole city\n2. City center\nEnter choice (1/2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    mode = 'whole' if choice == '1' else 'center'
    process_func = process_city_whole if choice == '1' else process_city_center
    
    for city in cities:
        process_func(city, mode)

if __name__ == "__main__":
    main()
