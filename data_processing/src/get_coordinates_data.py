import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path
import signal

data_dirs = {
    'coordinates': {
        'world': {
            'whole': {
                'original': Path('data/coordinates/world/whole/original/'),
                'transformed': Path('data/coordinates/world/whole/transformed/')
            },
            'center': {
                'original': Path('data/coordinates/world/center/original/'),
                'transformed': Path('data/coordinates/world/center/transformed/')
            }
        }
    },
    'adj_matrices': {
        'world': {
            'whole': Path('data/adj_matrices/world/whole/'),
            'center': Path('data/adj_matrices/world/center/')
        }
    }
}

for category in data_dirs.values():
    for region in category.values():
        if isinstance(region, dict):
            for mode in region.values():
                if isinstance(mode, dict):
                    for dir_path in mode.values():
                        dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    mode.mkdir(parents=True, exist_ok=True)
        else:
            region.mkdir(parents=True, exist_ok=True)

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

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def process_city(city_name, country_code, mode):
    print(f"Processing {city_name}, {country_code}...")
    try:
        clean_name = f"{city_name.split('/')[0].strip().lower().replace(' ', '_')}_{country_code}"
        original_path = data_dirs['coordinates']['world'][mode]['original'] / f"{clean_name}_coords.npy"
        transformed_path = data_dirs['coordinates']['world'][mode]['transformed'] / f"{clean_name}_coords.npy"
        
        if original_path.exists() and transformed_path.exists():
            print(f"Files already exist for {city_name}, {country_code} ({mode}). Skipping download...")
            return np.load(original_path), np.load(transformed_path)
        
        print(f"Downloading new data for {city_name}, {country_code}...")
        center_point = ox.geocode(f"{city_name}, {country_code}")
        if center_point is None:
            print(f"Could not geocode {city_name}, {country_code}")
            return None
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # Set 120-second timeout
        
        try:
            if mode == 'whole':
                G = ox.graph_from_place(f"{city_name}, {country_code}", 
                                        network_type='drive',
                                        custom_filter='["highway"~"primary|secondary|residential|motorway"]')
            else:  # center mode
                G = ox.graph_from_point(center_point, 
                                        dist=500,
                                        network_type='drive',
                                        custom_filter='["highway"~"primary|secondary|residential|motorway"]')
            
            signal.alarm(0)  # Cancel the alarm
            return process_graph(G, city_name, country_code, mode)
        except TimeoutError:
            print(f"Timeout occurred while processing {city_name}, {country_code}")
            return None
    except Exception as e:
        print(f"Error processing {city_name}, {country_code}: {e}")
        return None

def process_graph(G, city_name, country_code, mode):
    G.remove_edges_from(nx.selfloop_edges(G))
    
    nodes, _ = ox.graph_to_gdfs(G)
    nodes = nodes[['y', 'x']]
    
    nodes = normalize_coordinates(nodes)
    nodes = quantize_coordinates(nodes)
    
    clean_name = f"{city_name.lower().replace(' ', '_')}_{country_code}"
    
    original_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y', 'x']].iterrows()],
                               dtype=[('osmid', np.int64), ('y', np.float64), ('x', np.float64)])
    
    transformed_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y_quant', 'x_quant']].iterrows()],
                                  dtype=[('osmid', np.int64), ('y', np.int64), ('x', np.int64)])
    
    np.save(data_dirs['coordinates']['world'][mode]['original'] / f"{clean_name}_coords.npy", original_coords)
    np.save(data_dirs['coordinates']['world'][mode]['transformed'] / f"{clean_name}_coords.npy", transformed_coords)
    
    print(f"Saved coordinates for {city_name}, {country_code} ({mode})")
    print(f"Number of nodes: {len(original_coords)}")
    return original_coords, transformed_coords

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
