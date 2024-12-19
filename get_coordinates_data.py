import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path

# Create data directories if they don't exist
data_dir_original = Path('data/coordinates/original/')
data_dir_transformed = Path('data/coordinates/transformed/')

for directory in [data_dir_original, data_dir_transformed]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

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

def process_city(city_name):
    print(f"Processing {city_name}...")
    try:
        G = ox.graph_from_place(city_name,
                                network_type='drive',
                                custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        G.remove_edges_from(nx.selfloop_edges(G))
        
        nodes, _ = ox.graph_to_gdfs(G)
        nodes = nodes[['y', 'x']]
        
        nodes = normalize_coordinates(nodes)
        nodes = quantize_coordinates(nodes)
        
        clean_name = city_name.split(',')[0].lower().replace(' ', '_')
        
        original_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y', 'x']].iterrows()],
                                  dtype=[('osmid', np.int64), ('y', np.float64), ('x', np.float64)])
        
        transformed_coords = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes[['y_quant', 'x_quant']].iterrows()],
                                     dtype=[('osmid', np.int64), ('y', np.int64), ('x', np.int64)])
        
        np.save(data_dir_original / f"{clean_name}_coords.npy", original_coords)
        np.save(data_dir_transformed / f"{clean_name}_coords.npy", transformed_coords)
        
        print(f"Saved coordinates for {city_name}")
        print(f"Number of nodes: {len(original_coords)}")
        
    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def main():
    with open('cities_germany.txt', 'r') as f:
        cities = [line.strip() for line in f if line.strip()]
    
    for city in cities:
        process_city(city)

if __name__ == "__main__":
    main()
