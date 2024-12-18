import osmnx as ox
import numpy as np
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data/coordinates/')
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

def process_city(city_name):
    print(f"Processing {city_name}...")
    try:
        # Download street network
        G = ox.graph_from_place(city_name,
                                network_type='drive',
                                custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        # Extract node coordinates
        nodes, _ = ox.graph_to_gdfs(G)
        nodes = nodes[['y', 'x']]
        
        # Create structured array for coordinates
        coordinates = np.array([(int(osmid), y, x) for osmid, (y, x) in nodes.iterrows()],
                               dtype=[('osmid', np.int64), ('y', np.float64), ('x', np.float64)])
        
        # Create clean filename
        clean_name = city_name.split(',')[0].lower().replace(' ', '_')
        
        # Save coordinates
        file_path = data_dir / f"{clean_name}_coords.npy"
        np.save(file_path, coordinates)
        
        print(f"Saved coordinates for {city_name}")
        print(f"Number of nodes: {len(coordinates)}")
        
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
