import osmnx as ox
import os
from pathlib import Path

def get_city_graph():
    try:
        city_name = "Trier, Germany"
        print(f"Processing {city_name}...")
        
        # Get city center coordinates
        center_point = ox.geocoder.geocode(city_name)
        
        # Define the distance in meters (500 meters in each direction = 1km x 1km area)
        dist = 500
        
        # Download street network within bounding box with filtered roads
        G = ox.graph_from_point(center_point, 
                              dist=dist,
                              network_type='drive',
                              custom_filter='["highway"~"primary|secondary|residential|motorway"]')
        
        # Convert to undirected graph
        G = G.to_undirected()
        
        # Print basic statistics
        print(f"Number of nodes: {len(G.nodes)}")
        print(f"Number of edges: {len(G.edges)}")
        print("\nRoad types in the network:")
        road_types = set([G.edges[u, v, 0].get('highway', 'Unknown') for u, v, k in G.edges(keys=True)])
        for rt in road_types:
            print(f"- {rt}")
        
        return G
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    get_city_graph()