import osmnx as ox
import os
from pathlib import Path
import matplotlib.pyplot as plt
def get_city_graph():
    try:
        city_name = "Trier, Germany"
        print(f"Processing {city_name}...")
        
        # Get city center coordinates
        center_point = ox.geocoder.geocode(city_name)
        print(f"City center coordinates (lat, lon): {center_point}")
        
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
        
        # Plot configuration
        fig, ax = ox.plot_graph(
            G,
            node_size=3,           # Small nodes
            node_color='red',      # Red nodes
            edge_color='black',    # Black edges
            edge_linewidth=0.5,    # Thin edges
            bgcolor='white',       # White background
            show=False,           
            close=False
        )
        
        
        return G, fig, ax
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    G, fig, ax = get_city_graph()
    plt.show()  # Display the plot