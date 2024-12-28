import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from test_vgae_without_coordinates_X import ImprovedVGAEEncoder, create_node_features, VGAE

def load_city_data(city_name):
    try:
        # Load coordinates
        coord_path = f'data/coordinates/center/transformed/{city_name}_coords.npy'
        coords = np.load(coord_path)
        print(f"Loaded coordinates shape: {coords.shape}")
        
        # Create sequential node mapping
        num_nodes = len(coords)
        positions = {}
        for i in range(num_nodes):
            positions[i] = (coords[i]['x'], coords[i]['y'])
        
        print(f"Created positions for {len(positions)} nodes")
        
        # Create features
        features = create_node_features(np.zeros((num_nodes, num_nodes)))
        return features, positions
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def inference(model, features, device):
    model.eval()
    with torch.no_grad():
        # Get latent representation
        z = model.encode(features.to(device), torch.zeros((2, 0)).long().to(device))
        # Generate adjacency matrix
        adj_pred = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
    return adj_pred > 0.5  # Threshold predictions

def plot_graph(adj_matrix, positions):
    # Create networkx graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Ensure all nodes have positions
    missing_nodes = set(G.nodes()) - set(positions.keys())
    if missing_nodes:
        raise ValueError(f"Missing positions for nodes: {missing_nodes}")
    
    # Plot
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos=positions, 
            node_size=20,
            node_color='black',
            edge_color='gray',
            width=0.5)
    plt.title("Inferred Street Network")
    plt.savefig('inferred_network.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    # Load city data
    city_name = "trier"  # or any other city
    features, positions = load_city_data(city_name)
    
    # Load best model
    checkpoint = torch.load('best_vgae_model.pt', map_location=device)
    model = VGAE(
        encoder=ImprovedVGAEEncoder(
            in_channels=features.shape[1],
            hidden_channels=512,
            out_channels=256,
            dropout=0.3
        )
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate adjacency matrix
    adj_pred = inference(model, features, device)
    
    # Plot result
    plot_graph(adj_pred, positions)

if __name__ == "__main__":
    main()