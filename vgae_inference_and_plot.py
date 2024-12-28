import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from test_vgae_without_coordinates_X import ImprovedVGAEEncoder, create_node_features, VGAE
import os

def load_city_data(city_name):
    try:
        # Load coordinates and adjacency matrix
        coord_path = f'data/coordinates/center/transformed/{city_name}_coords.npy'
        adj_path = f'data/adj_matrices/center/{city_name}_adj.npy'
        
        coords = np.load(coord_path)
        adj_matrix = np.load(adj_path)
        adj_matrix = (adj_matrix > 0).astype(float)  # Convert to binary
        
        # Create node features using adjacency matrix
        node_features = create_node_features(adj_matrix)
        
        # Convert to torch tensors
        adj_tensor = torch.FloatTensor(adj_matrix)
        edge_index, _ = dense_to_sparse(adj_tensor)
        
        # Create positions dictionary for visualization
        positions = {i: (coords[i]['x'], coords[i]['y']) for i in range(len(coords))}
            
        return edge_index, adj_matrix, node_features, positions
        
    except Exception as e:
        print(f"Error loading city data: {str(e)}")
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
    if (missing_nodes):
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

def visualize_networks(city_name,adj_matrix, pred_adj, positions, threshold=0.5):
    plt.figure(figsize=(15, 6))
    
    # Original network
    plt.subplot(121)
    G_original = nx.from_numpy_array(adj_matrix)
    nx.draw(G_original, positions, 
            node_size=30,
            node_color='blue',
            width=0.5,
            with_labels=False)
    plt.title(f"{city_name} Original")
    
    # Predicted network
    plt.subplot(122)
    pred_adj_binary = (pred_adj > threshold).astype(int)
    G_pred = nx.from_numpy_array(pred_adj_binary)
    nx.draw(G_pred, positions, 
            node_size=30,
            node_color='red',
            width=0.5,
            with_labels=False)
    plt.title(f'{city_name} Generated')
    
    plt.tight_layout()
    plt.savefig('{city_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def is_remote_execution():
    """Check if code is running on remote server"""
    return torch.cuda.is_available() and not torch.backends.mps.is_available()

def ensure_dir_exists(path):
    """Create directory if it doesn't exist"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def visualize_networks(adj_matrix, pred_adj, positions, city_name, is_remote=False):
    # Ensure results directory exists
    results_dir = 'results'
    ensure_dir_exists(results_dir)
    
    plt.figure(figsize=(15, 6))
    
    # Original network
    plt.subplot(121)
    G_original = nx.from_numpy_array(adj_matrix)
    nx.draw(G_original, positions, 
            node_size=30,
            node_color='blue',
            width=0.5,
            with_labels=False)
    plt.title(f"Original {city_name} Network")
    
    # Predicted network
    plt.subplot(122)
    pred_adj_binary = (pred_adj > 0.5).astype(int)
    G_pred = nx.from_numpy_array(pred_adj_binary)
    nx.draw(G_pred, positions, 
            node_size=30,
            node_color='red',
            width=0.5,
            with_labels=False)
    plt.title(f"Generated {city_name} Network")
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(results_dir, f'{city_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Show plot only on local machine
    if not is_remote:
        plt.show()
    plt.close()

def main():
    # Configuration
    city_name = "trier"  # Change this to test different cities
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    print(f"Using device: {device}")
    
    # Detect environment
    is_remote = is_remote_execution()
    print(f"Running in {'remote' if is_remote else 'local'} mode")
    
    # Load city data
    print(f"\nProcessing city: {city_name}")
    edge_index, adj_matrix, features, positions = load_city_data(city_name)
    
    # Initialize model
    in_channels = features.shape[1]
    model = VGAE(ImprovedVGAEEncoder(
        in_channels=in_channels,
        hidden_channels=512,
        out_channels=256,
        dropout=0.3
    )).to(device)
    
    # Load best model
    checkpoint = torch.load('best_vgae_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inference
    with torch.no_grad():
        features = features.to(device)
        edge_index = edge_index.to(device)
        z = model.encode(features, edge_index)
        pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
        
        # Evaluate
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
        ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
        print(f"{city_name} Test AUC: {auc:.4f}, AP: {ap:.4f}")
        
        # Visualize networks (removed redundant save)
        visualize_networks(adj_matrix, pred_adj, positions, city_name, is_remote)

if __name__ == "__main__":
    main()