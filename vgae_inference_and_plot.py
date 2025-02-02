import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import VGAE, GCNConv
import os
from vgae.VGAE import VariationalGCNEncoder


def create_node_features(coordinates, normalize=True):
    """
    Create rich node features from coordinates including positional and structural information.
    
    Args:
        coordinates: numpy array of shape (N, 2) containing x,y coordinates
        normalize: whether to normalize features
    
    Returns:
        node_features: torch tensor of shape (N, num_features)
    """
    import torch
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    N = coordinates.shape[0]
    features_list = []
    
    # 1. Original coordinates
    features_list.append(coordinates)
    
    # 2. Relative positions to center of mass
    center = np.mean(coordinates, axis=0)
    relative_pos = coordinates - center
    features_list.append(relative_pos)
    
    # 3. Radial and angular coordinates
    distances = np.linalg.norm(relative_pos, axis=1).reshape(-1, 1)
    angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0]).reshape(-1, 1)
    features_list.append(distances)
    features_list.append(angles)
    
    # 4. Local density features
    from sklearn.neighbors import NearestNeighbors
    k = min(10, N-1)  # number of neighbors to consider
    nbrs = NearestNeighbors(n_neighbors=k).fit(coordinates)
    distances, _ = nbrs.kneighbors(coordinates)
    local_density = np.mean(distances, axis=1).reshape(-1, 1)
    features_list.append(local_density)
    
    # 5. Grid-based positional encoding
    grid_size = 8
    x_pos = coordinates[:, 0].reshape(-1, 1)
    y_pos = coordinates[:, 1].reshape(-1, 1)
    pe_x = np.zeros((N, grid_size))
    pe_y = np.zeros((N, grid_size))
    
    for i in range(grid_size):
        pe_x[:, i] = np.sin(2 ** i * np.pi * x_pos.flatten())
        pe_y[:, i] = np.cos(2 ** i * np.pi * y_pos.flatten())
    
    features_list.extend([pe_x, pe_y])
    
    # Combine all features
    node_features = np.hstack(features_list)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
    
    return torch.FloatTensor(node_features)


def load_city_data(city_name):
    try:
        coord_path = f'data/coordinates/world/center/transformed/test/{city_name}_coords.npy'
        adj_path = f'data/adj_matrices/world/center/test/{city_name}_adj.npy'
        
        coords = np.load(coord_path)
        adj_matrix = np.load(adj_path)
        adj_matrix = (adj_matrix > 0).astype(float)
        
        # Use coordinates as features (2D)
        coordinates = np.column_stack((coords['y'], coords['x'])).astype(np.float32)
        node_features = create_node_features(coordinates)

        
        adj_tensor = torch.FloatTensor(adj_matrix)
        edge_index, _ = dense_to_sparse(adj_tensor)
        
        positions = {i: (coords[i]['x'], coords[i]['y']) for i in range(len(coords))}
        
        return edge_index, adj_matrix, node_features, positions
    except Exception as e:
        print(f"Error loading city data: {str(e)}")
        raise

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def visualize_networks(adj_matrix, pred_adj, positions, city_name, is_remote=False):
    results_dir = 'results'
    ensure_dir_exists(results_dir)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(121)
    G_original = nx.from_numpy_array(adj_matrix)
    G_original.remove_edges_from(nx.selfloop_edges(G_original))
    G_original = nx.Graph(G_original)  # Remove multi-edges
    nx.draw(G_original, positions, 
            node_size=30,
            node_color='blue',
            width=0.5,
            with_labels=False)
    plt.title(f"Original {city_name} Network")
    
    plt.subplot(122)
    pred_adj_binary = (pred_adj > 0.9).astype(int)
    G_pred = nx.from_numpy_array(pred_adj_binary)
    G_pred.remove_edges_from(nx.selfloop_edges(G_pred))
    G_pred = nx.Graph(G_pred)  # Remove multi-edges
    nx.draw(G_pred, positions, 
            node_size=30,
            node_color='red',
            width=0.5,
            with_labels=False)
    plt.title(f"Generated {city_name} Network")
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'{city_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if not is_remote:
        plt.show()
    plt.close()

def main():
    city_name = "möckern_DE"
    
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    print(f"Using device: {device}")
    
    is_remote = torch.cuda.is_available() and not torch.backends.mps.is_available()
    print(f"Running in {'remote' if is_remote else 'local'} mode")
    
    print(f"\nProcessing city: {city_name}")
    edge_index, adj_matrix, features, positions = load_city_data(city_name)
    
    model = VGAE(
        encoder=VariationalGCNEncoder(
            in_channels=23,
            hidden_channels=512,
            out_channels=256,
            dropout=0.3
        )
    ).to(device)
    
    model.load_state_dict(torch.load('model_checkpoints/best_vgae_model_improved_I_as_node_features_10_50_world.pt', map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        features = features.to(device)
        edge_index = edge_index.to(device)
        z = model.encode(features, edge_index)
        pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
        ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
        print(f"{city_name} Test AUC: {auc:.4f}, AP: {ap:.4f}")
        
        visualize_networks(adj_matrix, pred_adj, positions, city_name, is_remote)

if __name__ == "__main__":
    main()
