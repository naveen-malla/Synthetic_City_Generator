import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import VGAE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from test_pytorch_geometric_VGAE import VariationalGCNEncoder

# Define paths
BASE_DIR = '/Users/naveenmalla/Documents/Projects/Thesis/Images/Final_Results'
MODEL_DIR = 'vgae_best_model_10_50/test'
save_path = os.path.join(BASE_DIR, MODEL_DIR)
os.makedirs(save_path, exist_ok=True)

# Color scheme
COLORS = {
    'original': '#2ECC71',  # Green
    'generated': '#E74C3C', # Red
    'betweenness': '#1ABC9C',  # Turquoise
    'closeness': '#3498DB'    # Light Blue
}

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'

def create_node_features(coordinates, normalize=True):
    N = coordinates.shape[0]
    features_list = []
    
    features_list.append(coordinates)
    center = np.mean(coordinates, axis=0)
    relative_pos = coordinates - center
    features_list.append(relative_pos)
    
    distances = np.linalg.norm(relative_pos, axis=1).reshape(-1, 1)
    angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0]).reshape(-1, 1)
    features_list.extend([distances, angles])
    
    k = min(10, N-1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(coordinates)
    distances, _ = nbrs.kneighbors(coordinates)
    local_density = np.mean(distances, axis=1).reshape(-1, 1)
    features_list.append(local_density)
    
    grid_size = 8
    pe_x = np.zeros((N, grid_size))
    pe_y = np.zeros((N, grid_size))
    for i in range(grid_size):
        pe_x[:, i] = np.sin(2 ** i * np.pi * coordinates[:, 0])
        pe_y[:, i] = np.cos(2 ** i * np.pi * coordinates[:, 1])
    features_list.extend([pe_x, pe_y])
    
    node_features = np.hstack(features_list)
    if normalize:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
    return torch.FloatTensor(node_features)

def load_city_data(city_name, adj_dir, coord_dir):
    coord_path = os.path.join(coord_dir, f'{city_name}_coords.npy')
    adj_path = os.path.join(adj_dir, f'{city_name}_adj.npy')
    coords = np.load(coord_path)
    adj_matrix = np.load(adj_path)
    adj_matrix = (adj_matrix > 0).astype(float)
    coordinates = np.column_stack((coords['y'], coords['x'])).astype(np.float32)
    node_features = create_node_features(coordinates)
    adj_tensor = torch.FloatTensor(adj_matrix)
    edge_index, _ = dense_to_sparse(adj_tensor)
    return edge_index, adj_matrix, node_features, coordinates

def plot_centrality_comparisons(metrics):
    # Closeness Centrality
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Generated'], 
            [np.mean(metrics['closeness_original']), np.mean(metrics['closeness_generated'])],
            color=[COLORS['original'], COLORS['generated']])
    plt.title('Average Closeness Centrality Comparison')
    plt.ylabel('Closeness Centrality')
    plt.savefig(os.path.join(save_path, 'closeness_centrality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Betweenness Centrality
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Generated'], 
            [np.mean(metrics['betweenness_original']), np.mean(metrics['betweenness_generated'])],
            color=[COLORS['betweenness'], COLORS['generated']])
    plt.title('Average Betweenness Centrality Comparison')
    plt.ylabel('Betweenness Centrality')
    plt.savefig(os.path.join(save_path, 'betweenness_centrality.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_degree_distribution_comparison(metrics):
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['degrees_original'], bins=20, alpha=0.7, 
             label='Original', color=COLORS['original'], density=True)
    plt.hist(metrics['degrees_generated'], bins=20, alpha=0.7, 
             label='Generated', color=COLORS['generated'], density=True)
    plt.title('Degree Distribution Comparison')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'degree_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load model
    model = VGAE(encoder=VariationalGCNEncoder(in_channels=23, hidden_channels=512, out_channels=256, dropout=0.3))
    model.load_state_dict(torch.load('model_checkpoints/best_vgae_model_10_50_world_center.pt', 
                                   map_location='cpu',
                                   weights_only=True))
    model.eval()

    adj_dir = 'data/adj_matrices/world/center/test'
    coord_dir = 'data/coordinates/world/center/transformed/test'

    city_files = sorted(os.listdir(adj_dir))[:10]  # Using first 10 cities
    selected_cities = [city.replace('_adj.npy', '') for city in city_files]

    metrics = {
        'degrees_original': [], 'degrees_generated': [],
        'closeness_original': [], 'closeness_generated': [],
        'betweenness_original': [], 'betweenness_generated': []
    }
    
    for city_name in selected_cities:
        edge_index, adj_matrix, features, coords = load_city_data(city_name, adj_dir, coord_dir)
        
        with torch.no_grad():
            z = model.encode(features, edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).numpy()
            pred_adj_binary = (pred_adj > 0.9).astype(int)

        # Collect network metrics for both original and generated
        G_original = nx.from_numpy_array(adj_matrix)
        G_generated = nx.from_numpy_array(pred_adj_binary)
        
        # Degrees
        metrics['degrees_original'].extend(dict(G_original.degree()).values())
        metrics['degrees_generated'].extend(dict(G_generated.degree()).values())
        
        # Centrality measures
        metrics['closeness_original'].extend(nx.closeness_centrality(G_original).values())
        metrics['closeness_generated'].extend(nx.closeness_centrality(G_generated).values())
        metrics['betweenness_original'].extend(nx.betweenness_centrality(G_original).values())
        metrics['betweenness_generated'].extend(nx.betweenness_centrality(G_generated).values())
    
    # Generate plots
    plot_centrality_comparisons(metrics)
    plot_degree_distribution_comparison(metrics)

if __name__ == "__main__":
    main()
