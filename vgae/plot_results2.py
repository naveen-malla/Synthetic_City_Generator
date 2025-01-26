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
MODEL_DIR = 'vgae_best_model_10_50'
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

def plot_centrality_and_degree(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Degree Distribution
    ax1.hist(metrics['degrees_original'], bins=np.logspace(0, 3, 20), alpha=0.7, 
             label='Original', color=COLORS['original'], density=True)
    ax1.hist(metrics['degrees_generated'], bins=np.logspace(0, 3, 20), alpha=0.7, 
             label='Generated', color=COLORS['generated'], density=True)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Degree Distribution')
    ax1.set_xlabel('Degree (log scale)')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Closeness Centrality
    sns.boxplot(data=[metrics['closeness_original'], metrics['closeness_generated']], 
                ax=ax2, palette=[COLORS['original'], COLORS['generated']])
    ax2.set_xticks([0, 1])  
    ax2.set_xticklabels(['Original', 'Generated'])
    ax2.set_title('Closeness Centrality Distribution')
    ax2.set_ylabel('Closeness Centrality')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'degree_and_centrality.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_length_distributions(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Path Length Distribution
    sns.boxplot(data=[metrics['path_length_original'], metrics['path_length_generated']], 
                ax=ax1, palette=[COLORS['original'], COLORS['generated']])
    ax1.set_xticks([0, 1])  
    ax1.set_xticklabels(['Original', 'Generated'])
    ax1.set_title('Average Path Length Distribution')
    ax1.set_ylabel('Path Length')
    
    # Street Length Distribution
    sns.kdeplot(data=metrics['street_length_original'], ax=ax2,
            color=COLORS['original'],
            alpha=0.8, label='Original',
            fill=True)
    sns.kdeplot(data=metrics['street_length_generated'], ax=ax2,
                color=COLORS['generated'],
                alpha=0.8, label='Generated',
                fill=True)
    ax2.set_title('Street Length Distribution')
    ax2.set_xlabel('Street Length')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'length_distributions.png'), dpi=300, bbox_inches='tight')
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

    city_files = sorted(os.listdir(adj_dir))[:10]
    selected_cities = [city.replace('_adj.npy', '') for city in city_files]
    
    print(f"\nProcessing {len(selected_cities)} cities...")
    print("=" * 50)

    metrics = {
        'degrees_original': [], 'degrees_generated': [],
        'closeness_original': [], 'closeness_generated': [],
        'path_length_original': [], 'path_length_generated': [],
        'street_length_original': [], 'street_length_generated': []
    }
    
    for i, city_name in enumerate(selected_cities):
        # Progress tracking
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(selected_cities)} cities ({(i + 1)/len(selected_cities)*100:.1f}%)")
        
        edge_index, adj_matrix, features, coords = load_city_data(city_name, adj_dir, coord_dir)
        
        with torch.no_grad():
            z = model.encode(features, edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).numpy()
            pred_adj_binary = (pred_adj > 0.9).astype(int)

        # Collect network metrics
        G_original = nx.from_numpy_array(adj_matrix)
        G_generated = nx.from_numpy_array(pred_adj_binary)
        
        # Degrees
        metrics['degrees_original'].extend(dict(G_original.degree()).values())
        metrics['degrees_generated'].extend(dict(G_generated.degree()).values())
        
        # Closeness centrality
        metrics['closeness_original'].extend(nx.closeness_centrality(G_original).values())
        metrics['closeness_generated'].extend(nx.closeness_centrality(G_generated).values())
        
        # Path length
        if nx.is_connected(G_original):
            metrics['path_length_original'].append(nx.average_shortest_path_length(G_original))
        if nx.is_connected(G_generated):
            metrics['path_length_generated'].append(nx.average_shortest_path_length(G_generated))
        
        # Street lengths
        for (i, j) in G_original.edges():
            length = np.linalg.norm(coords[i] - coords[j])
            metrics['street_length_original'].append(length)
        for (i, j) in G_generated.edges():
            length = np.linalg.norm(coords[i] - coords[j])
            metrics['street_length_generated'].append(length)
    
    print("\nGenerating plots...")
    # Generate plots
    plot_centrality_and_degree(metrics)
    plot_length_distributions(metrics)
    print("\nDone! Plots saved in:", save_path)
    print("=" * 50)

if __name__ == "__main__":
    main()
