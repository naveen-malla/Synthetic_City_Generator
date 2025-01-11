import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Set random seed
torch.manual_seed(42)

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        return self.conv_mu(x3, edge_index), self.conv_logstd(x3, edge_index)

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

def load_and_process_matrices(data_dir, min_nodes, max_nodes):
    matrices = []
    features_list = []
    processed_cities = []

    print(f"\nProcessing cities with size range: {min_nodes} - {max_nodes}")
    print("-" * 50)

    adj_dir = os.path.join(data_dir, 'adj_matrices/center')
    coord_dir = os.path.join(data_dir, 'coordinates/center/transformed')

    if not os.path.exists(adj_dir) or not os.path.exists(coord_dir):
        raise ValueError(f"Required directories not found in {data_dir}")

    cities = [f.split('_adj.npy')[0] for f in os.listdir(adj_dir) if f.endswith('_adj.npy')]
    print(f"Found {len(cities)} potential cities")

    for city in cities:
        adj_file = os.path.join(adj_dir, f'{city}_adj.npy')
        coord_file = os.path.join(coord_dir, f'{city}_coords.npy')

        if not os.path.exists(adj_file) or not os.path.exists(coord_file):
            print(f"Skipping {city}: Missing required files")
            continue

        try:
            adj_matrix = np.load(adj_file)
            matrix_size = adj_matrix.shape[0]

            if matrix_size < min_nodes or matrix_size > max_nodes:
                continue

            adj_matrix = (adj_matrix > 0).astype(float)
            structured_coords = np.load(coord_file)
            coordinates = np.column_stack((structured_coords['y'], structured_coords['x'])).astype(np.float32)

            if coordinates.shape[0] != matrix_size:
                print(f"Skipping {city}: Size mismatch - Adj: {matrix_size}, Coords: {coordinates.shape[0]}")
                continue

            node_features = create_node_features(coordinates)
            edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))

            matrices.append((edge_index, adj_matrix))
            features_list.append(node_features)
            processed_cities.append(city)
            #print(f"Loaded {city}: Shape {adj_matrix.shape}")

        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            continue

    print(f"\nSuccessfully loaded {len(matrices)} cities")
    print(f"Size range: {min_nodes}-{max_nodes} nodes")

    if len(matrices) == 0:
        raise ValueError(f"No valid cities found within size range {min_nodes}-{max_nodes}")

    return matrices, features_list, processed_cities

def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x.to(device), data.edge_index.to(device))
    adj_orig = to_dense_adj(data.edge_index)[0]
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

    num_edges = data.edge_index.shape[1]
    num_nodes = data.x.size(0)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    pos_weight = torch.tensor((num_possible_edges - num_edges) / num_edges).to(device)

    loss = F.binary_cross_entropy_with_logits(
        adj_pred.view(-1),
        adj_orig.view(-1).to(device),
        pos_weight=pos_weight
    )

    kl_loss = (0.05 / data.x.size(0)) * model.kl_loss()
    total_loss = loss + kl_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def split_dataset(matrices, features_list, processed_cities, train_ratio=0.7, val_ratio=0.15):
    n = len(matrices)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    def subset(data, idx):
        return [data[i] for i in idx]
        
    return (subset(matrices, train_indices), subset(features_list, train_indices), subset(processed_cities, train_indices)), \
           (subset(matrices, val_indices), subset(features_list, val_indices), subset(processed_cities, val_indices)), \
           (subset(matrices, test_indices), subset(features_list, test_indices), subset(processed_cities, test_indices))

def evaluate(model, data, adj_matrix):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
        binary_pred_adj = (pred_adj > 0.5).astype(float)
        auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
        ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
        accuracy = (binary_pred_adj == adj_matrix).mean()
    return auc, ap, accuracy, binary_pred_adj

def final_evaluation(model, matrices, features_list, processed_cities):
    print("\nFinal Evaluation:")
    results = []
    predicted_adjs = {}
    
    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        with torch.no_grad():
            data = Data(x=features, edge_index=edge_index).to(device)
            z = model.encode(data.x, data.edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            binary_pred_adj = (pred_adj > 0.5).astype(float)
            
        auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
        ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
        accuracy = (binary_pred_adj == adj_matrix).mean()
        
        results.append({
            'City': processed_cities[idx],
            'Nodes': adj_matrix.shape[0],
            'AUC': auc,
            'AP': ap,
            'Accuracy': accuracy
        })
        predicted_adjs[processed_cities[idx]] = binary_pred_adj
    
    df = pd.DataFrame(results)
    
    print("\nSummary Statistics:")
    print(df[['AUC', 'AP', 'Accuracy']].describe())
    
    print("\nTop 5 Cities by AUC:")
    print(tabulate(df.nlargest(5, 'AUC')[['City', 'Nodes', 'AUC', 'AP', 'Accuracy']], 
                  headers='keys', tablefmt='pretty'))
    
    print("\nBottom 5 Cities by AUC:")
    print(tabulate(df.nsmallest(5, 'AUC')[['City', 'Nodes', 'AUC', 'AP', 'Accuracy']], 
                  headers='keys', tablefmt='pretty'))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(df['AUC'], bins=20, edgecolor='black')
    axes[0].set_title('Distribution of AUC Scores')
    axes[0].set_xlabel('AUC')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df['AP'], bins=20, edgecolor='black')
    axes[1].set_title('Distribution of AP Scores')
    axes[1].set_xlabel('AP')
    
    axes[2].hist(df['Accuracy'], bins=20, edgecolor='black')
    axes[2].set_title('Distribution of Accuracy Scores')
    axes[2].set_xlabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.close()
    
    df['Size'] = pd.cut(df['Nodes'], bins=[0, 60, 80, 100], labels=['Small', 'Medium', 'Large'])
    size_performance = df.groupby('Size')[['AUC', 'AP', 'Accuracy']].mean()
    
    print("\nPerformance by City Size:")
    print(tabulate(size_performance, headers='keys', tablefmt='pretty'))
    
    return df, predicted_adjs

def main():
    min_nodes = 10
    max_nodes = 100
    matrices, features_list, processed_cities = load_and_process_matrices('data', min_nodes, max_nodes)
    train_data, val_data, test_data = split_dataset(matrices, features_list, processed_cities)
    print(f"\nTraining on {len(train_data[0])} cities within size range {min_nodes}-{max_nodes}")

    # Initialize model
    model = VGAE(
        encoder=VariationalGCNEncoder(
            in_channels=23,  
            hidden_channels=512,
            out_channels=256,
            dropout=0.3
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(1, 1001):
        total_loss = 0
        # Train on batches
        for (edge_index, adj_matrix), features in zip(*train_data[:2]):
            data = Data(x=features, edge_index=edge_index)
            loss = train_epoch(model, optimizer, data)
            total_loss += loss

        # Validation
        val_aucs = []
        val_aps = []
        for (val_edge_index, val_adj), val_features in zip(val_data[0], val_data[1]):
            val_data_obj = Data(x=val_features, edge_index=val_edge_index)
            val_auc, val_ap, accuracy, binary_pred_adj = evaluate(model, val_data_obj, val_adj)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)

        avg_val_auc = np.mean(val_aucs)
        avg_val_ap = np.mean(val_aps)
        
        scheduler.step(avg_val_auc)

        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_vgae_model.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(train_data[0]):.4f}, '
                  f'Val AUC: {avg_val_auc:.4f}, Val AP: {avg_val_ap:.4f}')

        if patience_counter >= 50:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load best model and evaluate
    model.load_state_dict(torch.load("best_vgae_model.pt"))
    model.eval()
    results_df = final_evaluation(model, test_data[0], test_data[1], test_data[2])

if __name__ == "__main__":
    main()
