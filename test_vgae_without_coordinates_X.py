import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed
torch.manual_seed(42)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class ImprovedVGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImprovedVGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = x.to(device)
        edge_index = edge_index.to(device)
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        return self.conv_mu(x3, edge_index), self.conv_logstd(x3, edge_index)

def create_node_features(adj_matrix):
    degrees = adj_matrix.sum(axis=1)
    in_degrees = adj_matrix.sum(axis=0)
    clustering = np.zeros(adj_matrix.shape[0])
    
    for i in range(adj_matrix.shape[0]):
        neighbors = np.where(adj_matrix[i] > 0)[0]
        if len(neighbors) > 1:
            sub_graph = adj_matrix[np.ix_(neighbors, neighbors)]
            clustering[i] = sub_graph.sum() / (len(neighbors) * (len(neighbors) - 1))
    
    features = np.column_stack([
        degrees,
        in_degrees.reshape(-1),
        clustering,
        np.random.normal(0, 0.1, (adj_matrix.shape[0], 2))
    ])
    
    features = (features - features.mean(0)) / (features.std(0) + 1e-8)
    return torch.FloatTensor(features)

def load_and_process_matrices(data_dir, min_nodes, max_nodes):
    matrices = []
    features_list = []
    processed_cities = []
    
    print(f"\nProcessing cities with size range: {min_nodes} - {max_nodes}")
    print("-" * 50)
    
    try:
        # Get list of cities from adjacency matrices
        adj_files = [f for f in os.listdir(data_dir) if f.endswith('_adj.npy')]
        
        print(f"Found {len(adj_files)} potential cities")
        
        for adj_file in adj_files:
            city = adj_file.replace('_adj.npy', '')
            
            # Load and check adjacency matrix size
            adj_matrix = np.load(os.path.join(data_dir, adj_file))
            matrix_size = adj_matrix.shape[0]
            
            # Skip if size is outside range
            if matrix_size < min_nodes or matrix_size > max_nodes:
                #print(f"Skipping {city}: Size {matrix_size} outside range [{min_nodes}, {max_nodes}]")
                continue
                
            adj_matrix = (adj_matrix > 0).astype(float)
            
            try:
                # Create node features
                node_features = create_node_features(adj_matrix)
                edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))
                
                matrices.append((edge_index, adj_matrix))
                features_list.append(node_features)
                processed_cities.append(city)
                print(f"Loaded {city}: Shape {adj_matrix.shape}")
                
            except Exception as e:
                print(f"Error processing {city}: {str(e)}")
                continue
        
        print(f"\nSuccessfully loaded {len(matrices)} cities")
        print(f"Size range: {min_nodes}-{max_nodes} nodes")
        
        if len(matrices) == 0:
            raise ValueError(f"No valid cities found within size range {min_nodes}-{max_nodes}")
        
        return matrices, features_list, processed_cities
        
    except Exception as e:
        print(f"Fatal error in data loading: {str(e)}")
        raise

def train_epoch(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, edge_index)
    loss = model.recon_loss(z, edge_index)
    kl_loss = (0.05 / data.x.size(0)) * model.kl_loss()  # Further reduced KL impact
    total_loss = loss + kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss

def main():
    # Set size range for cities
    min_nodes = 10  # Minimum number of nodes
    max_nodes = 100  # Maximum number of nodes
    
    # Load data with size constraints
    matrices, features_list, processed_cities = load_and_process_matrices('data/adj_matrices', min_nodes, max_nodes)
    
    in_channels = features_list[0].shape[1]
    model = VGAE(ImprovedVGAEEncoder(
        in_channels=in_channels,
        hidden_channels=512,
        out_channels=256,
        dropout=0.3
    )).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)
    
    best_val_auc = 0
    patience = 50
    counter = 0
    
    print("Training VGAE on multiple street networks...")
    for epoch in range(1, 1001):
        total_loss = 0
        
        for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
            data = Data(x=features, edge_index=edge_index).to(device)
            loss = train_epoch(model, optimizer, data, edge_index.to(device))
            total_loss += loss.item()
        
        val_idx = np.random.randint(len(matrices))
        val_edge_index, val_adj = matrices[val_idx]
        val_features = features_list[val_idx].to(device)
        val_edge_index = val_edge_index.to(device)
        
        model.eval()
        with torch.no_grad():
            z = model.encode(val_features, val_edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            auc = roc_auc_score(val_adj.flatten(), pred_adj.flatten())
            ap = average_precision_score(val_adj.flatten(), pred_adj.flatten())
        
        scheduler.step(auc)
        
        if auc > best_val_auc:
            best_val_auc = auc
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc
            }, 'best_vgae_model.pt')
        else:
            counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(matrices):.4f}, '
                  f'Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
        
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    checkpoint = torch.load('best_vgae_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nFinal Evaluation on Each City:")
    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        with torch.no_grad():
            z = model.encode(features.to(device), edge_index.to(device))
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
            ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
            print(f"{processed_cities[idx]} - Test AUC: {auc:.4f}, AP: {ap:.4f}")

if __name__ == "__main__":
    main()
