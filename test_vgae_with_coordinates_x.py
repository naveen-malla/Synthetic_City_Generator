import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed
torch.manual_seed(42)

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
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        return self.conv_mu(x3, edge_index), self.conv_logstd(x3, edge_index)

class ImprovedVGAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 2, latent_dim),  # +2 for coordinates
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, z, x):
        h = torch.cat([z, x], dim=1)
        return self.decoder(h)

def load_and_process_matrices(data_dir, min_nodes, max_nodes):
    matrices = []
    features_list = []
    processed_cities = []
    
    print(f"\nProcessing cities with size range: {min_nodes} - {max_nodes}")
    print("-" * 50)
    
    try:
        adj_dir = os.path.join(data_dir, 'adj_matrices')
        coord_dir = os.path.join(data_dir, 'coordinates/transformed')
        
        # Verify directories exist
        if not os.path.exists(adj_dir) or not os.path.exists(coord_dir):
            raise ValueError(f"Required directories not found in {data_dir}")
        
        # Get list of cities from adjacency matrices
        cities = [f.split('_adj.npy')[0] for f in os.listdir(adj_dir) 
                 if f.endswith('_adj.npy')]
        
        print(f"Found {len(cities)} potential cities")
        
        for city in cities:
            adj_file = os.path.join(adj_dir, f'{city}_adj.npy')
            coord_file = os.path.join(coord_dir, f'{city}_coords.npy')
            
            # Check file existence
            if not os.path.exists(adj_file) or not os.path.exists(coord_file):
                print(f"Skipping {city}: Missing required files")
                continue
            
            # Load adjacency matrix
            try:
                adj_matrix = np.load(adj_file)
                matrix_size = adj_matrix.shape[0]
                
                # Check size constraints
                if matrix_size < min_nodes or matrix_size > max_nodes:
                    # print(f"Skipping {city}: Size {matrix_size} outside range [{min_nodes}, {max_nodes}]")
                    continue
                
                # Convert to binary adjacency matrix
                adj_matrix = (adj_matrix > 0).astype(float)
                
                # Load and process coordinates
                structured_coords = np.load(coord_file)
                coordinates = np.column_stack((structured_coords['y'], 
                                            structured_coords['x'])).astype(np.float32)
                
                # Verify dimensions match
                if coordinates.shape[0] != matrix_size:
                    print(f"Skipping {city}: Size mismatch - Adj: {matrix_size}, Coords: {coordinates.shape[0]}")
                    continue
                
                # Convert to torch tensors
                node_features = torch.FloatTensor(coordinates)
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
        print(f"Processed cities: {', '.join(processed_cities)}")
        
        if len(matrices) == 0:
            raise ValueError(f"No valid cities found within size range {min_nodes}-{max_nodes}")
        
        return matrices, features_list, processed_cities
        
    except Exception as e:
        print(f"Fatal error in data loading: {str(e)}")
        raise




def train_epoch(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    
    # Get latent representation
    z = model.encode(data.x, edge_index)
    
    # Reconstruct adjacency matrix using decoder
    # First, get dense adjacency from edge_index
    adj_orig = to_dense_adj(edge_index)[0]
    
    # Decode and get reconstructed adjacency
    decoded = model.decoder(z, data.x)
    adj_pred = torch.sigmoid(torch.matmul(decoded, decoded.t()))
    
    # Calculate class weights
    num_edges = edge_index.shape[1]
    num_nodes = data.x.size(0)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    pos_weight = torch.tensor((num_possible_edges - num_edges) / num_edges)
    
    # Calculate loss with proper dimensions
    loss = F.binary_cross_entropy_with_logits(
        adj_pred.view(-1),
        adj_orig.view(-1),
        pos_weight=pos_weight
    )
    
    kl_loss = (0.05 / data.x.size(0)) * model.kl_loss()
    total_loss = loss + kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss


def main():
    # Set size range for cities
    min_nodes = 50  # Minimum number of nodes
    max_nodes = 100  # Maximum number of nodes
    
    # Load data with size constraints
    matrices, features_list, processed_cities = load_and_process_matrices('data', min_nodes, max_nodes)
    
    print(f"\nTraining on {len(processed_cities)} cities within size range {min_nodes}-{max_nodes}")

    model = VGAE(
        encoder=ImprovedVGAEEncoder(
            in_channels=2,  # Only x,y coordinates
            hidden_channels=512,
            out_channels=256,
            dropout=0.3
        ),
        decoder=ImprovedVGAEDecoder(latent_dim=256)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)
    
    best_val_auc = 0
    patience = 50
    counter = 0
    
    print("Training VGAE on multiple street networks...")
    for epoch in range(1, 1001):
        total_loss = 0
        
        for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
            data = Data(x=features, edge_index=edge_index)
            loss = train_epoch(model, optimizer, data, edge_index)
            total_loss += loss.item()
        
        val_idx = np.random.randint(len(matrices))
        val_edge_index, val_adj = matrices[val_idx]
        val_features = features_list[val_idx]
        
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
    
    checkpoint = torch.load('best_vgae_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nFinal Evaluation on Each City:")
    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        with torch.no_grad():
            z = model.encode(features, edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
            ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
            print(f"{processed_cities[idx]} - Test AUC: {auc:.4f}, AP: {ap:.4f}")

if __name__ == "__main__":
    main()
