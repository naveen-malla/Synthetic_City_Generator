import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score

# Set random seed
torch.manual_seed(42)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def load_and_process_matrices(data_dir, min_nodes, max_nodes):
    matrices = []
    features_list = []
    processed_cities = []
    
    print(f"\nProcessing cities with size range: {min_nodes} - {max_nodes}")
    print("-" * 50)
    
    try:
        adj_dir = os.path.join(data_dir, 'adj_matrices/center')
        coord_dir = os.path.join(data_dir, 'coordinates/center/transformed')
        
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

def main():
    # Load data with size constraints
    matrices, features_list, processed_cities = load_and_process_matrices('data', min_nodes=50, max_nodes=100)
    print(f"\nTraining on {len(processed_cities)} cities within size range 50-100")

    out_channels = 16  # Increased from 2 for better representation
    num_features = features_list[0].shape[1]  # Should be 2 for x, y coordinates
    model = VGAE(VariationalGCNEncoder(num_features, out_channels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    for epoch in range(1, 301):
        total_loss = 0
        for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
            data = Data(x=features.to(device), edge_index=edge_index.to(device))
            x = data.x
            train_pos_edge_index = data.edge_index

            loss = train()
            total_loss += loss

        if epoch % 10 == 0:
            # Use the first city for validation
            val_edge_index, val_adj = matrices[0]
            val_features = features_list[0]
            val_data = Data(x=val_features.to(device), edge_index=val_edge_index.to(device))
            
            pos_edge_index = val_data.edge_index
            neg_edge_index = torch.randint(0, val_data.num_nodes, pos_edge_index.shape, device=device)
            
            auc, ap = test(pos_edge_index, neg_edge_index)
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(matrices):.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')

    print("\nFinal Evaluation on Each City:")
    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        data = Data(x=features.to(device), edge_index=edge_index.to(device))
        pos_edge_index = data.edge_index
        neg_edge_index = torch.randint(0, data.num_nodes, pos_edge_index.shape, device=device)
        auc, ap = test(pos_edge_index, neg_edge_index)
        print(f"{processed_cities[idx]} - Test AUC: {auc:.4f}, AP: {ap:.4f}")

if __name__ == "__main__":
    main()
