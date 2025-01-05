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

def main():
    min_nodes = 10
    max_nodes = 100
    matrices, features_list, processed_cities = load_and_process_matrices('data', min_nodes, max_nodes)
    print(f"\nTraining on {len(processed_cities)} cities within size range {min_nodes}-{max_nodes}")

    model = VGAE(
        encoder=VariationalGCNEncoder(
            in_channels=2,  # Only x,y coordinates
            hidden_channels=512,
            out_channels=256,
            dropout=0.3
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(1, 301):
        total_loss = 0

        for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
            data = Data(x=features.to(device), edge_index=edge_index.to(device))
            loss = train_epoch(model, optimizer, data)
            total_loss += loss

        # Validation
        val_idx = np.random.randint(len(matrices))
        val_edge_index, val_adj = matrices[val_idx]
        val_features = features_list[val_idx]

        model.eval()
        with torch.no_grad():
            z = model.encode(val_features.to(device), val_edge_index.to(device))
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            
            # Calculate both metrics
            auc = roc_auc_score(val_adj.flatten(), pred_adj.flatten())
            ap = average_precision_score(val_adj.flatten(), pred_adj.flatten())

        scheduler.step(auc)

        if auc > best_val_auc:
            best_val_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_vgae_model.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(matrices):.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')

        if patience_counter >= 50:
            print(f'Early stopping at epoch {epoch}')
            break

    # Final evaluation
    print("\nFinal Evaluation on Each City:")
    model.load_state_dict(torch.load("best_vgae_model.pt", map_location=device))
    model.eval()

    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        with torch.no_grad():
            z = model.encode(features.to(device), edge_index.to(device))
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            
            # Calculate both metrics for final evaluation
            auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
            ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
            print(f"{processed_cities[idx]} - Test AUC: {auc:.4f}, AP: {ap:.4f}")

if __name__ == "__main__":
    main()
