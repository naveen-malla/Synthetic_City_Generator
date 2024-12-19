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
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def load_and_process_data(data_dir):
    matrices = []
    features_list = []
    
    cities = [f.split('_')[0] for f in os.listdir(os.path.join(data_dir, 'adj_matrices')) 
             if f.endswith('_adj.npy')]
    
    for city in cities:
        # Load adjacency matrix
        adj_file = os.path.join(data_dir, 'adj_matrices', f'{city}_adj.npy')
        adj_matrix = np.load(adj_file)
        adj_matrix = (adj_matrix > 0).astype(float)
        
        # Load coordinates
        coord_file = os.path.join(data_dir, 'coordinates/transformed', f'{city}_coords.npy')
        coords = np.load(coord_file)
        coordinates = np.column_stack((coords['y'], coords['x'])).astype(np.float32)
        
        # Convert to torch tensors
        node_features = torch.FloatTensor(coordinates)
        edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))
        
        matrices.append((edge_index, adj_matrix))
        features_list.append(node_features)
        print(f"Loaded {city}: Shape {adj_matrix.shape}")
    
    return matrices, features_list

def train_epoch(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, edge_index)
    
    # Get dense adjacency from edge_index
    adj_orig = to_dense_adj(edge_index)[0]
    
    # Calculate reconstruction using decoder
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
    
    # Calculate class weights
    num_edges = edge_index.shape[1]
    num_nodes = data.x.size(0)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    pos_weight = torch.tensor((num_possible_edges - num_edges) / num_edges)
    
    # Use BCE loss with pos_weight directly
    loss = F.binary_cross_entropy_with_logits(
        adj_pred.view(-1),
        adj_orig.view(-1),
        pos_weight=pos_weight
    )
    
    kl_loss = (1 / data.x.size(0)) * model.kl_loss()
    total_loss = loss + kl_loss
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss


def main():
    # Load data
    matrices, features_list = load_and_process_data('data')
    
    # Model parameters
    in_channels = 2  # x,y coordinates
    hidden_channels = 256
    out_channels = 128
    
    # Initialize model
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
            data = Data(x=features, edge_index=edge_index)
            loss = train_epoch(model, optimizer, data, edge_index)
            total_loss += loss.item()
        
        # Validation on random city
        val_idx = np.random.randint(len(matrices))
        val_edge_index, val_adj = matrices[val_idx]
        val_features = features_list[val_idx]
        
        model.eval()
        with torch.no_grad():
            z = model.encode(val_features, val_edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
            auc = roc_auc_score(val_adj.flatten(), pred_adj.flatten())
            ap = average_precision_score(val_adj.flatten(), pred_adj.flatten())
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(matrices):.4f}, '
                  f'Val AUC: {auc:.4f}, Val AP: {ap:.4f}')

if __name__ == "__main__":
    main()
