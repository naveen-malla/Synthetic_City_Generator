import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges, to_undirected, dense_to_sparse
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import Parameter

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess adjacency matrix
adj_matrix = pd.read_csv('adjacency_matrix.csv', header=None).values
adj_matrix = adj_matrix.astype(float)
adj_matrix = (adj_matrix > 0).astype(float)

# Create enhanced node features
degrees = adj_matrix.sum(axis=1)
in_degrees = adj_matrix.sum(axis=0)
out_degrees = adj_matrix.sum(axis=1)
clustering_coeff = np.zeros(adj_matrix.shape[0])

for i in range(adj_matrix.shape[0]):
    neighbors = np.where(adj_matrix[i] > 0)[0]
    if len(neighbors) > 1:
        sub_graph = adj_matrix[np.ix_(neighbors, neighbors)]
        clustering_coeff[i] = sub_graph.sum() / (len(neighbors) * (len(neighbors) - 1))

# Combine features
node_features = np.column_stack([
    degrees.reshape(-1, 1),
    in_degrees.reshape(-1, 1),
    out_degrees.reshape(-1, 1),
    clustering_coeff.reshape(-1, 1)
])

# Normalize features
node_features = (node_features - node_features.mean(axis=0)) / node_features.std(axis=0)

# Convert to PyTorch tensors
adj_tensor = torch.FloatTensor(adj_matrix)
node_features = torch.FloatTensor(node_features)

# Create PyGeometric Data object
edge_index, _ = dense_to_sparse(adj_tensor)
data = Data(x=node_features, edge_index=edge_index)

# Prepare data with smaller test split
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

class CustomVGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomVGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        
        # Residual connection weights
        self.w_residual = Parameter(torch.ones(1))
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        x1 = F.elu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.elu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        # Residual connection
        x_combined = x2 + self.w_residual * x1
        return self.conv_mu(x_combined, edge_index), self.conv_logstd(x_combined, edge_index)

class CustomVGAE(VGAE):
    def __init__(self, encoder, decoder=None):
        super(CustomVGAE, self).__init__(encoder, decoder)
        
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        # Use torch_geometric's negative_sampling directly
        if neg_edge_index is None:
            from torch_geometric.utils import negative_sampling
            neg_edge_index = negative_sampling(
                pos_edge_index,
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        return pos_loss + neg_loss

# Initialize model with custom components
model = CustomVGAE(CustomVGAEEncoder(
    in_channels=node_features.shape[1],
    hidden_channels=256,
    out_channels=128
))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

# Training with improved monitoring
best_val_auc = 0
patience = 50
counter = 0
best_model_state = None

print("Training VGAE on street network...")
for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    kl_loss = (0.1 / data.num_nodes) * model.kl_loss()  # Reduced KL impact
    total_loss = loss + kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    scheduler.step(auc)
    
    if auc > best_val_auc:
        best_val_auc = auc
        counter = 0
        best_model_state = model.state_dict().copy()
    else:
        counter += 1
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# Final evaluation
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    final_auc, final_ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    pred_adj = torch.sigmoid(torch.matmul(z, z.t())).detach().cpu().numpy()

print(f'\nFinal Test AUC: {final_auc:.4f}')
print(f'Final Test AP: {final_ap:.4f}')

# Save results and print comparison
np.save('predicted_adjacency.npy', pred_adj)
print("\nOriginal Adjacency Matrix (subset):")
print(adj_matrix[:5, :5])
print("\nPredicted Adjacency Matrix (subset):")
print(pred_adj[:5, :5])