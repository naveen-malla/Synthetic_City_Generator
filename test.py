import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges, to_undirected, dense_to_sparse
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess adjacency matrix
adj_matrix = pd.read_csv('adjacency_matrix.csv', header=None).values
adj_matrix = adj_matrix.astype(float)  # Convert scientific notation to float
adj_matrix = (adj_matrix > 0).astype(float)  # Convert to binary adjacency matrix

# Create node features using structural properties
degrees = adj_matrix.sum(axis=1)
clustering_coeff = np.zeros(adj_matrix.shape[0])
for i in range(adj_matrix.shape[0]):
    neighbors = np.where(adj_matrix[i] > 0)[0]
    if len(neighbors) > 1:
        sub_graph = adj_matrix[np.ix_(neighbors, neighbors)]
        clustering_coeff[i] = sub_graph.sum() / (len(neighbors) * (len(neighbors) - 1))

# Combine features
node_features = np.column_stack([
    degrees,
    clustering_coeff,
    np.random.normal(0, 0.1, (adj_matrix.shape[0], 3))  # Add some noise features
])

# Convert to PyTorch tensors
adj_tensor = torch.FloatTensor(adj_matrix)
node_features = torch.FloatTensor(node_features)

# Create PyGeometric Data object
edge_index, _ = dense_to_sparse(adj_tensor)
data = Data(
    x=node_features,
    edge_index=edge_index
)

# Store original edge_index and prepare data
original_edge_index = data.edge_index.clone()
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

class ImprovedVariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ImprovedVariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Initialize model with larger dimensions
model = VGAE(ImprovedVariationalGCNEncoder(
    in_channels=node_features.shape[1],
    hidden_channels=512,
    out_channels=256
))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

# Training loop with early stopping
best_val_auc = 0
patience = 30
counter = 0
best_model_state = None

print("Training VGAE on street network...")
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    kl_loss = (1 / data.num_nodes) * model.kl_loss()
    loss = loss + 0.1 * kl_loss  # Reduce KL loss impact
    loss.backward()
    optimizer.step()
    
    # Validation
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    
    if auc > best_val_auc:
        best_val_auc = auc
        counter = 0
        best_model_state = model.state_dict().copy()
    else:
        counter += 1
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# Load best model and evaluate
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    final_auc, final_ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    pred_adj = torch.sigmoid(torch.matmul(z, z.t())).detach().cpu().numpy()

print(f'\nFinal Test AUC: {final_auc:.4f}')
print(f'Final Test AP: {final_ap:.4f}')

# Save results
np.save('predicted_adjacency.npy', pred_adj)
print("\nOriginal Adjacency Matrix (subset):")
print(adj_matrix[:5, :5])
print("\nPredicted Adjacency Matrix (subset):")
print(pred_adj[:5, :5])