import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges, to_undirected, to_dense_adj
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Set random seed for reproducibility
torch.manual_seed(42)

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Store original edge_index
original_edge_index = data.edge_index.clone()

# Prepare the data
data.train_mask = data.val_mask = data.test_mask = data.y = None
data.edge_index = to_undirected(data.edge_index)
data = train_test_split_edges(data)

class ImprovedVariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ImprovedVariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Initialize improved model with larger dimensions
model = VGAE(ImprovedVariationalGCNEncoder(dataset.num_features, 256, 128))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

# Training loop with early stopping
best_val_auc = 0
patience = 20
counter = 0
best_epoch = 0

print("Training VGAE on Cora dataset...")
for epoch in range(1, 501):
    loss = train()
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    
    if auc > best_val_auc:
        best_val_auc = auc
        counter = 0
        best_epoch = epoch
    else:
        counter += 1
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# Final evaluation
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    final_auc, final_ap = test(data.test_pos_edge_index, data.test_neg_edge_index)

print(f'\nFinal Test AUC: {final_auc:.4f}')
print(f'Final Test AP: {final_ap:.4f}')