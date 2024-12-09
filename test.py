import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges, to_undirected
from torch_geometric.datasets import Planetoid

# Set random seed for reproducibility
torch.manual_seed(42)

# Load a sample dataset (Cora)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Prepare the data
data.train_mask = data.val_mask = data.test_mask = data.y = None
data.edge_index = to_undirected(data.edge_index)
data = train_test_split_edges(data)

# Define the VGAE model
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Initialize the model
model = VGAE(VariationalGCNEncoder(dataset.num_features, 16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

# Train the model
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Generate new adjacency matrix
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    pred_adj = model.decoder.forward_all(z)

# Print original and predicted adjacency matrices
print("\nOriginal Adjacency Matrix (subset):")
print(to_undirected(data.edge_index)[:, :10])  # Show first 10 edges

print("\nPredicted Adjacency Matrix (subset):")
pred_edges = (pred_adj > 0.5).nonzero(as_tuple=False).t()
print(pred_edges[:, :10])  # Show first 10 predicted edges

# Print node feature matrix (subset)
print("\nNode Feature Matrix (first 5 nodes, first 5 features):")
print(data.x[:5, :5])
