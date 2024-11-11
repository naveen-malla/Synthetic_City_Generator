import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Mean and variance
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Modified decoder
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final layer to output edge predictions
        self.edge_predictor = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x, adj):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        # Process latent vectors
        h = self.decoder_hidden(z)
        # Get edge predictions
        decoded = self.edge_predictor(h)
        # Create adjacency matrix through inner product
        adj_pred = torch.sigmoid(torch.mm(decoded, decoded.t()))
        # Ensure symmetry
        adj_pred = (adj_pred + adj_pred.t()) / 2
        return adj_pred
    
    def forward(self, x, adj):
        """
        Forward pass through the entire model
        """
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed, mu, logvar

# Create a simple example
def create_example_graph():
    # Create a simple graph with 4 nodes
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])
    
    # Create adjacency matrix
    adj = nx.adjacency_matrix(G).todense()
    adj = torch.FloatTensor(adj)
    
    # Create random node features (2-dimensional for this example)
    features = torch.randn(4, 2)
    
    return G, adj, features

def print_adjacency_comparison(original_adj, reconstructed_adj):
    """Helper function to print and compare adjacency matrices"""
    print("\nOriginal Adjacency Matrix:")
    print(original_adj.detach().numpy().round(3))
    
    print("\nReconstructed Adjacency Matrix (raw probabilities):")
    print(reconstructed_adj.detach().numpy().round(3))
    
    print("\nReconstructed Adjacency Matrix (thresholded):")
    print((reconstructed_adj.detach() > 0.5).float().numpy())
    
    # Print difference
    diff = torch.abs(original_adj - reconstructed_adj.detach())
    print("\nAbsolute Difference:")
    print(diff.numpy().round(3))

def margin_loss(pred, target, margin=0.3):
    """Custom margin loss to encourage more decisive predictions"""
    # Push 1s to be above 0.5 + margin
    pos_loss = torch.clamp(0.5 + margin - pred, min=0) * target
    # Push 0s to be below 0.5 - margin
    neg_loss = torch.clamp(pred - (0.5 - margin), min=0) * (1 - target)
    return pos_loss + neg_loss

def main():
    # Create example data
    G, adj, features = create_example_graph()
    
    model = VGAE(input_dim=2, hidden_dim=64, latent_dim=16)  # Increased dimensions
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Increased learning rate
    
    for epoch in range(300):  # Reduced epochs, focus on quality
        model.train()
        optimizer.zero_grad()
        
        reconstructed_adj, mu, logvar = model(features, adj)
        
        # Custom losses
        recon_loss = margin_loss(reconstructed_adj, adj).mean() * 15.0
        kl_loss = -0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Remove L1 loss as margin loss handles sparsity
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                pred_adj = (reconstructed_adj > 0.5).float()
                accuracy = (pred_adj == adj).float().mean().item()
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
                print(f'Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}')
                print(f'Accuracy: {accuracy*100:.2f}%')
                
                if accuracy == 1.0:  # Early stop if perfect accuracy
                    print("Perfect accuracy achieved!")
                    break
    
    # Final comparison after training
    print("\n=== Final Results ===")
    print_adjacency_comparison(adj, reconstructed_adj)
    
    # Visualize results
    model.eval()
    with torch.no_grad():
        reconstructed_adj, _, _ = model(features, adj)
        
    # Plot original vs reconstructed graphs
    plt.figure(figsize=(10, 4))
    
    plt.subplot(121)
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.title("Original Graph")
    
    plt.subplot(122)
    reconstructed_G = nx.from_numpy_array((reconstructed_adj > 0.5).numpy())
    nx.draw(reconstructed_G, with_labels=True, node_color='lightgreen')
    plt.title("Reconstructed Graph")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
