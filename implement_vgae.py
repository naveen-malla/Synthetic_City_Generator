import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes=1000):
        super().__init__()
        position = torch.arange(max_nodes).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_nodes, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VGAE, self).__init__()
        
        self.pos_encoder = PositionalEncoding(input_dim)
        
        # Deeper encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.encoder_mean = nn.Linear(hidden_dim2, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim2, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x, adj):
        x = self.pos_encoder(x)
        hidden1 = self.encoder1(torch.mm(adj, x))
        hidden2 = self.encoder2(torch.mm(adj, hidden1))
        return self.encoder_mean(hidden2), self.encoder_logvar(hidden2)
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    def decode(self, z):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred
    
    def forward(self, x, adj):
        mean, logvar = self.encode(x, adj)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def main():
    # Increased dimensions
    num_nodes = 4
    input_dim = 32    # Increased input dimension
    hidden_dim1 = 64  # Increased hidden dimensions
    hidden_dim2 = 32
    latent_dim = 16   # Increased latent dimensions
    
    # Create more informative feature matrix X
    X = torch.randn(num_nodes, input_dim)
    
    A = torch.tensor([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=torch.float32)
    
    model = VGAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5
    )
    
    num_epochs = 2000
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    # Class weights for weighted BCE
    pos_weight = torch.ones([1]) * 3.0  # Adjust based on sparsity
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        A_pred, mean, logvar = model(X, A)
        
        # Weighted BCE loss
        recon_loss = F.binary_cross_entropy_with_logits(
            A_pred, A, pos_weight=pos_weight, reduction='mean'
        )
        
        # KL divergence loss with annealing
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        beta = min(1.0, epoch / 200)  # KL annealing
        loss = recon_loss + beta * kl_loss * 0.001
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                A_pred, _, _ = model(X, A)
                pred_labels = (A_pred > 0.5).float()
                accuracy = (pred_labels == A).float().mean().item()
                print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        A_pred, _, _ = model(X, A)
        print("\nOriginal adjacency matrix:")
        print(A)
        print("\nPredicted adjacency matrix:")
        print(A_pred.round(decimals=3))
        
        pred_labels = (A_pred > 0.5).float()
        accuracy = (pred_labels == A).float().mean().item()
        print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
