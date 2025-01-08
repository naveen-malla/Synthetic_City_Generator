import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

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
            torch.nn.Linear(latent_dim + 2, latent_dim),
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
        adj_dir = os.path.join(data_dir, 'adj_matrices/world/center')
        coord_dir = os.path.join(data_dir, 'coordinates/world/center/transformed')

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
    
    except Exception as e:
        print(f"Fatal error in data loading: {str(e)}")
        raise

def split_dataset(matrices, features_list, processed_cities, train_ratio=0.7, val_ratio=0.15):
    n = len(matrices)
    indices = np.random.permutation(n)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    def subset(data, idx):
        return [data[i] for i in idx]
    
    return (subset(matrices, train_indices), subset(features_list, train_indices), subset(processed_cities, train_indices)), \
           (subset(matrices, val_indices), subset(features_list, val_indices), subset(processed_cities, val_indices)), \
           (subset(matrices, test_indices), subset(features_list, test_indices), subset(processed_cities, test_indices))

def train_epoch(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    data = data.to(model.device)
    edge_index = edge_index.to(model.device)
    z = model.encode(data.x, edge_index)
    adj_orig = to_dense_adj(edge_index)[0]
    decoded = model.decoder(z, data.x)
    adj_pred = torch.sigmoid(torch.matmul(decoded, decoded.t()))
    num_edges = edge_index.shape[1]
    num_nodes = data.x.size(0)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    pos_weight = torch.tensor((num_possible_edges - num_edges) / num_edges).to(model.device)
    loss = F.binary_cross_entropy_with_logits(adj_pred.view(-1), adj_orig.view(-1), pos_weight=pos_weight)
    kl_loss = (0.05 / data.x.size(0)) * model.kl_loss()
    total_loss = loss + kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss

def evaluate(model, data, adj_matrix):
    model.eval()
    data = data.to(model.device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
    auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
    ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
    return auc, ap

def final_evaluation(model, matrices, features_list, processed_cities):
    print("\nFinal Evaluation:")
    results = []
    
    for idx, ((edge_index, adj_matrix), features) in enumerate(zip(matrices, features_list)):
        with torch.no_grad():
            data = Data(x=features, edge_index=edge_index).to(model.device)
            z = model.encode(data.x, data.edge_index)
            pred_adj = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
        auc = roc_auc_score(adj_matrix.flatten(), pred_adj.flatten())
        ap = average_precision_score(adj_matrix.flatten(), pred_adj.flatten())
        results.append({
            'City': processed_cities[idx],
            'Nodes': adj_matrix.shape[0],
            'AUC': auc,
            'AP': ap
        })
    
    df = pd.DataFrame(results)
    
    print("\nSummary Statistics:")
    print(df[['AUC', 'AP']].describe())
    
    print("\nTop 5 Cities by AUC:")
    print(tabulate(df.nlargest(5, 'AUC')[['City', 'Nodes', 'AUC', 'AP']], headers='keys', tablefmt='pretty'))
    
    print("\nBottom 5 Cities by AUC:")
    print(tabulate(df.nsmallest(5, 'AUC')[['City', 'Nodes', 'AUC', 'AP']], headers='keys', tablefmt='pretty'))
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(df['AUC'], bins=20, edgecolor='black')
    ax1.set_title('Distribution of AUC Scores')
    ax1.set_xlabel('AUC')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(df['AP'], bins=20, edgecolor='black')
    ax2.set_title('Distribution of AP Scores')
    ax2.set_xlabel('AP')
    ax2.set_ylabel('Frequency')
   
    plt.tight_layout()
    plt.savefig('score_distributions.png')
    plt.close()
    
    df['Size'] = pd.cut(df['Nodes'], bins=[0, 60, 80, 100], labels=['Small', 'Medium', 'Large'])
    size_performance = df.groupby('Size')[['AUC', 'AP']].mean()
    
    print("\nPerformance by City Size:")
    print(tabulate(size_performance, headers='keys', tablefmt='pretty'))
    
    return df

def main():
    min_nodes, max_nodes = 10, 100
    matrices, features_list, processed_cities = load_and_process_matrices('data', min_nodes, max_nodes)
    
    train_data, val_data, test_data = split_dataset(matrices, features_list, processed_cities)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VGAE(
        encoder=ImprovedVGAEEncoder(in_channels=2, hidden_channels=512, out_channels=256, dropout=0.3),
        decoder=ImprovedVGAEDecoder(latent_dim=256)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)
   
    best_val_auc = 0
    patience, counter = 50, 0
    
    for epoch in range(1, 1001):
        total_loss = 0
        for (edge_index, adj_matrix), features in zip(*train_data[:2]):
            data = Data(x=features, edge_index=edge_index).to(device)
            loss = train_epoch(model, optimizer, data, edge_index)
            total_loss += loss.item()
    
        val_aucs = []
        val_aps = []
        for (val_edge_index, val_adj), val_features in zip(val_data[0], val_data[1]):
            val_data_obj = Data(x=val_features, edge_index=val_edge_index).to(device)
            val_auc, val_ap = evaluate(model, val_data_obj, val_adj)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)
     
        avg_val_auc = np.mean(val_aucs)
        avg_val_ap = np.mean(val_aps)
        scheduler.step(avg_val_auc)
     
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
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
            print(f'Epoch: {epoch:03d}, Loss: {total_loss/len(train_data[0]):.4f}, '
                  f'Val AUC: {avg_val_auc:.4f}, Val AP: {avg_val_ap:.4f}')
      
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    checkpoint = torch.load('best_vgae_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    results_df = final_evaluation(model, test_data[0], test_data[1], test_data[2])

if __name__ == "__main__":
    main()
