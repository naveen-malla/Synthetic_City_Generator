import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class CityCoordinateDataset(Dataset):
    def __init__(self, coord_folder, adj_folder, min_nodes=10, max_nodes=50):
        self.coord_folder = coord_folder
        self.adj_folder = adj_folder
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.file_names = self._get_valid_files()
        
    def _get_valid_files(self):
        coord_files = set(f.replace('_coords.npy', '') 
                         for f in os.listdir(self.coord_folder) 
                         if f.endswith('_coords.npy'))
        adj_files = set(f.replace('_adj.npy', '') 
                       for f in os.listdir(self.adj_folder) 
                       if f.endswith('_adj.npy'))
        return list(coord_files.intersection(adj_files))
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        coord_file = os.path.join(self.coord_folder, f"{file_name}_coords.npy")
        
        structured_coords = np.load(coord_file)
        coordinates = np.column_stack((structured_coords['y'], 
                                     structured_coords['x'])).astype(np.float32)
        
        total_pairs = coordinates.shape[0]
        split_idx = int(total_pairs * 0.2)
        
        input_seq = torch.FloatTensor(coordinates[:split_idx])
        target_seq = torch.FloatTensor(coordinates[split_idx:])
        
        return input_seq, target_seq

class CityCoordinateLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.fc(lstm_out)
        return predictions, hidden

def evaluate_model(model, data_loader, criterion, device, dataset_name=""):
    model.eval()
    total_loss = 0
    batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in tqdm(data_loader, desc=f'Evaluating on {dataset_name}'):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output, _ = model(input_seq)
            loss = criterion(output, target_seq)
            total_loss += loss.item()
            batches += 1
    
    avg_loss = total_loss / batches
    print(f'{dataset_name} Loss: {avg_loss:.6f}')
    return avg_loss

def train_model(model, train_loader, valid_loader, num_epochs, device, save_dir):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for input_seq, target_seq in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            output, _ = model(input_seq)
            loss = criterion(output, target_seq)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        valid_loss = evaluate_model(model, valid_loader, criterion, device, "Validation")
        valid_losses.append(valid_loss)
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {valid_loss:.6f}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    return train_losses, valid_losses

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Base paths
    BASE_PATH = 'data/coordinates/world/center/transformed'
    ADJ_BASE_PATH = 'data/adj_matrices/world/center'
    SAVE_DIR = 'lstm/results'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Create datasets for train, valid, and test
    train_dataset = CityCoordinateDataset(
        os.path.join(BASE_PATH, 'train'),
        os.path.join(ADJ_BASE_PATH, 'train')
    )
    valid_dataset = CityCoordinateDataset(
        os.path.join(BASE_PATH, 'valid'),
        os.path.join(ADJ_BASE_PATH, 'valid')
    )
    test_dataset = CityCoordinateDataset(
        os.path.join(BASE_PATH, 'test'),
        os.path.join(ADJ_BASE_PATH, 'test')
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CityCoordinateLSTM().to(device)
    
    # Train model
    train_losses, valid_losses = train_model(
        model, 
        train_loader, 
        valid_loader, 
        num_epochs=50,
        device=device,
        save_dir=SAVE_DIR
    )
    
    # Load best model and evaluate on test set
    best_model = CityCoordinateLSTM().to(device)
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_lstm_model.pth'))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.MSELoss()
    test_loss = evaluate_model(best_model, test_loader, criterion, device, "Test")
    
    # Save all metrics
    np.save(os.path.join(SAVE_DIR, 'train_losses.npy'), train_losses)
    np.save(os.path.join(SAVE_DIR, 'valid_losses.npy'), valid_losses)
    np.save(os.path.join(SAVE_DIR, 'test_loss.npy'), test_loss)
    
    print(f"Final Test Loss: {test_loss:.6f}")
    print("Training completed. Model and metrics saved.")

if __name__ == "__main__":
    main()
