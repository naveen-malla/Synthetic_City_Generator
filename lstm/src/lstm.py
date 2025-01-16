import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # Sort sequences by length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate input and target sequences
    input_seqs, target_seqs = zip(*batch)
    
    # Pad sequences
    input_seqs = pad_sequence(input_seqs, batch_first=True)
    target_seqs = pad_sequence(target_seqs, batch_first=True)
    
    return input_seqs, target_seqs


class CityCoordinateDataset(Dataset):
    def __init__(self, coord_folder, min_nodes=10, max_nodes=50):
        self.coord_folder = coord_folder
        # Filter files during initialization
        self.file_names = []
        for f in os.listdir(coord_folder):
            if f.endswith('_coords.npy'):
                coords = np.load(os.path.join(coord_folder, f))
                coordinates = np.column_stack((coords['y'], coords['x']))
                num_nodes = len(coordinates)
                if min_nodes <= num_nodes <= max_nodes:
                    self.file_names.append(f)
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        coord_file = os.path.join(self.coord_folder, self.file_names[idx])
        coords = np.load(coord_file)
        coordinates = np.column_stack((coords['y'], coords['x'])).astype(np.float32)
        
        # Split into input (20%) and target (80%)
        split_idx = int(len(coordinates) * 0.2)
        input_seq = torch.FloatTensor(coordinates[:split_idx])
        target_seq = torch.FloatTensor(coordinates[split_idx:])
        
        return input_seq, target_seq


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x, target_length=None):
        batch_size = x.size(0)
        if target_length is None:
            target_length = int(x.size(1) / 0.2 * 0.8)
        
        lstm_out, (h, c) = self.lstm(x)
        outputs = []
        current_input = x[:, -1:, :]
        
        for _ in range(target_length):
            out, (h, c) = self.lstm(current_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            current_input = pred
            
        return torch.cat(outputs, dim=1)



def train_model(model, train_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for input_seq, target_seq in tqdm(train_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            # Pass target sequence length
            output = model(input_seq, target_seq.size(1))
            loss = criterion(output, target_seq)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()



def main():
    coord_folder = 'data/coordinates/world/center/transformed/train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = CityCoordinateDataset(coord_folder)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    model = SimpleLSTM().to(device)
    train_model(model, train_loader, num_epochs=10, device=device)

if __name__ == "__main__":
    main()
