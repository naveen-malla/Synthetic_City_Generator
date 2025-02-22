import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class CityCoordinateDataset:
    def __init__(self, coord_folder, min_nodes=10, max_nodes=50):
        self.coord_folder = coord_folder
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
        return coordinates

def create_sequences(coordinates, seq_length=5):
    sequences = []
    targets = []
    for i in range(len(coordinates) - seq_length):
        seq = coordinates[i:i+seq_length]
        target = coordinates[i+seq_length]
        sequences.append(seq.flatten())  # Flatten the sequence
        targets.append(target)
    return np.array(sequences), np.array(targets)

def train_model(X_train, y_train, model_type='linear', alpha=1.0):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train)
    
    return model, scaler

import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(actual, predicted, title="Enhanced Predictions vs Actual"):
    # Set style parameters
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points with enhanced styling
    ax.scatter(actual[:, 1], actual[:, 0],  
               label='Actual', 
               c='#2ECC71',  # Green shade from the first code
               marker='o', 
               s=100)

    ax.scatter(predicted[:, 1], predicted[:, 0], 
               label='Predicted', 
               c='#E74C3C',  # Red shade from the first code
               marker='x', 
               s=100)

    # Plot connecting lines
    for i in range(len(actual)):
        ax.plot([actual[i,1], predicted[i,1]], 
                [actual[i,0], predicted[i,0]], 
                color='royalblue',  # Blue shade from the first code
                linestyle='-',
                linewidth=1,
                alpha=0.3,
                zorder=2)

    # Update labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    
    # Enhance grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Custom legend styling
    legend = ax.legend(frameon=True, 
                       fontsize=10,
                       loc='upper right')
    
    # Adjust tick parameters
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize=10)
    
    plt.tight_layout()
    plt.show()



def evaluate_model(model, X_test, y_test, scaler):
    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': predictions
    }

def main():
    # Set paths to your data directories
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    
    # Model parameters
    seq_length = 5
    model_type = 'elasticnet'
    alpha = 1.0
    
    # Load and prepare training data
    train_dataset = CityCoordinateDataset(train_folder)
    train_sequences = []
    train_targets = []
    
    print(f"Processing {len(train_dataset)} training cities...")
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i]
        sequences, targets = create_sequences(coordinates, seq_length)
        train_sequences.append(sequences)
        train_targets.append(targets)
    
    X_train = np.vstack(train_sequences)
    y_train = np.vstack(train_targets)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    # Train model
    print(f"\nTraining {model_type} regression model...")
    model, scaler = train_model(X_train, y_train, model_type=model_type, alpha=alpha)
    
    # Load and prepare test data
    test_dataset = CityCoordinateDataset(test_folder)
    test_sequences = []
    test_targets = []
    
    print(f"\nProcessing {len(test_dataset)} test cities...")
    for i in range(len(test_dataset)):
        coordinates = test_dataset[i]
        sequences, targets = create_sequences(coordinates, seq_length)
        test_sequences.append(sequences)
        test_targets.append(targets)
    
    X_test = np.vstack(test_sequences)
    y_test = np.vstack(test_targets)
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, X_test, y_test, scaler)
    print(f"Test Set Performance:")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2 Score: {results['r2']:.4f}")

    # Visualize results (using a subset for clarity)
    sample_size = min(1000, len(y_test))
    visualize_predictions(
        y_test[:sample_size], 
        results['predictions'][:sample_size],
        "ElasticNet: First 1000 Predictions vs Actual"
    )
    
   

if __name__ == "__main__":
    main()
