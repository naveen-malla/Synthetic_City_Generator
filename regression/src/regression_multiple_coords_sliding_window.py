import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class CityCoordinateDataset:
    def __init__(self, coord_folder, min_nodes=20, max_nodes=50):
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

def create_sliding_window_sequences(coordinates, seq_length=5):
    sequences = []
    targets = []
    for i in range(len(coordinates) - seq_length):
        seq = coordinates[i:i+seq_length]
        target = coordinates[i+seq_length]
        sequences.append(seq.flatten())
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

def predict_city_coordinates(coordinates, model, scaler, seq_length=5):
    """
    Predicts all remaining coordinates for a city given the first seq_length coordinates.
    """
    original_coords = coordinates[seq_length:]
    current_sequence = coordinates[:seq_length].copy()
    predictions = []
    
    while len(predictions) < len(original_coords):
        input_sequence = current_sequence.flatten().reshape(1, -1)
        scaled_input = scaler.transform(input_sequence)
        next_coord = model.predict(scaled_input)[0]
        predictions.append(next_coord)
        
        # Slide window
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_coord
    
    return np.array(original_coords), np.array(predictions)

def plot_best_city(original, predicted, initial, title="Best City Prediction Comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original coordinates
    axes[0].scatter(original[:, 0], original[:, 1], c='blue', label='Original Remaining')
    axes[0].set_title("Original Coordinates")
    axes[0].set_xlabel("Y Coordinate")
    axes[0].set_ylabel("X Coordinate")
    axes[0].legend()
    axes[0].grid(True)

    # Plot predicted coordinates
    axes[1].scatter(initial[:, 0], initial[:, 1], c='green', label='Initial Coordinates')
    axes[1].scatter(predicted[:, 0], predicted[:, 1], c='red', label='Predicted Remaining')
    axes[1].set_title("Predicted Coordinates")
    axes[1].set_xlabel("Y Coordinate")
    axes[1].set_ylabel("X Coordinate")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    
    seq_length = 5
    model_type = 'linear'
    alpha = 1.0
    
    # Load and prepare training data
    train_dataset = CityCoordinateDataset(train_folder)
    train_sequences = []
    train_targets = []
    
    print(f"Processing {len(train_dataset)} training cities...")
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i]
        sequences, targets = create_sliding_window_sequences(coordinates, seq_length)
        train_sequences.append(sequences)
        train_targets.append(targets)
    
    X_train = np.vstack(train_sequences)
    y_train = np.vstack(train_targets)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    # Train model
    print(f"\nTraining {model_type} regression model...")
    model, scaler = train_model(X_train, y_train, model_type=model_type, alpha=alpha)
    
    # Test predictions on all cities
    test_dataset = CityCoordinateDataset(test_folder)
    print(f"\nTesting predictions on {len(test_dataset)} cities...")
    
    best_rmse = float('inf')
    best_city = None
    best_original = None
    best_predicted = None
    best_initial = None
    
    for i in range(len(test_dataset)):
        test_city = test_dataset[i]
        original, predicted = predict_city_coordinates(test_city, model, scaler, seq_length)
        
        rmse = np.sqrt(mean_squared_error(original, predicted))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_city = test_city
            best_original = original
            best_predicted = predicted
            best_initial = test_city[:seq_length]
    
    print(f"\nBest city RMSE: {best_rmse:.4f}")
    plot_best_city(best_original, best_predicted, best_initial, "Best City Prediction Comparison")

if __name__ == "__main__":
    main()
