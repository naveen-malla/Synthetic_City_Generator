import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class CityCoordinateDataset:
    def __init__(self, coord_folder, min_nodes=20, max_nodes=30):
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
        city_name = self.file_names[idx].replace('_coords.npy', '')
        return coordinates, city_name

def create_sliding_window_sequences(coordinates, seq_length=5):
    if len(coordinates) <= seq_length:
        return np.array([]), np.array([])
    
    sequences = []
    targets = []
    for i in range(len(coordinates) - seq_length):
        seq = coordinates[i:i+seq_length]
        target = coordinates[i+seq_length]
        sequences.append(seq.flatten())
        targets.append(target)
    return np.array(sequences), np.array(targets)

def train_models(X_train, y_train):
    models = {
        'Linear Regression': (LinearRegression(), StandardScaler()),
        'Ridge': (Ridge(alpha=0.1), StandardScaler()),
        'Lasso': (Lasso(alpha=0.001), StandardScaler()),
        'ElasticNet': (ElasticNet(alpha=0.01, l1_ratio=0.5), StandardScaler())
    }
    
    trained_models = {}
    for name, (model, scaler) in models.items():
        X_scaled = scaler.fit_transform(X_train)
        model.fit(X_scaled, y_train)
        trained_models[name] = (model, scaler)
    
    return trained_models

def predict_city_coordinates(coordinates, model, scaler, seq_length=5):
    if len(coordinates) <= seq_length:
        return np.array([]), np.array([])
    
    original_coords = coordinates[seq_length:]
    current_sequence = coordinates[:seq_length].copy()
    predictions = []
    
    while len(predictions) < len(original_coords):
        input_sequence = current_sequence.flatten().reshape(1, -1)
        scaled_input = scaler.transform(input_sequence)
        next_coord = model.predict(scaled_input)[0]
        predictions.append(next_coord)
        
        current_sequence = np.vstack((current_sequence[1:], next_coord))
    
    return np.array(original_coords), np.array(predictions)

def calculate_errors(original, predicted):
    if len(original) == 0 or len(predicted) == 0:
        return {'euclidean': float('inf'), 'percentage': float('inf')}
    
    # Calculate coordinate-wise percentage errors
    coord_ranges = np.ptp(original, axis=0)
    relative_errors = np.abs(original - predicted) / coord_ranges * 100
    
    # Calculate scaled Euclidean distances
    lat_range = np.ptp(original[:, 0])
    lon_range = np.ptp(original[:, 1])
    
    scaled_original = np.column_stack([
        (original[:, 0] - np.min(original[:, 0])) / lat_range * 100,
        (original[:, 1] - np.min(original[:, 1])) / lon_range * 100
    ])
    
    scaled_predicted = np.column_stack([
        (predicted[:, 0] - np.min(original[:, 0])) / lat_range * 100,
        (predicted[:, 1] - np.min(original[:, 1])) / lon_range * 100
    ])
    
    euclidean_distances = np.sqrt(np.sum((scaled_original - scaled_predicted) ** 2, axis=1))
    
    return {
        'euclidean': np.mean(euclidean_distances),
        'percentage': np.mean(relative_errors)
    }

def plot_model_comparison(original_full, predicted, initial, model_name, city_name, errors):
    # Set style parameters
    plt.style.use('bmh')  # Using built-in style
    
    # Create figure with a light gray background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#f0f0f0')
    
    # Common plotting parameters
    plot_params = {
        'original': {'color': '#1f77b4', 'alpha': 0.7, 's': 80, 'label': 'Original Coordinates'},
        'initial': {'color': '#2ecc71', 'alpha': 1.0, 's': 120, 'label': 'Initial Coordinates', 'edgecolor': 'white'},
        'predicted': {'color': '#e74c3c', 'alpha': 0.7, 's': 80, 'label': 'Predicted Coordinates'}
    }
    
    # Left subplot - Original coordinates
    ax1.scatter(original_full[:, 0], original_full[:, 1], **plot_params['original'])
    ax1.scatter(initial[:, 0], initial[:, 1], **plot_params['initial'])
    ax1.set_title("Original Coordinates", fontsize=12, pad=15)
    ax1.set_xlabel("Latitude", fontsize=10)
    ax1.set_ylabel("Longitude", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('white')
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Right subplot - Predicted coordinates
    ax2.scatter(predicted[:, 0], predicted[:, 1], **plot_params['predicted'])
    ax2.scatter(initial[:, 0], initial[:, 1], **plot_params['initial'])
    ax2.set_title("Predicted Coordinates", fontsize=12, pad=15)
    ax2.set_xlabel("Latitude", fontsize=10)
    ax2.set_ylabel("Longitude", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor('white')
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Main title with metrics
    title = f"{model_name} Predictions for {city_name}\n"
    metrics = f"Mean Euclidean Distance: {errors['euclidean']:.2f} | Mean Percentage Error: {errors['percentage']:.2f}%"
    plt.suptitle(title + metrics, fontsize=14, y=1.05)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()



def main():
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    seq_length = 5
    
    # Load and prepare training data
    train_dataset = CityCoordinateDataset(train_folder)
    train_sequences = []
    train_targets = []
    
    print(f"Processing {len(train_dataset)} training cities...")
    for i in range(len(train_dataset)):
        coordinates, _ = train_dataset[i]
        if len(coordinates) >= seq_length + 1:
            sequences, targets = create_sliding_window_sequences(coordinates, seq_length)
            if sequences.size > 0 and targets.size > 0:
                train_sequences.extend(sequences)
                train_targets.extend(targets)
    
    X_train = np.array(train_sequences)
    y_train = np.array(train_targets)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    # Train all models
    trained_models = train_models(X_train, y_train)
    
    # Select random test city
    test_dataset = CityCoordinateDataset(test_folder)
    random_idx = np.random.randint(len(test_dataset))
    test_city, city_name = test_dataset[random_idx]
    
    print(f"\nSelected random city for evaluation: {city_name}")
    print(f"Initial {seq_length} coordinates used for prediction:")
    print(test_city[:seq_length])
    
    # Evaluate all models
    for name, (model, scaler) in trained_models.items():
        print(f"\n{name} Results:")
        original, predicted = predict_city_coordinates(test_city, model, scaler, seq_length)
        
        print("Original coordinates:")
        print(original)
        print("\nPredicted coordinates:")
        print(predicted)
        print(f"Number of coordinates: Original = {len(original)}, Predicted = {len(predicted)}")
        
        errors = calculate_errors(original, predicted)
        plot_model_comparison(test_city, predicted, test_city[:seq_length], 
                            name, city_name, errors)

if __name__ == "__main__":
    main()
