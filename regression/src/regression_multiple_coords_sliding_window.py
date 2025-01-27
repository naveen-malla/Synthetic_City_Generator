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
        city_name = self.file_names[idx].replace('_coords.npy', '')  # Extract city name from file name
        return coordinates, city_name


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
        model = Lasso(alpha=0.001)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    
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

def calculate_errors(original, predicted):
    """
    Calculate errors between original and predicted coordinates
    """
    # Scale coordinates to a more interpretable range (0-100)
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
    
    # Calculate Euclidean distance on scaled coordinates
    euclidean_distances = np.sqrt(np.sum((scaled_original - scaled_predicted) ** 2, axis=1))
    mean_euclidean = np.mean(euclidean_distances)
    
    # Calculate percentage error relative to the range of coordinates
    coord_ranges = np.ptp(original, axis=0)
    relative_errors = np.abs(original - predicted) / coord_ranges * 100
    mean_percentage = np.mean(relative_errors)
    
    return {
        'euclidean': mean_euclidean,
        'percentage': mean_percentage
    }



def plot_model_comparison(original_full, predicted, initial, model_name, city_name, errors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot - Original coordinates
    ax1.scatter(original_full[:, 0], original_full[:, 1], 
               c='blue', label='Original Coordinates', alpha=0.6)
    ax1.scatter(initial[:, 0], initial[:, 1], 
               c='green', label='Initial Coordinates', alpha=0.8, s=100)
    ax1.set_title("Original Coordinates")
    ax1.set_xlabel("Y Coordinate")
    ax1.set_ylabel("X Coordinate")
    ax1.legend()
    ax1.grid(True)
    
    # Right subplot - Predicted coordinates
    ax2.scatter(predicted[:, 0], predicted[:, 1], 
               c='red', label='Predicted Coordinates', alpha=0.6)
    ax2.scatter(initial[:, 0], initial[:, 1], 
               c='green', label='Initial Coordinates', alpha=0.8, s=100)
    ax2.set_title("Predicted Coordinates")
    ax2.set_xlabel("Y Coordinate")
    ax2.set_ylabel("X Coordinate")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"{model_name} for {city_name}\nMean Euclidean Distance: {errors['euclidean']:.6f} | Mean Percentage Error: {errors['percentage']:.2f}%", 
                fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_all_model_comparisons(best_city, best_city_name, seq_length, models_dict):
    initial = best_city[:seq_length]
    original_full = best_city
    
    print(f"\nCoordinate Analysis for {best_city_name}")
    print(f"Initial {seq_length} coordinates used for prediction:")
    print(initial)
    
    for model_name, model in models_dict.items():
        print(f"\n{model_name} Results:")
        sequences, targets = create_sliding_window_sequences(best_city, seq_length)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sequences)
        model.fit(X_scaled, targets)
        
        original, predicted = predict_city_coordinates(best_city, model, scaler, seq_length)
        errors = calculate_errors(original, predicted)
        
        print("Original coordinates:")
        print(original)
        print("\nPredicted coordinates:")
        print(predicted)
        print(f"Number of coordinates: Original = {len(original)}, Predicted = {len(predicted)}")
        print(f"Mean Euclidean Distance: {errors['euclidean']:.6f}")
        print(f"Mean Percentage Error: {errors['percentage']:.2f}%")
        
        plot_model_comparison(original_full, predicted, initial, 
                            model_name, best_city_name, errors)

def main():
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    
    seq_length = 5
    alpha = 1.0
    
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
    
    # Convert to numpy arrays
    X_train = np.array(train_sequences)
    y_train = np.array(train_targets)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    # Get a random city from test set
    test_dataset = CityCoordinateDataset(test_folder)
    random_idx = np.random.randint(len(test_dataset))
    test_city, city_name = test_dataset[random_idx]
    
    print(f"\nSelected random city for evaluation: {city_name}")
    
    # Define models
    models_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=alpha),
        'Lasso': Lasso(alpha=alpha),
        'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=0.5)
    }
    
    # Train all models on the full training data
    for model_name, model in models_dict.items():
        print(f"\nTraining {model_name}...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model.fit(X_scaled, y_train)
    
    # Plot comparisons for all models using the random city
    plot_all_model_comparisons(test_city, city_name, seq_length, models_dict)


if __name__ == "__main__":
    main()
