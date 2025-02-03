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
        'Lasso': (Lasso(alpha=0.1), StandardScaler()),
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


def plot_model_comparison(original, predicted, initial, model_name, city_name, errors):
    # Convert inputs to numpy arrays if they aren't already
    original = np.array(original)
    predicted = np.array(predicted)
    initial = np.array(initial)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Define color scheme
    colors = {
        'original': '#2ECC71',  # Green
        'predicted': '#E74C3C',  # Red
        'initial': 'royalblue'  # Blue
    }
    
    # Plot original coordinates
    ax1.scatter(original[:, 0], original[:, 1], 
                c=colors['original'], marker='o', s=100,
                label='Original Coordinates')
    ax1.scatter(initial[:, 0], initial[:, 1], 
                c=colors['initial'], marker='o', s=120, 
                label='Initial Coordinates')
    ax1.set_title('Original Coordinates', fontsize=14, pad=15)
    ax1.set_xlabel('Latitude', fontsize=12)
    ax1.set_ylabel('Longitude', fontsize=12)
    ax1.legend()
    
    # Plot predicted coordinates
    ax2.scatter(predicted[:, 0], predicted[:, 1], 
                c=colors['predicted'], marker='o', s=100, 
                label='Predicted Coordinates')
    ax2.scatter(initial[:, 0], initial[:, 1], 
                c=colors['initial'], marker='o', s=120, 
                label='Initial Coordinates')
    ax2.set_title('Predicted Coordinates', fontsize=14, pad=15)
    ax2.set_xlabel('Latitude', fontsize=12)
    ax2.set_ylabel('Longitude', fontsize=12)
    ax2.legend()

    # Main title with metrics
    title = f"{model_name} Predictions for {city_name}\n"
    metrics = f"Euclidean Distance: {errors['euclidean']:.2f} | Percentage Error: {errors['percentage']:.2f}%"
    
    # Reduced spacing adjustments
    plt.subplots_adjust(top=0.9)  # Changed from 0.85
    plt.suptitle(title + metrics, fontsize=14, y=0.95)  # Changed from 0.98
    
    # Tighter layout with less space at top
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Changed from 0.90
    plt.show()

def evaluate_model_performance(test_dataset, trained_models, seq_length=5):
    """
    Evaluate models on all test cities and generate performance statistics
    """
    # Initialize statistics dictionary for each model
    stats = {model_name: {
        'mean_euclidean': [],
        'mean_percentage': [],
        'num_predictions': [],
        'convergence_rate': 0,  # Percentage of successful predictions
        'min_error': float('inf'),
        'max_error': 0,
        'std_error': 0
    } for model_name in trained_models.keys()}
    
    # Evaluate each model on all test cities
    total_cities = len(test_dataset)
    for i in range(total_cities):
        test_city, city_name = test_dataset[i]
        
        for model_name, (model, scaler) in trained_models.items():
            original, predicted = predict_city_coordinates(test_city, model, scaler, seq_length)
            
            if len(original) > 0 and len(predicted) > 0:
                errors = calculate_errors(original, predicted)
                stats[model_name]['mean_euclidean'].append(errors['euclidean'])
                stats[model_name]['mean_percentage'].append(errors['percentage'])
                stats[model_name]['num_predictions'].append(len(predicted))
                
                # Update min/max errors
                stats[model_name]['min_error'] = min(stats[model_name]['min_error'], 
                                                   errors['euclidean'])
                stats[model_name]['max_error'] = max(stats[model_name]['max_error'], 
                                                   errors['euclidean'])
    
    # Calculate final statistics
    for model_name in trained_models.keys():
        model_stats = stats[model_name]
        num_successful = len(model_stats['mean_euclidean'])
        
        if num_successful > 0:
            model_stats['convergence_rate'] = (num_successful / total_cities) * 100
            model_stats['avg_euclidean'] = np.mean(model_stats['mean_euclidean'])
            model_stats['avg_percentage'] = np.mean(model_stats['mean_percentage'])
            model_stats['std_error'] = np.std(model_stats['mean_euclidean'])
            model_stats['avg_predictions'] = np.mean(model_stats['num_predictions'])
        
        # Clean up temporary lists
        del model_stats['mean_euclidean']
        del model_stats['mean_percentage']
        del model_stats['num_predictions']
    
    # Create performance comparison table
    headers = ['Model', 'Avg Euclidean', 'Avg % Error', 'Min Error', 'Max Error', 
              'Std Error', 'Avg # Predictions', 'Convergence %']
    
    print("\nModel Performance Comparison:")
    print("-" * 100)
    print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} "
          f"{headers[4]:<15} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15}")
    print("-" * 100)
    
    for model_name, model_stats in stats.items():
        print(f"{model_name:<15} "
              f"{model_stats['avg_euclidean']:<15.2f} "
              f"{model_stats['avg_percentage']:<15.2f} "
              f"{model_stats['min_error']:<15.2f} "
              f"{model_stats['max_error']:<15.2f} "
              f"{model_stats['std_error']:<15.2f} "
              f"{model_stats['avg_predictions']:<15.2f} "
              f"{model_stats['convergence_rate']:<15.2f}")
    
    return stats


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
    
    # Load test dataset
    test_dataset = CityCoordinateDataset(test_folder)
    print(f"\nProcessing {len(test_dataset)} test cities...")


    # Generate performance statistics
    print("\nGenerating model performance statistics...")
    performance_stats = evaluate_model_performance(test_dataset, trained_models, seq_length)
    
    # Select random test city for visualization
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
        
        # Get initial coordinates (first seq_length coordinates from original)
        initial_coords = original[:seq_length]
        
        # Call the plot function with original, predicted, and initial coordinates
        plot_model_comparison(original, predicted, initial_coords, name, city_name, errors)



if __name__ == "__main__":
    main()
