import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
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

def create_dataset(coordinates, seq_length=5):
    input_seq = coordinates[:seq_length].flatten()
    target_seq = coordinates[seq_length:]
    return input_seq, target_seq.flatten()


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

def visualize_full_path(original, predicted, title="Original vs Predicted Path"):
    plt.figure(figsize=(12, 8))
    
    # Plot original path
    plt.plot(original[:, 0], original[:, 1], 'b-', label='Original Path', linewidth=2)
    plt.scatter(original[:5, 0], original[:5, 1], c='green', marker='o', 
               s=100, label='First 5 Points', zorder=5)
    plt.scatter(original[5:, 0], original[5:, 1], c='blue', marker='o', 
               s=80, label='Original Remaining', zorder=4)
    
    # Plot predicted path
    plt.plot(predicted[:, 0], predicted[:, 1], 'r--', label='Predicted Path', linewidth=2)
    plt.scatter(predicted[5:, 0], predicted[5:, 1], c='red', marker='x', 
               s=80, label='Predicted Points', zorder=3)
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_predictions(test_city, models, scalers):
    """
    Compare predictions from all models for a single test city
    """
    input_seq, target_seq = create_dataset(test_city)
    num_predictions = len(test_city) - 5  # Dynamic number of predictions needed
    
    print("\nOriginal Coordinates:")
    print("First 5 points (Input):")
    print(test_city[:5])
    print("\nRemaining points (Target):")
    print(test_city[5:])
    
    for model_type, model in models.items():
        print(f"\n{model_type.upper()} Model Predictions:")
        input_scaled = scalers[model_type].transform([input_seq])
        predicted_seq = model.predict(input_scaled).reshape(-1, 2)[:num_predictions]  # Limit predictions
        full_predicted = np.vstack((test_city[:5], predicted_seq))
        
        print("Predicted remaining points:")
        print(predicted_seq)
        
        visualize_full_path(test_city, full_predicted, 
                          f"{model_type.capitalize()} Model: Original vs Predicted Path")


def main():
    # Set paths to your data directories
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    
    # Model parameters
    seq_length = 5
    alpha = 1.0
    
    # Load and prepare training data
    train_dataset = CityCoordinateDataset(train_folder)
    train_inputs = []
    train_targets = []
    
    print(f"Processing {len(train_dataset)} training cities...")
    
    # Create training dataset
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i]
        input_seq, target_seq = create_dataset(coordinates, seq_length)
        train_inputs.append(input_seq)
        # Pad each target sequence to the same length as the longest sequence
        train_targets.append(target_seq)
    
    # Convert inputs to numpy array (this is safe as all inputs have same length)
    X_train = np.array(train_inputs)
    
    # Find the maximum length among target sequences
    max_length = max(len(target) for target in train_targets)
    
    # Pad all target sequences to max_length
    y_train = np.array([
        np.pad(target, (0, max_length - len(target)), 'constant')
        for target in train_targets
    ])
    
    # Train all models
    models = {}
    scalers = {}
    for model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
        print(f"\nTraining {model_type} regression model...")
        model, scaler = train_model(X_train, y_train, model_type=model_type, alpha=alpha)
        models[model_type] = model
        scalers[model_type] = scaler
    
    # Load test data
    test_dataset = CityCoordinateDataset(test_folder)
    
    # Compare predictions for first test city
    if len(test_dataset) > 0:
        test_city = test_dataset[0]
        compare_predictions(test_city, models, scalers)



if __name__ == "__main__":
    main()
