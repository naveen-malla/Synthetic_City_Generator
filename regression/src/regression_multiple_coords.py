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

def create_multi_step_sequences(coordinates, input_length=5, predict_length=3):
    sequences = []
    targets = []
    
    for i in range(len(coordinates) - input_length - predict_length + 1):
        input_seq = coordinates[i:i+input_length]
        target_seq = coordinates[i+input_length:i+input_length+predict_length]
        sequences.append(input_seq.flatten())
        targets.append(target_seq.flatten())
    
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

def evaluate_model(model, X_test, y_test, scaler, predict_length):
    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled)
    
    # Reshape predictions and targets for proper evaluation
    predictions_reshaped = predictions.reshape(-1, predict_length, 2)
    y_test_reshaped = y_test.reshape(-1, predict_length, 2)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': predictions_reshaped,
        'actual': y_test_reshaped
    }

def visualize_predictions(actual, predicted, title="Predictions vs Actual"):
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    for step in range(actual.shape[1]):
        plt.scatter(actual[:, step, 0], actual[:, step, 1], 
                   label=f'Actual Step {step+1}', 
                   alpha=0.6, 
                   c=colors[step], 
                   marker='o')
        plt.scatter(predicted[:, step, 0], predicted[:, step, 1], 
                   label=f'Predicted Step {step+1}', 
                   alpha=0.6, 
                   c=colors[step], 
                   marker='x')
    
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    base_path = 'data/coordinates/world/center/original'
    train_folder = os.path.join(base_path, 'train')
    test_folder = os.path.join(base_path, 'test')
    
    input_length = 5
    predict_length = 3
    model_types = ['linear', 'ridge', 'lasso', 'elasticnet']
    alpha = 1.0
    
    train_dataset = CityCoordinateDataset(train_folder)
    train_sequences = []
    train_targets = []
    
    print(f"Processing {len(train_dataset)} training cities...")
    for i in range(len(train_dataset)):
        coordinates = train_dataset[i]
        sequences, targets = create_multi_step_sequences(coordinates, input_length, predict_length)
        if len(sequences) > 0:
            train_sequences.append(sequences)
            train_targets.append(targets)
    
    X_train = np.vstack(train_sequences)
    y_train = np.vstack(train_targets)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    test_dataset = CityCoordinateDataset(test_folder)
    test_sequences = []
    test_targets = []
    
    print(f"\nProcessing {len(test_dataset)} test cities...")
    for i in range(len(test_dataset)):
        coordinates = test_dataset[i]
        sequences, targets = create_multi_step_sequences(coordinates, input_length, predict_length)
        if len(sequences) > 0:
            test_sequences.append(sequences)
            test_targets.append(targets)
    
    X_test = np.vstack(test_sequences)
    y_test = np.vstack(test_targets)
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    for model_type in model_types:
        print(f"\nTraining {model_type} regression model...")
        model, scaler = train_model(X_train, y_train, model_type=model_type, alpha=alpha)
        
        print(f"Evaluating {model_type} model...")
        results = evaluate_model(model, X_test, y_test, scaler, predict_length)
        print(f"{model_type.capitalize()} Model Performance:")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R2 Score: {results['r2']:.4f}")
        
        sample_size = min(1000, len(results['actual']))
        visualize_predictions(
            results['actual'][:sample_size], 
            results['predictions'][:sample_size],
            f"{model_type.capitalize()} Model: Multi-step Predictions"
        )

if __name__ == "__main__":
    main()
