import os
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict
import json
from tabulate import tabulate

# Constants
NODE_RANGES = [(10, 50), (50, 100), (100, 500), (500, 1000), (1000, 2000)]
SPLIT_RATIOS = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
BASE_PATHS = {
    'adj': 'adj_matrices',
    'coord': 'coordinates'
}

def validate_city(data_dir, city, region, location):
    """Validate if city exists in all required locations with matching sizes"""
    
    # Construct paths
    adj_path = Path(data_dir) / BASE_PATHS['adj'] / region / location / f"{city}_adj.npy"
    coord_orig = Path(data_dir) / BASE_PATHS['coord'] / region / location / 'original' / f"{city}_coords.npy"
    coord_trans = Path(data_dir) / BASE_PATHS['coord'] / region / location / 'transformed' / f"{city}_coords.npy"
    
    # Check existence
    if not all(p.exists() for p in [adj_path, coord_orig, coord_trans]):
        return False, None
    
    # Load and check sizes
    try:
        adj_size = np.load(adj_path).shape[0]
        orig_size = len(np.load(coord_orig))
        trans_size = len(np.load(coord_trans))
        
        if adj_size == orig_size == trans_size:
            return True, adj_size
    except:
        return False, None
        
    return False, None

def find_cities_in_range(data_dir, min_nodes, max_nodes):
    """Find cities within node range with valid files across all locations"""
    valid_cities = defaultdict(dict)
    
    for region in ['germany', 'world']:
        for location in ['center', 'whole']:
            adj_dir = Path(data_dir) / BASE_PATHS['adj'] / region / location
            
            if not adj_dir.exists():
                continue
                
            cities = {}
            for f in adj_dir.glob("*_adj.npy"):
                city = f.stem.replace('_adj', '')
                is_valid, size = validate_city(data_dir, city, region, location)
                
                if is_valid and min_nodes <= size <= max_nodes:
                    cities[city] = size
                    
            valid_cities[(region, location)] = cities
            
    return valid_cities

def create_splits(cities):
    """Create consistent train/valid/test splits"""
    n_total = len(cities)
    n_train = int(n_total * SPLIT_RATIOS['train'])
    n_valid = int(n_total * SPLIT_RATIOS['valid'])
    
    cities_list = list(cities.keys())
    np.random.shuffle(cities_list)
    
    splits = {
        'train': cities_list[:n_train],
        'valid': cities_list[n_train:n_train+n_valid],
        'test': cities_list[n_train+n_valid:],
        'rest': []
    }
    
    return splits

def reorganize_files(data_dir, valid_cities, splits):
    """Reorganize files into train/valid/test folders"""
    for (region, location), cities in valid_cities.items():
        # Create splits for adjacency matrices
        adj_base = Path(data_dir) / BASE_PATHS['adj'] / region / location
        for split_name, split_cities in splits.items():
            split_dir = adj_base / split_name
            split_dir.mkdir(exist_ok=True)
            
            for city in split_cities:
                if city in cities:
                    src = adj_base / f"{city}_adj.npy"
                    dst = split_dir / f"{city}_adj.npy"
                    shutil.move(str(src), str(dst))
        
        # Create splits for coordinates
        coord_base = Path(data_dir) / BASE_PATHS['coord'] / region / location
        for coord_type in ['original', 'transformed']:
            coord_path = coord_base / coord_type
            for split_name, split_cities in splits.items():
                split_dir = coord_path / split_name
                split_dir.mkdir(exist_ok=True)
                
                for city in split_cities:
                    if city in cities:
                        src = coord_path / f"{city}_coords.npy"
                        dst = split_dir / f"{city}_coords.npy"
                        shutil.move(str(src), str(dst))

def get_folder_stats(folder):
    """Calculate statistics for splits in a folder"""
    stats = {}
    total_files = 0
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(folder, split)
        if os.path.exists(split_path):
            n_files = len([f for f in os.listdir(split_path) if f.endswith('.npy')])
            stats[split] = n_files
            total_files += n_files
    
    # Calculate percentages
    if total_files > 0:
        stats.update({
            f"{k}_pct": (v/total_files)*100 
            for k, v in stats.items()
        })
    
    return stats, total_files

def print_statistics(folder_stats):
    """Print formatted statistics table"""
    headers = ["Folder", "Total", "Train", "Train%", "Val", "Val%", "Test", "Test%"]
    rows = []
    
    for folder, (stats, total) in folder_stats.items():
        row = [
            folder,
            total,
            stats.get('train', 0),
            f"{stats.get('train_pct', 0):.1f}%",
            stats.get('validation', 0),
            f"{stats.get('validation_pct', 0):.1f}%",
            stats.get('test', 0),
            f"{stats.get('test_pct', 0):.1f}%"
        ]
        rows.append(row)
    
    print("\nDataset Split Statistics:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def main():
    data_dir = Path("data")
    
    for min_nodes, max_nodes in NODE_RANGES:
        print(f"\nProcessing cities with {min_nodes}-{max_nodes} nodes")
        
        # Find valid cities within node range
        valid_cities = find_cities_in_range(data_dir, min_nodes, max_nodes)
        
        # Create splits for each region/location
        for (region, location), cities in valid_cities.items():
            if not cities:
                continue
                
            print(f"\nSplitting {region}/{location}: {len(cities)} cities")
            splits = create_splits(cities)
            
            # Reorganize files
            reorganize_files(data_dir, {(region, location): cities}, splits)
            
            # Save split information
            split_info = {
                'node_range': f"{min_nodes}-{max_nodes}",
                'splits': splits
            }
            
            info_file = Path(data_dir) / f"split_info_{region}_{location}_{min_nodes}_{max_nodes}.json"
            with open(info_file, 'w') as f:
                json.dump(split_info, f, indent=2)

if __name__ == "__main__":
    main()