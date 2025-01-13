import os
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict

NODE_RANGES = [
    (10, 50, '10-50'),
    (50, 100, '50-100'),
    (100, 500, '100-500'),
    (500, 1000, '500-1000'),
    (1000, 2000, '1000-2000')
]

SPLITS = ['train', 'valid', 'test', 'rest']

def get_node_count(file_path):
    """Get number of nodes from file"""
    try:
        matrix = np.load(file_path)
        return matrix.shape[0] if isinstance(matrix, np.ndarray) else len(matrix)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def get_range_label(node_count):
    """Categorize node count into ranges"""
    for min_nodes, max_nodes, label in NODE_RANGES:
        if min_nodes <= node_count <= max_nodes:
            return label
    return 'other'

def check_file_consistency(data_dir):
    """Check consistency of files across folders"""
    stats = defaultdict(lambda: defaultdict(lambda: {
        'ranges': defaultdict(int),
        'splits': defaultdict(int),
        'missing_files': 0,
        'size_mismatches': 0,
        'total_files': 0
    }))
    
    for region in ['germany', 'world']:
        for location in ['center', 'whole']:
            adj_path = Path(data_dir) / 'adj_matrices' / region / location
            if not adj_path.exists():
                continue

            stat = stats[region][location]
            
            # Process each split directory
            for split in SPLITS:
                split_path = adj_path / split
                if not split_path.exists():
                    continue
                
                for adj_file in split_path.glob('*_adj.npy'):
                    city = adj_file.stem.replace('_adj', '')
                    stat['total_files'] += 1
                    stat['splits'][split] += 1

                    # Check coordinate files
                    coord_orig = Path(data_dir) / 'coordinates' / region / location / 'original' / split / f"{city}_coords.npy"
                    coord_trans = Path(data_dir) / 'coordinates' / region / location / 'transformed' / split / f"{city}_coords.npy"
                    
                    # Count missing files
                    if not (coord_orig.exists() and coord_trans.exists()):
                        stat['missing_files'] += 1
                        continue

                    # Check sizes
                    adj_nodes = get_node_count(adj_file)
                    orig_nodes = get_node_count(coord_orig)
                    trans_nodes = get_node_count(coord_trans)
                    
                    if not (adj_nodes == orig_nodes == trans_nodes):
                        stat['size_mismatches'] += 1
                    else:
                        range_label = get_range_label(adj_nodes)
                        stat['ranges'][range_label] += 1

    return stats

def print_statistics(stats):
    """Print distribution statistics and inconsistencies"""
    print("\nNode Distribution Statistics")
    print("=" * 80)
    
    for region in stats:
        print(f"\nRegion: {region}")
        print("-" * 40)
        
        for location in stats[region]:
            stat = stats[region][location]
            print(f"\nLocation: {location}")
            
            # Node range distribution
            range_table = [[range_label, count] 
                          for range_label, count in stat['ranges'].items()]
            print("\nNode Range Distribution:")
            print(tabulate(range_table, headers=['Range', 'Count'], tablefmt='grid'))
            
            # Split distribution
            split_table = [[split, stat['splits'].get(split, 0)] 
                          for split in SPLITS]
            print("\nSplit Distribution:")
            print(tabulate(split_table, headers=['Split', 'Count'], tablefmt='grid'))
            
            # Summary
            summary = [
                ['Total Files', stat['total_files']],
                ['Missing Coordinate Files', stat['missing_files']],
                ['Size Mismatches', stat['size_mismatches']]
            ]
            print("\nSummary:")
            print(tabulate(summary, headers=['Metric', 'Count'], tablefmt='grid'))

def main():
    data_dir = Path("data")
    stats = check_file_consistency(data_dir)
    print_statistics(stats)

if __name__ == "__main__":
    main()