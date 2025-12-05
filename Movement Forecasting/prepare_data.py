"""
Helper script to prepare dataset.csv by adding rally_id if missing.
This script ensures the dataset has the required rally_id column.
"""

import pandas as pd
import sys
import os

def add_rally_id(input_csv, output_csv=None):
    """
    Add rally_id column to dataset if it's missing.
    Also adds match_id and set columns if they're missing.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (default: overwrites input)
    """
    print(f"Reading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check if rally_id already exists
    if 'rally_id' in df.columns:
        print("✓ rally_id column already exists in the dataset")
        # Still check for match_id and set
        if 'match_id' not in df.columns or 'set' not in df.columns:
            print("⚠ Warning: rally_id exists but match_id or set is missing")
            print("  Adding match_id and set columns...")
        else:
            return df
    
    # Add match_id if missing
    if 'match_id' not in df.columns:
        print("⚠ match_id column not found. Adding match_id=1 (assuming single match)...")
        df['match_id'] = 1
    
    # Add set column if missing
    if 'set' not in df.columns:
        print("⚠ set column not found. Adding set=1 (assuming single set)...")
        # Try to infer set from roundscore changes if possible
        # For now, just set to 1 for all rows
        df['set'] = 1
    
    # Check for rally column
    if 'rally' not in df.columns:
        print("ERROR: 'rally' column is required but not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Create rally_id by grouping
    print("Creating rally_id from match_id, set, and rally...")
    df['rally_id'] = df.groupby(['match_id', 'set', 'rally']).ngroup()
    
    print(f"✓ Created rally_id column with {df['rally_id'].nunique()} unique rallies")
    print(f"✓ Dataset now has match_id, set, and rally_id columns")
    
    # Save the updated dataset
    if output_csv is None:
        output_csv = input_csv
    
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved updated dataset to: {output_csv}")
    
    return df

if __name__ == "__main__":
    # Default path
    default_path = "./data/dataset.csv"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_path
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = None
    
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        print(f"Usage: python prepare_data.py [input_csv] [output_csv]")
        sys.exit(1)
    
    result = add_rally_id(input_path, output_path)
    
    if result is not None:
        print("\nDataset preparation completed successfully!")
        print(f"Columns in dataset: {list(result.columns)}")
    else:
        print("\nDataset preparation failed!")
        sys.exit(1)

