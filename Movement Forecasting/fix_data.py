"""
Script to fix data quality issues in dataset.csv
"""

import pandas as pd
import numpy as np
import sys
import os

def fix_data(input_path="./data/dataset.csv", output_path=None):
    """Fix data quality issues."""
    if output_path is None:
        output_path = input_path
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"Original shape: {original_shape}")
    
    # Fix 1: Remove rows with missing coordinates
    print("\n1. Removing rows with missing coordinates...")
    before = len(df)
    df = df.dropna(subset=['player_location_x', 'player_location_y', 
                           'opponent_location_x', 'opponent_location_y'])
    after = len(df)
    removed = before - after
    print(f"   Removed {removed} rows with missing coordinates")
    
    # Fix 2: Remove rows with problematic shot types
    print("\n2. Removing rows with problematic shot types...")
    before = len(df)
    problematic_types = ['未知球種', 'nan', 'NaN', 'None', '']
    df = df[~df['type'].isin(problematic_types)]
    df = df[df['type'].notna()]
    after = len(df)
    removed = before - after
    print(f"   Removed {removed} rows with problematic shot types")
    
    # Fix 3: Normalize coordinates if they're not already normalized
    print("\n3. Checking and normalizing coordinates...")
    x_max = abs(df['player_location_x']).max()
    y_max = abs(df['player_location_y']).max()
    
    if x_max > 10 or y_max > 10:
        print(f"   Coordinates appear to be raw (max: x={x_max:.1f}, y={y_max:.1f})")
        print("   Normalizing coordinates...")
        
        # Use the same normalization as data_cleaner.py
        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        
        df['player_location_x'] = (df['player_location_x'] - mean_x) / std_x
        df['player_location_y'] = (df['player_location_y'] - mean_y) / std_y
        df['opponent_location_x'] = (df['opponent_location_x'] - mean_x) / std_x
        df['opponent_location_y'] = (df['opponent_location_y'] - mean_y) / std_y
        
        print(f"   ✓ Coordinates normalized")
        print(f"   New ranges: x=[{df['player_location_x'].min():.2f}, {df['player_location_x'].max():.2f}], "
              f"y=[{df['player_location_y'].min():.2f}, {df['player_location_y'].max():.2f}]")
    else:
        print(f"   ✓ Coordinates already appear normalized")
    
    # Fix 4: Remove rallies that are too short (less than encode_length + 1)
    print("\n4. Checking rally lengths...")
    if 'rally_id' in df.columns:
        rally_lengths = df.groupby('rally_id').size()
        min_length = 3  # Minimum rally length (encode_length=2 needs at least 3)
        short_rallies = rally_lengths[rally_lengths < min_length].index
        if len(short_rallies) > 0:
            before = len(df)
            df = df[~df['rally_id'].isin(short_rallies)]
            after = len(df)
            removed = before - after
            print(f"   Removed {len(short_rallies)} rallies with less than {min_length} shots")
            print(f"   Removed {removed} total rows")
        else:
            print(f"   ✓ All rallies have sufficient length")
    
    # Fix 5: Check for any remaining NaN/Inf
    print("\n5. Final check for NaN/Inf...")
    critical_cols = ['player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
            if nan_count > 0 or inf_count > 0:
                print(f"   ⚠️  {col}: {nan_count} NaN, {inf_count} Inf - removing rows...")
                df = df[~df[col].isna()]
                df = df[~np.isinf(df[col])]
            else:
                print(f"   ✓ {col}: No NaN/Inf")
    
    # Save fixed data
    print(f"\n6. Saving fixed data...")
    print(f"   Final shape: {df.shape} (removed {original_shape[0] - df.shape[0]} rows)")
    
    # Create backup
    if input_path == output_path:
        backup_path = input_path.replace('.csv', '_backup.csv')
        print(f"   Creating backup: {backup_path}")
        pd.read_csv(input_path).to_csv(backup_path, index=False)
    
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "./data/dataset.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = fix_data(input_path, output_path)
    print("\n" + "="*80)
    print("Data fixing complete!")
    print("="*80)

