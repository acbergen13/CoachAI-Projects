"""
Script to check and diagnose data quality issues in dataset.csv
"""

import pandas as pd
import numpy as np
import sys

def check_data(data_path="./data/dataset.csv"):
    """Check for data quality issues."""
    print(f"Checking data file: {data_path}")
    print("="*80)
    
    df = pd.read_csv(data_path)
    
    # Required columns for training
    required_cols = ['rally_id', 'player', 'type', 'player_location_x', 'player_location_y', 
                     'opponent_location_x', 'opponent_location_y', 'ball_round', 'set', 'match_id']
    
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ ERROR: Missing required columns: {missing_cols}")
    else:
        print(f"\n✓ All required columns present")
    
    # Check for NaN/Inf values in critical columns
    print(f"\n2. Checking for NaN/Inf values:")
    critical_cols = ['player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y', 'type']
    
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
            if nan_count > 0:
                print(f"   ❌ {col}: {nan_count} NaN values")
            if inf_count > 0:
                print(f"   ❌ {col}: {inf_count} Inf values")
            if nan_count == 0 and inf_count == 0:
                print(f"   ✓ {col}: No NaN/Inf values")
    
    # Check coordinate ranges
    print(f"\n3. Checking coordinate ranges:")
    if 'player_location_x' in df.columns:
        x_min, x_max = df['player_location_x'].min(), df['player_location_x'].max()
        y_min, y_max = df['player_location_y'].min(), df['player_location_y'].max()
        print(f"   player_location_x: [{x_min:.2f}, {x_max:.2f}]")
        print(f"   player_location_y: [{y_min:.2f}, {y_max:.2f}]")
        
        # Check if coordinates are normalized (should be around -2 to 2 if normalized)
        # Raw coordinates are typically 0-1000 range
        if abs(x_max) > 10 or abs(y_max) > 10:
            print(f"   ⚠️  WARNING: Coordinates appear to be RAW (not normalized)")
            print(f"      Expected normalized range: approximately [-2, 2]")
            print(f"      Your data range suggests raw pixel coordinates")
            print(f"      This could cause training issues!")
        else:
            print(f"   ✓ Coordinates appear normalized")
    
    # Check for empty shot types
    print(f"\n4. Checking shot types:")
    if 'type' in df.columns:
        unique_types = df['type'].unique()
        print(f"   Unique shot types: {len(unique_types)}")
        
        # Check for problematic types
        problematic = ['未知球種', 'nan', 'NaN', 'None', '']
        found_problematic = [t for t in unique_types if str(t) in problematic or pd.isna(t)]
        if found_problematic:
            print(f"   ⚠️  WARNING: Found problematic shot types: {found_problematic}")
            count = df['type'].isin(found_problematic).sum()
            print(f"      Count: {count} rows")
        else:
            print(f"   ✓ No problematic shot types found")
    
    # Check rally lengths
    print(f"\n5. Checking rally data:")
    if 'rally_id' in df.columns:
        rally_lengths = df.groupby('rally_id').size()
        min_rally = rally_lengths.min()
        max_rally = rally_lengths.max()
        avg_rally = rally_lengths.mean()
        print(f"   Rally lengths: min={min_rally}, max={max_rally}, avg={avg_rally:.1f}")
        
        # Check for very short rallies
        short_rallies = (rally_lengths < 3).sum()
        if short_rallies > 0:
            print(f"   ⚠️  WARNING: {short_rallies} rallies have less than 3 shots (may be filtered)")
    
    # Check for rows with missing coordinates
    print(f"\n6. Checking rows with missing coordinates:")
    if all(col in df.columns for col in ['player_location_x', 'player_location_y', 'opponent_location_x', 'opponent_location_y']):
        missing_coords = df[
            df['player_location_x'].isna() | 
            df['player_location_y'].isna() | 
            df['opponent_location_x'].isna() | 
            df['opponent_location_y'].isna()
        ]
        if len(missing_coords) > 0:
            print(f"   ❌ Found {len(missing_coords)} rows with missing coordinates")
            print(f"      These rows will cause training issues!")
            print(f"      Example rows:")
            print(missing_coords[['rally_id', 'player', 'type', 'player_location_x', 'player_location_y']].head())
        else:
            print(f"   ✓ No rows with missing coordinates")
    
    print("\n" + "="*80)
    print("Data check complete!")
    
    return df

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/dataset.csv"
    df = check_data(data_path)

