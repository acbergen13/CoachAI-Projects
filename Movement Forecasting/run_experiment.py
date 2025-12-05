#!/usr/bin/env python
"""
Windows-compatible experiment runner for Movement Forecasting.
This script replicates the functionality of experiment.sh for Windows users.
"""

import subprocess
import sys
import os
from datetime import datetime

# Configuration - modify these as needed
MODEL_LIST = ["DyMF"]  # Options: "DyMF", "LSTM", "GCN", "ShuttleNet", "rGCN", "Transformer"
SEEDS = [1]
SEQUENCE_LENGTHS = [2, 4, 8]  # encode_length values

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    return True

def main():
    """Main experiment loop."""
    print("Starting Movement Forecasting Experiment")
    print("="*80)
    
    for model in MODEL_LIST:
        for seed in SEEDS:
            print(f"\n{'='*80}")
            print(f"Seed: {seed}")
            print(f"{'='*80}")
            
            for encode_length in SEQUENCE_LENGTHS:
                # Use Windows-compatible timestamp format (replace colons with hyphens)
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                model_name = model
                model_path = f"./model/{model_name}_{encode_length}_{current_time}"
                
                print(f"\nModel Name: {model_name}\tEncode Length: {encode_length}")
                print(f"Model Path: {model_path}")
                
                # Training
                print("\nTraining Start:")
                train_cmd = [
                    sys.executable, "train.py",
                    "--model_type", model_name,
                    "--model_folder", model_path,
                    "--encode_length", str(encode_length),
                    "--seed", str(seed)
                ]
                
                if not run_command(train_cmd, f"Training {model_name} with encode_length={encode_length}"):
                    print(f"Training failed for {model_name} with encode_length={encode_length}")
                    continue
                
                # Evaluation
                print("\nEvaluating Start:")
                eval_cmd = [
                    sys.executable, "evaluate.py",
                    model_path,
                    "10"  # sample_num
                ]
                
                if not run_command(eval_cmd, f"Evaluating {model_name} with encode_length={encode_length}"):
                    print(f"Evaluation failed for {model_name} with encode_length={encode_length}")
                
                print("\n" + "="*80)
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()

