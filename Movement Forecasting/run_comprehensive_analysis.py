"""
Helper script to run multiple experiments and then analyze all results.
This automates the process of generating comprehensive model comparisons.
"""

import subprocess
import sys
import os
import glob
from pathlib import Path

def run_experiments(configs):
    """Run multiple training experiments with different configurations."""
    trained_models = []
    
    print("="*80)
    print("Running Multiple Experiments for Comprehensive Analysis")
    print("="*80)
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment: {config['name']}")
        print(f"  Configuration: {config}")
        
        # Build command
        cmd = [
            sys.executable, "train.py",
            "--model_type", config.get('model_type', 'DyMF'),
            "--encode_length", str(config.get('encode_length', 2)),
            "--seed", str(config.get('seed', 1)),
            "--lr", str(config.get('lr', 0.0001)),
            "--epochs", str(config.get('epochs', 100))
        ]
        
        # Add optional parameters
        if 'train_batch_size' in config:
            cmd.extend(["--train_batch_size", str(config['train_batch_size'])])
        if 'hidden_size' in config:
            cmd.extend(["--hidden_size", str(config['hidden_size'])])
        
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                # Extract model folder from output
                output_lines = result.stdout.split('\n')
                model_folder = None
                for line in output_lines:
                    if './model/' in line or 'model/' in line:
                        model_folder = line.strip()
                        break
                
                if model_folder:
                    trained_models.append(model_folder)
                    print(f"  ✓ Training completed: {model_folder}")
                else:
                    print(f"  ⚠️  Training completed but model path not found")
            else:
                print(f"  ✗ Training failed with return code {result.returncode}")
                print(f"  Error: {result.stderr[:200]}")
                
        except Exception as e:
            print(f"  ✗ Error running experiment: {str(e)}")
            continue
    
    return trained_models

def main():
    """Main function to run experiments and analysis."""
    
    # Define experiment configurations
    # You can modify these to test different hyperparameters
    experiment_configs = [
        {
            'name': 'DyMF_encode2_lr0.0001',
            'model_type': 'DyMF',
            'encode_length': 2,
            'seed': 1,
            'lr': 0.0001,
            'epochs': 150
        },
        {
            'name': 'DyMF_encode4_lr0.0001',
            'model_type': 'DyMF',
            'encode_length': 4,
            'seed': 1,
            'lr': 0.0001,
            'epochs': 150
        },
        {
            'name': 'DyMF_encode8_lr0.0001',
            'model_type': 'DyMF',
            'encode_length': 8,
            'seed': 1,
            'lr': 0.0001,
            'epochs': 150
        },
        # Add more configurations as needed
        # {
        #     'name': 'DyMF_encode2_lr0.00005',
        #     'model_type': 'DyMF',
        #     'encode_length': 2,
        #     'seed': 1,
        #     'lr': 0.00005,
        #     'epochs': 150
        # },
    ]
    
    print("Experiment Configurations:")
    for i, config in enumerate(experiment_configs, 1):
        print(f"  {i}. {config['name']}: encode_length={config['encode_length']}, lr={config['lr']}, epochs={config['epochs']}")
    
    response = input("\nRun these experiments? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    trained_models = run_experiments(experiment_configs)
    
    if not trained_models:
        print("\nNo models were successfully trained!")
        return
    
    print(f"\n{'='*80}")
    print(f"Successfully trained {len(trained_models)} model(s)")
    print(f"{'='*80}")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    cmd = [sys.executable, "analyze_model_performance.py"] + trained_models
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode == 0:
        print("\n✓ Comprehensive analysis complete!")
        print("Check the 'visualizations/' folder for all results.")
    else:
        print("\n✗ Analysis failed. Run manually:")
        print(f"  python analyze_model_performance.py {' '.join(trained_models)}")

if __name__ == "__main__":
    main()

