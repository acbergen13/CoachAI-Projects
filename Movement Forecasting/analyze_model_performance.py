"""
Comprehensive analysis script for DyMF model across multiple configurations.
Generates publication-ready comparative visualizations showing model effectiveness.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import sys
import glob
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from prepare_dataset import prepare_dataset
from utils import load_args_file

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def collect_evaluation_results(model_folders):
    """Collect evaluation results from multiple model folders."""
    results = []
    
    for model_folder in model_folders:
        if not os.path.exists(model_folder):
            print(f"Warning: Model folder not found: {model_folder}")
            continue
        
        # Load model args
        try:
            args = load_args_file(model_folder)
        except:
            print(f"Warning: Could not load args from {model_folder}")
            continue
        
        # Try to run evaluation and capture results
        print(f"Evaluating model: {model_folder}")
        try:
            # Import evaluation function
            from DyMF.runner import evaluate
            from DyMF.model import Encoder, Decoder
            import torch.nn as nn
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Prepare data
            train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)
            
            # Load model
            encoder = Encoder(args, device)
            decoder = Decoder(args, device)
            encoder.load_state_dict(torch.load(os.path.join(model_folder, 'encoder')))
            decoder.load_state_dict(torch.load(os.path.join(model_folder, 'decoder')))
            
            # Evaluation criteria
            location_MSE_criterion = nn.MSELoss(reduction='sum')
            location_MAE_criterion = nn.L1Loss(reduction='sum')
            shot_type_criterion = nn.CrossEntropyLoss()
            
            encoder.to(device)
            decoder.to(device)
            
            # Run evaluation
            total_loss, mse_loss, mae_loss, type_loss = evaluate(
                test_dataloader, encoder, decoder, 
                location_MSE_criterion, location_MAE_criterion, shot_type_criterion,
                args, device=device
            )
            
            results.append({
                'model_folder': model_folder,
                'encode_length': args.get('encode_length', 'Unknown'),
                'seed': args.get('seed', 'Unknown'),
                'lr': args.get('lr', 'Unknown'),
                'epochs': args.get('epochs', 'Unknown'),
                'total_loss': float(total_loss),
                'location_MSE': float(mse_loss),
                'location_MAE': float(mae_loss),
                'type_loss': float(type_loss),
                'hidden_size': args.get('hidden_size', 'Unknown'),
                'batch_size': args.get('train_batch_size', 'Unknown')
            })
            
            print(f"  ✓ Total Loss: {total_loss:.4f}, MAE: {mae_loss:.4f}, Type Loss: {type_loss:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model_folder}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def create_comprehensive_comparison(df, save_dir='visualizations'):
    """Create comprehensive comparison visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    if len(df) == 0:
        print("No results to visualize!")
        return
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Encode length comparison
    if 'encode_length' in df.columns:
        encode_groups = df.groupby('encode_length').agg({
            'location_MAE': 'mean',
            'location_MSE': 'mean',
            'type_loss': 'mean',
            'total_loss': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(encode_groups))
        width = 0.2
        
        axes[0, 0].bar(x_pos - 1.5*width, encode_groups['location_MAE'], width, 
                      label='Location MAE', color='steelblue', alpha=0.8)
        axes[0, 0].bar(x_pos - 0.5*width, encode_groups['location_MSE']/10, width,
                      label='Location MSE/10', color='coral', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.5*width, encode_groups['type_loss'], width,
                      label='Type Loss', color='mediumseagreen', alpha=0.8)
        axes[0, 0].bar(x_pos + 1.5*width, encode_groups['total_loss']/100, width,
                      label='Total Loss/100', color='purple', alpha=0.8)
        
        axes[0, 0].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Performance by Encode Length', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(encode_groups['encode_length'])
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Location MAE comparison
    if 'encode_length' in df.columns:
        sns.boxplot(data=df, x='encode_length', y='location_MAE', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Location MAE', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Location Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Type Loss comparison
    if 'encode_length' in df.columns:
        sns.boxplot(data=df, x='encode_length', y='type_loss', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Shot Type Loss', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Shot Type Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Scatter: Location MAE vs Type Loss
    scatter = axes[1, 1].scatter(df['location_MAE'], df['type_loss'], 
                                c=df['encode_length'] if 'encode_length' in df.columns else range(len(df)),
                                s=100, alpha=0.6, cmap='viridis')
    axes[1, 1].set_xlabel('Location MAE', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Shot Type Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Location vs Shot Type Performance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    if 'encode_length' in df.columns:
        plt.colorbar(scatter, ax=axes[1, 1], label='Encode Length')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive comparison to: {os.path.join(save_dir, 'comprehensive_comparison.png')}")
    plt.close()

def create_performance_table(df, save_dir='visualizations'):
    """Create a detailed performance table."""
    os.makedirs(save_dir, exist_ok=True)
    
    if len(df) == 0:
        return
    
    # Create summary statistics
    summary = df.groupby('encode_length').agg({
        'location_MAE': ['mean', 'std', 'min', 'max'],
        'location_MSE': ['mean', 'std', 'min', 'max'],
        'type_loss': ['mean', 'std', 'min', 'max'],
        'total_loss': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Save to CSV
    summary.to_csv(os.path.join(save_dir, 'performance_summary.csv'))
    print(f"Saved performance summary to: {os.path.join(save_dir, 'performance_summary.csv')}")
    
    # Create formatted table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for encode_len in sorted(df['encode_length'].unique()):
        subset = df[df['encode_length'] == encode_len]
        table_data.append([
            f"Encode={encode_len}",
            f"{subset['location_MAE'].mean():.2f} ± {subset['location_MAE'].std():.2f}",
            f"{subset['location_MSE'].mean():.2f} ± {subset['location_MSE'].std():.2f}",
            f"{subset['type_loss'].mean():.2f} ± {subset['type_loss'].std():.2f}",
            f"{subset['total_loss'].mean():.2f} ± {subset['total_loss'].std():.2f}",
            len(subset)
        ])
    
    # Overall statistics
    table_data.append([
        "Overall",
        f"{df['location_MAE'].mean():.2f} ± {df['location_MAE'].std():.2f}",
        f"{df['location_MSE'].mean():.2f} ± {df['location_MSE'].std():.2f}",
        f"{df['type_loss'].mean():.2f} ± {df['type_loss'].std():.2f}",
        f"{df['total_loss'].mean():.2f} ± {df['total_loss'].std():.2f}",
        len(df)
    ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Configuration', 'Location MAE', 'Location MSE', 
                              'Type Loss', 'Total Loss', 'N'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('DyMF Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'performance_table.png'), dpi=300, bbox_inches='tight')
    print(f"Saved performance table to: {os.path.join(save_dir, 'performance_table.png')}")
    plt.close()

def create_statistical_analysis(df, save_dir='visualizations'):
    """Create statistical analysis visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Performance improvement over encode length
    if 'encode_length' in df.columns:
        encode_stats = df.groupby('encode_length').agg({
            'location_MAE': 'mean',
            'type_loss': 'mean',
            'total_loss': 'mean'
        }).reset_index()
        
        axes[0].plot(encode_stats['encode_length'], encode_stats['location_MAE'], 
                    'o-', linewidth=2, markersize=10, label='Location MAE', color='steelblue')
        axes[0].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Location MAE', fontsize=12, fontweight='bold')
        axes[0].set_title('Location Prediction vs Sequence Length', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    
    # 2. Type loss trend
    if 'encode_length' in df.columns:
        axes[1].plot(encode_stats['encode_length'], encode_stats['type_loss'],
                    's-', linewidth=2, markersize=10, label='Type Loss', color='coral')
        axes[1].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Shot Type Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Shot Type Prediction vs Sequence Length', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    
    # 3. Overall performance radar (if multiple metrics)
    metrics = ['location_MAE', 'type_loss']
    if len(encode_stats) > 0:
        # Normalize metrics for comparison
        normalized_mae = 1 - (encode_stats['location_MAE'] - encode_stats['location_MAE'].min()) / (encode_stats['location_MAE'].max() - encode_stats['location_MAE'].min() + 1e-6)
        normalized_type = 1 - (encode_stats['type_loss'] - encode_stats['type_loss'].min()) / (encode_stats['type_loss'].max() - encode_stats['type_loss'].min() + 1e-6)
        
        x = np.arange(len(encode_stats))
        width = 0.35
        axes[2].bar(x - width/2, normalized_mae, width, label='Location (normalized)', alpha=0.8, color='steelblue')
        axes[2].bar(x + width/2, normalized_type, width, label='Type (normalized)', alpha=0.8, color='coral')
        axes[2].set_xlabel('Encode Length', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Normalized Performance (1=best)', fontsize=12, fontweight='bold')
        axes[2].set_title('Normalized Performance Comparison', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(encode_stats['encode_length'])
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved statistical analysis to: {os.path.join(save_dir, 'statistical_analysis.png')}")
    plt.close()

def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_model_performance.py <model_folder1> [model_folder2] ...")
        print("Or: python analyze_model_performance.py --auto (to find all models in ./model/)")
        sys.exit(1)
    
    # Collect model folders
    if sys.argv[1] == '--auto':
        # Auto-discover models
        model_folders = glob.glob('./model/DyMF_*')
        if not model_folders:
            print("No DyMF models found in ./model/ directory")
            sys.exit(1)
        print(f"Auto-discovered {len(model_folders)} models")
    else:
        model_folders = sys.argv[1:]
    
    print("="*80)
    print("DyMF Model Comprehensive Performance Analysis")
    print("="*80)
    print(f"Analyzing {len(model_folders)} model(s)...")
    print()
    
    # Collect results
    df = collect_evaluation_results(model_folders)
    
    if len(df) == 0:
        print("No valid results collected!")
        sys.exit(1)
    
    print(f"\nCollected results from {len(df)} model evaluation(s)")
    print("\nSummary:")
    print(df[['encode_length', 'location_MAE', 'type_loss', 'total_loss']].to_string())
    
    # Create visualizations
    save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80)
    
    # 1. Comprehensive comparison
    print("\n1. Creating comprehensive comparison...")
    create_comprehensive_comparison(df, save_dir)
    
    # 2. Performance table
    print("\n2. Creating performance table...")
    create_performance_table(df, save_dir)
    
    # 3. Statistical analysis
    print("\n3. Creating statistical analysis...")
    create_statistical_analysis(df, save_dir)
    
    # Save raw results
    df.to_csv(os.path.join(save_dir, 'raw_results.csv'), index=False)
    print(f"\nSaved raw results to: {os.path.join(save_dir, 'raw_results.csv')}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"All visualizations saved to: {save_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()

