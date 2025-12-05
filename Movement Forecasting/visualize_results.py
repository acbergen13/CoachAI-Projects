"""
Comprehensive visualization script for Movement Forecasting model performance.
Generates publication-ready figures for research posters.
"""

import torch
import torch.distributions as torchdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
import sys
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

from prepare_dataset import prepare_dataset
from utils import load_args_file

# Shot type translations
SHOT_TYPE_TRANSLATIONS = {
    '放小球': 'net shot',
    '擋小球': 'return net',
    '殺球': 'smash',
    '點扣': 'wrist smash',
    '挑球': 'lob',
    '防守回挑': 'defensive return lob',
    '長球': 'clear',
    '平球': 'drive',
    '小平球': 'driven flight',
    '後場抽平球': 'back-court drive',
    '切球': 'drop',
    '過度切球': 'passive drop',
    '推球': 'push',
    '撲球': 'rush',
    '防守回抽': 'defensive return drive',
    '勾球': 'cross-court net shot',
    '發短球': 'short service',
    '發長球': 'long service'
}

def draw_badminton_court(ax, court_width=710, court_length=1340):
    """Draw a badminton court outline."""
    # Court dimensions (in pixels, approximate)
    # Full court: 6.1m x 13.4m
    
    # Outer lines
    ax.add_patch(patches.Rectangle((0, 0), court_width, court_length, 
                                   linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3))
    
    # Center line
    ax.plot([court_width/2, court_width/2], [0, court_length], 'k-', linewidth=1.5)
    
    # Service boxes
    service_line = court_length / 2
    # Front service line
    ax.plot([0, court_width], [service_line, service_line], 'k--', linewidth=1, alpha=0.5)
    
    # Center service boxes
    ax.plot([0, court_width/2], [service_line, service_line], 'k-', linewidth=1)
    ax.plot([court_width/2, court_width], [service_line, service_line], 'k-', linewidth=1)
    
    ax.set_xlim(-50, court_width + 50)
    ax.set_ylim(-50, court_length + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Badminton Court', fontsize=14, fontweight='bold')

def visualize_trajectories(model_folder, num_rallies=3, save_path='trajectory_visualization.png'):
    """Visualize actual vs predicted player trajectories on court."""
    print(f"Loading model from: {model_folder}")
    args = load_args_file(model_folder)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from DyMF.model import Encoder, Decoder
    encoder = Encoder(args, device)
    decoder = Decoder(args, device)
    encoder.load_state_dict(torch.load(os.path.join(model_folder, 'encoder')))
    decoder.load_state_dict(torch.load(os.path.join(model_folder, 'decoder')))
    encoder.eval()
    decoder.eval()
    
    # Prepare data
    train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)
    
    # Denormalization parameters
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    court_height = 960
    
    fig, axes = plt.subplots(1, num_rallies, figsize=(6*num_rallies, 8))
    if num_rallies == 1:
        axes = [axes]
    
    rally_count = 0
    with torch.no_grad():
        for rally, target in test_dataloader:
            if rally_count >= num_rallies:
                break
            
            ax = axes[rally_count]
            draw_badminton_court(ax)
            
            # Get data
            player = rally[0].to(device).long()
            shot_type = rally[1].to(device).long()
            player_A_x = rally[2].to(device)
            player_A_y = rally[3].to(device)
            player_B_x = rally[4].to(device)
            player_B_y = rally[5].to(device)
            
            # Get actual trajectories (denormalize)
            actual_A_x = (player_A_x[0].cpu().numpy() * std_x + mean_x)
            actual_A_y = court_height - (player_A_y[0].cpu().numpy() * std_y + mean_y)
            actual_B_x = (player_B_x[0].cpu().numpy() * std_x + mean_x)
            actual_B_y = court_height - (player_B_y[0].cpu().numpy() * std_y + mean_y)
            
            # Get predictions
            encode_length = args['encode_length']
            encoder_player = player[:, 0:2]
            encoder_shot_type = shot_type[:, :encode_length-1]
            encoder_player_A_x = player_A_x[:, :encode_length]
            encoder_player_A_y = player_A_y[:, :encode_length]
            encoder_player_B_x = player_B_x[:, :encode_length]
            encoder_player_B_y = player_B_y[:, :encode_length]
            
            encode_node_embedding, original_embedding, adjacency_matrix = encoder(
                encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y,
                encoder_player_B_x, encoder_player_B_y, encode_length
            )
            
            # Predict next position
            decoder_player = player[:, 0:2]
            decode_node_embedding = encode_node_embedding.clone()
            
            predict_xy, predict_shot_type_logit, _, _, _ = decoder(
                decoder_player, encode_length+1, decode_node_embedding, original_embedding,
                adjacency_matrix, encoder_player_A_x[:, -1:], encoder_player_A_y[:, -1:],
                encoder_player_B_x[:, -1:], encoder_player_B_y[:, -1:],
                shot_type=None, train=False, first=True
            )
            
            # Extract predicted positions
            predict_A_xy = predict_xy[:, 0, :]
            predict_B_xy = predict_xy[:, 1, :]
            
            # Denormalize predictions
            pred_A_x = (predict_A_xy[0, 0].item() * std_x + mean_x)
            pred_A_y = court_height - (predict_A_xy[0, 1].item() * std_y + mean_y)
            pred_B_x = (predict_B_xy[0, 0].item() * std_x + mean_x)
            pred_B_y = court_height - (predict_B_xy[0, 1].item() * std_y + mean_y)
            
            # Plot actual trajectories
            valid_A = ~np.isnan(actual_A_x) & ~np.isnan(actual_A_y)
            valid_B = ~np.isnan(actual_B_x) & ~np.isnan(actual_B_y)
            
            if valid_A.any():
                ax.plot(actual_A_x[valid_A], actual_A_y[valid_A], 'b-', linewidth=2, 
                        label='Player A (Actual)', alpha=0.7, marker='o', markersize=4)
            if valid_B.any():
                ax.plot(actual_B_x[valid_B], actual_B_y[valid_B], 'r-', linewidth=2,
                        label='Player B (Actual)', alpha=0.7, marker='s', markersize=4)
            
            # Plot predicted positions
            ax.plot(pred_A_x, pred_A_y, 'b*', markersize=15, label='Player A (Predicted)', 
                   markeredgecolor='darkblue', markeredgewidth=2)
            ax.plot(pred_B_x, pred_B_y, 'r*', markersize=15, label='Player B (Predicted)',
                   markeredgecolor='darkred', markeredgewidth=2)
            
            # Draw uncertainty ellipse for predictions
            sx = np.exp(predict_A_xy[0, 2].item()) * std_x
            sy = np.exp(predict_A_xy[0, 3].item()) * std_y
            corr = np.tanh(predict_A_xy[0, 4].item())
            
            # Create covariance ellipse
            from matplotlib.patches import Ellipse
            ellipse_A = Ellipse((pred_A_x, pred_A_y), width=2*sx, height=2*sy,
                               angle=0, alpha=0.3, color='blue', label='Uncertainty (A)')
            ax.add_patch(ellipse_A)
            
            sx = np.exp(predict_B_xy[0, 2].item()) * std_x
            sy = np.exp(predict_B_xy[0, 3].item()) * std_y
            ellipse_B = Ellipse((pred_B_x, pred_B_y), width=2*sx, height=2*sy,
                               angle=0, alpha=0.3, color='red', label='Uncertainty (B)')
            ax.add_patch(ellipse_B)
            
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'Rally {rally_count + 1}', fontsize=12, fontweight='bold')
            
            rally_count += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory visualization to: {save_path}")
    plt.close()

def create_performance_comparison(model_folders, save_path='performance_comparison.png'):
    """Create bar chart comparing different model configurations."""
    results = []
    
    for model_folder in model_folders:
        if not os.path.exists(model_folder):
            continue
        
        # Try to load evaluation results
        # For now, we'll create a placeholder - you can modify this to load actual results
        args = load_args_file(model_folder)
        encode_length = args.get('encode_length', 'Unknown')
        
        # You would load actual evaluation results here
        # For demonstration, using placeholder structure
        results.append({
            'Model': f'DyMF (encode={encode_length})',
            'Location MAE': 13.0,  # Replace with actual values
            'Location MSE': 70.0,  # Replace with actual values
            'Type Loss': 2.8       # Replace with actual values
        })
    
    if not results:
        print("No valid model folders found")
        return
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Location MAE
    axes[0].bar(df['Model'], df['Location MAE'], color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Location MAE', fontsize=12, fontweight='bold')
    axes[0].set_title('Location Prediction Error (MAE)', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Location MSE
    axes[1].bar(df['Model'], df['Location MSE'], color='coral', alpha=0.7)
    axes[1].set_ylabel('Location MSE', fontsize=12, fontweight='bold')
    axes[1].set_title('Location Prediction Error (MSE)', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Type Loss
    axes[2].bar(df['Model'], df['Type Loss'], color='mediumseagreen', alpha=0.7)
    axes[2].set_ylabel('Shot Type Loss', fontsize=12, fontweight='bold')
    axes[2].set_title('Shot Type Prediction Error', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to: {save_path}")
    plt.close()

def create_shot_type_analysis(model_folder, save_path='shot_type_analysis.png'):
    """Analyze and visualize shot type prediction performance."""
    print(f"Analyzing shot type predictions from: {model_folder}")
    
    args = load_args_file(model_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from DyMF.model import Encoder, Decoder
    encoder = Encoder(args, device)
    decoder = Decoder(args, device)
    encoder.load_state_dict(torch.load(os.path.join(model_folder, 'encoder')))
    decoder.load_state_dict(torch.load(os.path.join(model_folder, 'decoder')))
    encoder.eval()
    decoder.eval()
    
    train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)
    
    all_predictions = []
    all_targets = []
    
    encode_length = args['encode_length']
    
    with torch.no_grad():
        for rally, target in test_dataloader:
            player = rally[0].to(device).long()
            shot_type = rally[1].to(device).long()
            player_A_x = rally[2].to(device)
            player_A_y = rally[3].to(device)
            player_B_x = rally[4].to(device)
            player_B_y = rally[5].to(device)
            
            target_type = target[4].to(device)
            
            encoder_player = player[:, 0:2]
            encoder_shot_type = shot_type[:, :encode_length-1]
            encoder_player_A_x = player_A_x[:, :encode_length]
            encoder_player_A_y = player_A_y[:, :encode_length]
            encoder_player_B_x = player_B_x[:, :encode_length]
            encoder_player_B_y = player_B_y[:, :encode_length]
            
            encode_node_embedding, original_embedding, adjacency_matrix = encoder(
                encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y,
                encoder_player_B_x, encoder_player_B_y, encode_length
            )
            
            decoder_player = player[:, 0:2]
            decode_node_embedding = encode_node_embedding.clone()
            
            predict_xy, predict_shot_type_logit, _, _, _ = decoder(
                decoder_player, encode_length+1, decode_node_embedding, original_embedding,
                adjacency_matrix, encoder_player_A_x[:, -1:], encoder_player_A_y[:, -1:],
                encoder_player_B_x[:, -1:], encoder_player_B_y[:, -1:],
                shot_type=None, train=False, first=True
            )
            
            # Get predictions
            pred_type = torch.argmax(predict_shot_type_logit, dim=-1)
            all_predictions.extend(pred_type.cpu().numpy().flatten())
            all_targets.extend(target_type[:, encode_length-2].cpu().numpy().flatten())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Get shot type names
    shot_type_names = [f'Type {i}' for i in range(len(cm))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=shot_type_names, yticklabels=shot_type_names)
    axes[0].set_xlabel('Predicted Shot Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual Shot Type', fontsize=12, fontweight='bold')
    axes[0].set_title('Shot Type Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Accuracy per class (handle division by zero for classes not in test set)
    class_sums = cm.sum(axis=1)
    class_acc = np.divide(cm.diagonal(), class_sums, 
                          out=np.zeros_like(cm.diagonal(), dtype=float), 
                          where=class_sums!=0)
    axes[1].bar(range(len(class_acc)), class_acc, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Shot Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy per Shot Type', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(class_acc)))
    axes[1].set_xticklabels(shot_type_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved shot type analysis to: {save_path}")
    plt.close()

def visualize_court_distribution(model_folder, save_path, observed_shots, predicted_shots, player_positions):
    """
    Creates a more readable and user-friendly court visualization with heatmaps.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.patch.set_facecolor('white')
    
    scenarios = [
        {'title': 'Original Defense Location', 'desc': "Observed:\n1: shot service\n2: net shot\n3: lob\n\nPredicted:\n4: net shot", 'pred_text': "4: net shot"},
        {'title': 'Modified Defense Location', 'desc': "Observed:\n1: shot service\n2: net shot\n3: lob\n\nPredicted:\n4: clear", 'pred_text': "4: clear"}
    ]

    for i, ax in enumerate(axs.flatten()):
        # Use the proper court drawing function
        draw_badminton_court(ax)
        ax.set_title(scenarios[i]['title'], fontsize=16, fontweight='bold', pad=20)

        # --- Plot Player Position Heatmaps using Gaussian ---
        p_a_x, p_a_y = player_positions[i][0]
        p_b_x, p_b_y = player_positions[i][1]
        
        court_width, court_length = 710, 1340
        x = np.linspace(0, court_width, 100)
        y = np.linspace(0, court_length, 100)
        X, Y = np.meshgrid(x, y)
        
        # Player A Distribution
        mean_a_x, mean_a_y = p_a_x * court_width/10, p_a_y * court_length/6
        sigma_a = 80
        Z_a = np.exp(-((X - mean_a_x)**2 + (Y - mean_a_y)**2) / (2 * sigma_a**2))
        
        # Player B Distribution
        mean_b_x, mean_b_y = p_b_x * court_width/10, p_b_y * court_length/6
        sigma_b = 80
        Z_b = np.exp(-((X - mean_b_x)**2 + (Y - mean_b_y)**2) / (2 * sigma_b**2))

        # Plot contours
        ax.contourf(X, Y, Z_a, levels=10, cmap='Reds', alpha=0.6)
        ax.contourf(X, Y, Z_b, levels=10, cmap='Blues', alpha=0.6)
        
        # Mark player positions
        ax.plot(mean_a_x, mean_a_y, 'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=2, label='Player A next move')
        ax.plot(mean_b_x, mean_b_y, 'b*', markersize=20, markeredgecolor='darkblue', markeredgewidth=2, label='Player B next move')

        # --- Plot Shot Annotations ---
        # Observed shots
        for x_shot, y_shot, label in observed_shots[i]:
            shot_x, shot_y = x_shot * court_width/10, y_shot * court_length/6
            ax.plot(shot_x, shot_y, 'o', markersize=12, markerfacecolor='cyan', markeredgecolor='black', mew=2)
            ax.text(shot_x, shot_y + 30, label, color='black', ha='center', fontsize=11, fontweight='bold')
        
        # Predicted shot
        if predicted_shots[i]:
            pred_x, pred_y, pred_label = predicted_shots[i][0]
            shot_px, shot_py = pred_x * court_width/10, pred_y * court_length/6
            ax.plot(shot_px, shot_py, 's', markersize=12, markerfacecolor='lime', markeredgecolor='black', mew=2)
            ax.text(shot_px, shot_py + 30, pred_label, color='darkgreen', ha='center', fontsize=11, fontweight='bold')

        # --- Add Text Box for Shot Types ---
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
        ax.text(20, court_length - 50, scenarios[i]['desc'], fontsize=9, verticalalignment='top', bbox=props)
        ax.text(court_width - 120, court_length - 50, f"Predicted:\n{scenarios[i]['pred_text']}", fontsize=9, color='darkgreen', verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced court visualization to: {save_path}")
    plt.close(fig)
def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <model_folder> [num_rallies]")
        print("Example: python visualize_results.py ./model/DyMF_2_2025-12-04-21-23-12 3")
        sys.exit(1)
    
    model_folder = sys.argv[1]
    num_rallies = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    if not os.path.exists(model_folder):
        print(f"Error: Model folder not found: {model_folder}")
        sys.exit(1)
    
    print("="*80)
    print("Generating Visualizations for Research Poster")
    print("="*80)
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Trajectory visualization
    print("\n1. Creating trajectory visualizations...")
    visualize_trajectories(model_folder, num_rallies=num_rallies,
                          save_path=os.path.join(output_dir, 'trajectories.png'))
    
    # 2. Shot type analysis
    print("\n2. Creating shot type analysis...")
    create_shot_type_analysis(model_folder,
                             save_path=os.path.join(output_dir, 'shot_type_analysis.png'))
    
    # 3. Performance comparison (if you have multiple models)
    print("\n3. Creating performance comparison...")
    # You can add multiple model folders here
    model_folders = [model_folder]  # Add more if you have them
    create_performance_comparison(model_folders,
                                 save_path=os.path.join(output_dir, 'performance_comparison.png'))
    
    print("\n" + "="*80)
    print("All visualizations saved to:", output_dir)
    print("="*80)

if __name__ == "__main__":
    main()

