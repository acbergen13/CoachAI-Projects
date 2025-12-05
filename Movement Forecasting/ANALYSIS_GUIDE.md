# Comprehensive Model Analysis Guide

This guide explains how to create comprehensive analysis visualizations showing DyMF model effectiveness across multiple configurations.

## Quick Start

### Option 1: Analyze Existing Models

If you already have trained models, analyze them directly:

```powershell
# Analyze specific models
python analyze_model_performance.py ./model/DyMF_2_2025-12-04-21-23-12 ./model/DyMF_4_2025-12-04-21-XX-XX

# Or auto-discover all DyMF models
python analyze_model_performance.py --auto
```

### Option 2: Run New Experiments + Analyze

Run multiple experiments and analyze them automatically:

```powershell
python run_comprehensive_analysis.py
```

This will:
1. Train multiple models with different configurations
2. Automatically analyze all results
3. Generate comprehensive visualizations

## What Gets Generated

The analysis script creates several publication-ready visualizations:

### 1. **comprehensive_comparison.png**
- 4-panel comparison showing:
  - Performance metrics by encode length
  - Location MAE distribution (box plots)
  - Shot type loss distribution (box plots)
  - Scatter plot: Location vs Shot Type performance

### 2. **performance_table.png**
- Formatted table with mean ± std for all metrics
- Shows performance across different configurations
- Includes overall statistics

### 3. **statistical_analysis.png**
- Performance trends over encode length
- Normalized performance comparison
- Shows how model improves with longer sequences

### 4. **raw_results.csv**
- CSV file with all raw evaluation results
- Can be imported into Excel/other tools for further analysis

### 5. **performance_summary.csv**
- Summary statistics grouped by configuration
- Mean, std, min, max for all metrics

## Customizing Experiments

Edit `run_comprehensive_analysis.py` to add more configurations:

```python
experiment_configs = [
    {
        'name': 'DyMF_encode2_lr0.0001',
        'model_type': 'DyMF',
        'encode_length': 2,
        'seed': 1,
        'lr': 0.0001,
        'epochs': 150
    },
    # Add more configurations...
]
```

## Analysis Script Details

### `analyze_model_performance.py`

**Usage:**
```powershell
python analyze_model_performance.py <model_folder1> [model_folder2] ...
python analyze_model_performance.py --auto
```

**What it does:**
1. Loads each model
2. Runs evaluation on test set
3. Collects metrics: Location MAE, Location MSE, Type Loss, Total Loss
4. Generates comparative visualizations
5. Creates statistical summaries

**Output:**
- All visualizations saved to `visualizations/` folder
- Raw data saved as CSV files

## For Your Research Poster

### Recommended Layout:

1. **Top Section:**
   - Model architecture diagram
   - Performance table (shows quantitative results)

2. **Middle Section:**
   - Comprehensive comparison (4-panel figure)
   - Statistical analysis (trends and patterns)

3. **Bottom Section:**
   - Individual trajectory visualizations (from `visualize_results.py`)
   - Shot type analysis

### Key Points to Highlight:

1. **Consistency**: Show that model performs well across different configurations
2. **Improvement**: If longer encode_length improves performance, highlight this
3. **Balanced Performance**: Show both location and shot type predictions are good
4. **Statistical Significance**: Use the summary statistics to show robustness

## Example Workflow

```powershell
# Step 1: Train multiple models (or use existing ones)
python train.py --model_type DyMF --encode_length 2 --seed 1 --lr 0.0001 --epochs 150
python train.py --model_type DyMF --encode_length 4 --seed 1 --lr 0.0001 --epochs 150
python train.py --model_type DyMF --encode_length 8 --seed 1 --lr 0.0001 --epochs 150

# Step 2: Analyze all models
python analyze_model_performance.py --auto

# Step 3: Create individual visualizations
python visualize_results.py ./model/DyMF_2_2025-12-04-21-23-12 3
```

## Tips for Poster Presentation

1. **Use the performance table** to show quantitative superiority
2. **Highlight trends** - if performance improves with encode_length, show this clearly
3. **Show consistency** - multiple runs with same config show robustness
4. **Compare metrics** - location MAE vs type loss shows balanced performance
5. **Statistical significance** - use mean ± std to show confidence

## Troubleshooting

### No models found
- Make sure models are in `./model/` directory
- Check that model folders contain `encoder` and `decoder` files

### Evaluation fails
- Ensure test data is available
- Check that model args are valid

### Visualizations look wrong
- Check that you have multiple models with different configurations
- Ensure matplotlib and seaborn are installed

