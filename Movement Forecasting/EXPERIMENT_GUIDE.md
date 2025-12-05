# Movement Forecasting Experiment Guide

This guide will help you run the play forecasting experiment on Windows.

## Prerequisites

1. **Conda Environment**: You need to set up the conda environment first
2. **Data**: Ensure your data files are in the `data/` folder
3. **Python**: Python 3.7 (as specified in environment.yml)

## Quick Start (Windows)

1. **Activate conda environment** (or install dependencies):
   ```powershell
   conda activate master_paper
   # OR if conda fails, install manually:
   pip install torch pandas numpy scikit-learn tqdm networkx
   ```

2. **Prepare dataset** (if rally_id is missing):
   ```powershell
   python prepare_data.py
   ```

3. **Run experiment**:
   ```powershell
   python run_experiment.py
   ```

## Step-by-Step Setup

### 1. Set Up Conda Environment

Open PowerShell or Command Prompt and navigate to the `Movement Forecasting` directory:

```powershell
cd "Movement Forecasting"
```

Create the conda environment:

```powershell
conda env create -f environment.yml
```

Activate the environment:

```powershell
conda activate master_paper
```

**Note**: The environment.yml file is configured for Linux. On Windows, you may need to:
- Install PyTorch separately if conda installation fails
- Adjust package versions if needed

If you encounter issues with the conda environment, you can install dependencies manually:

```powershell
pip install torch==1.9.0 pandas numpy scikit-learn tqdm
```

### 2. Verify Data

Ensure you have the preprocessed dataset in `data/dataset.csv`. The script expects this file by default.

The dataset should contain columns:
- `rally_id`
- `player`
- `type`
- `player_location_x`
- `player_location_y`
- `opponent_location_x`
- `opponent_location_y`
- `ball_round`
- `set`
- `match_id`

### 3. Run the Experiment

#### Option A: Using the Python Script (Recommended for Windows)

```powershell
python run_experiment.py
```

#### Option B: Run Individual Training and Evaluation

Train a model:
```powershell
python train.py --model_type DyMF --encode_length 2 --seed 1
```

Evaluate a model (replace with your model path):
```powershell
python evaluate.py ./model/DyMF_2_2024-01-01-12:00:00 10
```

#### Option C: Using Git Bash or WSL (if available)

If you have Git Bash or WSL installed, you can use the original bash script:

```bash
bash experiment.sh
```

## Configuration

### Modify Experiment Parameters

Edit `run_experiment.py` to change:

- **Models**: Modify `MODEL_LIST` (options: "DyMF", "LSTM", "GCN", "ShuttleNet", "rGCN", "Transformer")
- **Seeds**: Modify `SEEDS` list
- **Sequence Lengths**: Modify `SEQUENCE_LENGTHS` (encode_length values)

Example:
```python
MODEL_LIST = ["DyMF", "LSTM"]
SEEDS = [1, 42, 123]
SEQUENCE_LENGTHS = [2, 4, 8, 16]
```

### Training Parameters

You can modify training parameters in `train.py` or pass them as command-line arguments:

- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--train_batch_size`: Batch size for training (default: 32)
- `--hidden_size`: Hidden layer size (default: 16)

Example:
```powershell
python train.py --model_type DyMF --encode_length 4 --seed 1 --epochs 50 --lr 0.001
```

## Troubleshooting

### Issue: Conda environment creation fails

**Solution**: Install dependencies manually with pip:
```powershell
pip install torch pandas numpy scikit-learn tqdm networkx
```

### Issue: CUDA/GPU not available

The code will automatically fall back to CPU if CUDA is not available. To force CPU usage, you can modify the device selection in `train.py`.

### Issue: Data format errors

Ensure your `dataset.csv` has the required columns. If you need to preprocess raw data, set `--already_have_data 0` in train.py and provide:
- `match.csv` (list of matches)
- `homography.csv` (homography matrices)
- Individual match CSV files in subdirectories

### Issue: Out of memory

Reduce batch size:
```powershell
python train.py --model_type DyMF --encode_length 2 --train_batch_size 16
```

## Output

- **Models**: Saved in `./model/` directory with format: `{model_type}_{encode_length}_{timestamp}`
- **Training logs**: Printed to console
- **Evaluation results**: Printed to console after each evaluation

## Model Types Available

- **DyMF**: Dynamic Graphs and Hierarchical Fusion (main model)
- **LSTM**: LSTM-based baseline
- **GCN**: Graph Convolutional Network
- **ShuttleNet**: Position-aware fusion model
- **rGCN**: Relational Graph Convolutional Network
- **Transformer**: Transformer-based model

## Next Steps

After running the experiment:
1. Check the model outputs in `./model/` directory
2. Review evaluation metrics (MSE, MAE for location, accuracy for shot type)
3. Compare results across different encode_lengths and models

