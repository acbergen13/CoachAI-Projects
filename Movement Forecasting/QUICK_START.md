# Quick Start Guide - Movement Forecasting Experiment

## What You Need

1. ✅ Data files in `data/` folder (you already have `dataset.csv`)
2. ✅ Python environment with dependencies
3. ✅ Run the experiment script

## Step 1: Set Up Environment

### Option A: Using Conda (Recommended if available)

```powershell
cd "Movement Forecasting"
conda env create -f environment.yml
conda activate master_paper
```

**Note**: The environment.yml is for Linux. On Windows, you may need to install packages manually.

### Option B: Manual Installation (If conda fails)

```powershell
pip install torch==1.9.0 pandas numpy scikit-learn tqdm networkx
```

## Step 2: Prepare Your Data

Your dataset needs a `rally_id` column. If it's missing, run:

```powershell
python prepare_data.py
```

This will add the `rally_id` column to your `data/dataset.csv` file.

## Step 3: Run the Experiment

### Simple Run (Default Settings)

```powershell
python run_experiment.py
```

This will:
- Train DyMF model with encode_lengths [2, 4, 8]
- Use seed 1
- Automatically evaluate each trained model

### Customize the Experiment

Edit `run_experiment.py` to change:

```python
MODEL_LIST = ["DyMF"]  # Change to ["DyMF", "LSTM"] to run multiple models
SEEDS = [1]  # Change to [1, 42, 123] for multiple seeds
SEQUENCE_LENGTHS = [2, 4, 8]  # Change encode lengths
```

### Run Single Training

To train just one model:

```powershell
python train.py --model_type DyMF --encode_length 4 --seed 1
```

### Evaluate a Trained Model

```powershell
python evaluate.py ./model/DyMF_4_2024-01-01-12:00:00 10
```

(Replace with your actual model folder path)

## What to Expect

- **Training**: Takes time depending on your data size and hardware
- **Output**: Models saved in `./model/` directory
- **Results**: Evaluation metrics printed to console (MSE, MAE for location, accuracy for shot type)

## Troubleshooting

### "rally_id column not found"
→ Run `python prepare_data.py` first

### "Module not found" errors
→ Install missing packages: `pip install <package_name>`

### CUDA/GPU issues
→ Code automatically uses CPU if GPU unavailable

### Out of memory
→ Reduce batch size: Add `--train_batch_size 16` to train.py command

## Next Steps

After running:
1. Check `./model/` folder for saved models
2. Review console output for evaluation metrics
3. Compare results across different encode_lengths

For more details, see `EXPERIMENT_GUIDE.md`

