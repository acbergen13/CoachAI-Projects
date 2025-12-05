# Training and Evaluation Guide

## Training the Model

### Basic Training Command

```powershell
python train.py --model_type DyMF --encode_length 2
```

### Training Parameters

The `train.py` script accepts many parameters. Here are the most important ones:

#### Required Parameters
- `--model_type`: Model to train (e.g., `DyMF`, `LSTM`, `GCN`, `ShuttleNet`, `rGCN`, `Transformer`)
- `--encode_length`: Length of input sequence (e.g., `2`, `4`, `8`)

#### Common Optional Parameters
- `--seed`: Random seed (default: `22`)
- `--epochs`: Number of training epochs (default: `100`)
- `--lr`: Learning rate (default: `1e-3` or `0.001`)
- `--train_batch_size`: Batch size for training (default: `32`)
- `--hidden_size`: Hidden layer size (default: `16`)
- `--dropout`: Dropout rate (default: `0.1`)

#### Example Training Commands

**Basic training:**
```powershell
python train.py --model_type DyMF --encode_length 2 --seed 1
```

**Training with custom learning rate:**
```powershell
python train.py --model_type DyMF --encode_length 4 --lr 0.0001 --epochs 50
```

**Training with smaller batch size (if you have memory issues):**
```powershell
python train.py --model_type DyMF --encode_length 2 --train_batch_size 16
```

**Training with more epochs:**
```powershell
python train.py --model_type DyMF --encode_length 8 --epochs 200
```

### After Training

When training completes, `train.py` will print:
1. **Model folder path** - This is where your trained model is saved
2. **Training losses** - Total loss, location loss, and type loss

Example output:
```
./model/DyMF_2_2025-12-04-21-10-42
total loss: 1234.56
location loss: 1000.00
type loss: 234.56
```

**Save the model folder path** - you'll need it for evaluation!

## Evaluating the Model

### Basic Evaluation Command

```powershell
python evaluate.py <model_folder_path> <sample_num>
```

### Parameters
- `model_folder_path`: The path printed by `train.py` (e.g., `./model/DyMF_2_2025-12-04-21-10-42`)
- `sample_num`: Number of samples to use for evaluation (typically `10`)

### Example Evaluation Commands

**Evaluate a trained model:**
```powershell
python evaluate.py ./model/DyMF_2_2025-12-04-21-10-42 10
```

**If the model folder has spaces, use quotes:**
```powershell
python evaluate.py "./model/DyMF_2_2025-12-04-21-10-42" 10
```

### Evaluation Output

The evaluation script will print:
- **Total loss**: Combined prediction error
- **Location MSE loss**: Mean Squared Error for location predictions
- **Location MAE loss**: Mean Absolute Error for location predictions  
- **Type loss**: Error in shot type predictions

Lower values = better performance.

## Complete Workflow Example

### Step 1: Train the Model
```powershell
cd "Movement Forecasting"
python train.py --model_type DyMF --encode_length 2 --seed 1 --epochs 100
```

**Output:**
```
./model/DyMF_2_2025-12-04-21-10-42
total loss: 1234.56
location loss: 1000.00
type loss: 234.56
```

### Step 2: Evaluate the Model
```powershell
python evaluate.py ./model/DyMF_2_2025-12-04-21-10-42 10
```

**Output:**
```
total loss: 1234.56
location MSE loss: 1000.00
location MAE loss: 500.00
type loss: 234.56
```

## Using the Experiment Script

For automated training and evaluation of multiple configurations:

```powershell
python run_experiment.py
```

This will:
1. Train models with different `encode_length` values (2, 4, 8)
2. Automatically evaluate each trained model
3. Save all results

## Troubleshooting

### Model folder path not found
- Make sure you're in the `Movement Forecasting` directory
- Check that training completed successfully
- The path is printed at the end of training

### Evaluation fails
- Ensure the model folder path is correct
- Make sure training completed (check for `encoder` and `decoder` files in the model folder)
- Verify the dataset is still available

### Poor results (high losses)
- Try a lower learning rate: `--lr 0.0001`
- Train for more epochs: `--epochs 200`
- Check data preprocessing (coordinates should be normalized)
- Try different `encode_length` values

## Model Files

After training, the model folder contains:
- `encoder` - Encoder model weights
- `decoder` - Decoder model weights
- `args.json` - Training configuration (used by evaluation script)

