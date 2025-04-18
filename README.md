'''
# MNIST Classification with MLP/CNN - Ablation Study

## Overview
This implementation contains complete MLP/CNN training pipeline with:
- Multiple optimizers (SGD/MomentumGD)
- Learning rate schedulers (Step/MultiStep/Exponential)
- Data augmentation (rotation/shift/noise)
- Ablation study framework
- Weight visualization utilities

## Pre-trained Weights
Download from Google Drive (replace with your links):
MLP: https://drive.google.com/file/d/1sxr4OXKOzUwEtGE_8MBPyZu4LaRBMdaI/view?usp=drive_link
CNN: https://drive.google.com/file/d/1LAzYK6MEA6HCz4ApYG1Cnyold7fYnVrX/view?usp=drive_link

## Dataset Preparation
1. Create folder structure:
```
dataset/
└── MNIST/
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```
1. No installation required - pure Python implementation

## Usage Examples

# Run MLP ablation study
```python
from ablation_study import AblationStudy

study = AblationStudy(model_type="MLP")
config = {
    "name": "custom_mlp",
    "hidden_layers": [512, 256],
    "activation": "ReLU",
    "optimizer": "MomentGD",
    "learning_rate": 0.01,
    "num_epochs": 30
}
study.run_experiment(config)
```
# Run CNN experiment  
```python
cnn_config = {
    "name": "enhanced_cnn",
    "conv_configs": [
        {'type':'conv', 'in_channels':1, 'out_channels':32, 'kernel_size':3},
        {'type':'pool', 'pool_type':'max', 'kernel_size':2},
        {'type':'conv', 'in_channels':32, 'out_channels':64, 'kernel_size':3}
    ],
    "fc_configs": [(64 * 5 * 5, 128), (128, 10)],
    "augmentation": {"enabled":True, "rotation_prob":0.4}
}
cnn_study = AblationStudy(model_type="CNN")
cnn_study.run_experiment(cnn_config)
```
# Visualize weights
```python
from visualize_weights import visualize_weights
visualize_weights(
    model_path="ablation_results/CNN_20230418/cnn_baseline/best_model.pickle",
    model_type="cnn"
)
```
## Key Components
ablation_study.py - Main experiment controller
```bash
models/
├── mlp.py - MLP model class
├── cnn.py - CNN model with conv/pool layers
ops/
├── layers.py - Core layers (Linear/Conv2D/Pool)
├── optimizer.py - Optimizers (SGD/Momentum)
└── lr_scheduler.py - Learning rate schedulers
```

## Performance
| Model   | Val Acc | Test Acc | Params |
|---------|---------|----------|--------|
| MLP     | 98.1%   | 97.8%    | 1.1M   |
| CNN     | 99.2%   | 98.9%    | 2.7M   |


'''