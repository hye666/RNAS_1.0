## Introduction
We propose Robust Network Architecture Search (RNAS) based on Differentiable Architecture Search (DARTS). RNAS searches for a robust network architecture through restraining the latent feature distortion. 
paper title: Robust Network Architecture Search via Feature Distortion Restraining

## Main requirements
torch
torchvision
torchattacks

## Usage
#### Search on CIFAR10

```
python search.py
```

#### RNAS training and evaluation on CIFAR10

```
python RNAS_training.py
```
#### Other model training on CIFAR10

```
python other_models_training.py 
```

#### Other model evaluation on CIFAR10
```
python evaluation.py
```


