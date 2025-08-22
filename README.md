# Generative Adversarial Network (GAN) on CIFAR-10 Dataset

This project implements and compares two different Generative Adversarial Network architectures for image generation on the CIFAR-10 dataset, specifically focusing on the "ship" class (label 8).

## Project Overview

The project consists of two main implementations:
- **First Level**: Vanilla GAN implementation
- **Second Level**: Progressive GAN (ProGAN) mechanism implementation

Both implementations were developed and tested on Google Colab using TensorFlow/Keras.

## Dataset

- **Dataset**: CIFAR-10
- **Target Class**: Ships (label 8)
- **Image Size**: 32x32 pixels, RGB
- **Training Images**: ~5,000 ship images (from 50,000 total training images)
- **Data Preprocessing**: Normalized to [-1, 1] range

## Architecture Details

### Generator Architecture
- **Input**: 128-dimensional latent space
- **Architecture**: 
  - Dense layer: 128 → 4×4×256
  - Transposed convolutions with progressive upsampling:
    - 4×4×256 → 8×8×128
    - 8×8×128 → 16×16×64  
    - 16×16×64 → 32×32×32
  - Output: 32×32×3 (RGB images)
- **Activation**: LeakyReLU (α=0.2) for hidden layers, tanh for output

### Discriminator Architecture
- **Input**: 32×32×3 RGB images
- **Architecture**:
  - Convolutional layers with progressive downsampling:
    - 32×32×3 → 32×32×32
    - 32×32×32 → 16×16×64
    - 16×16×64 → 8×8×128
    - 8×8×128 → 4×4×256
  - Flatten + Dense layer
- **Output**: Binary classification (real/fake)
- **Activation**: LeakyReLU (α=0.2) for hidden layers, sigmoid for output

## Training Configuration

- **Latent Dimension**: 128
- **Training Epochs**: 250
- **Batch Size**: 64
- **Optimizer**: Adam (learning_rate=0.0002, β₁=0.5)
- **Loss Function**: Binary Cross-Entropy

## Key Features

### Training Process
- Alternating training of Generator and Discriminator
- Real samples: Label 1, Fake samples: Label 0
- Loss tracking for both networks
- Periodic image generation for visualization

### Visualization
- Training loss plots for Generator and Discriminator
- Generated image grids (10×10) every 10 epochs
- Final model evaluation with 25 sample images
- Model saving and loading capabilities

### Results
The models successfully generate ship-like images after training, demonstrating:
- Progressive improvement in image quality over epochs
- Stable training dynamics between Generator and Discriminator
- Realistic ship features and structures

## Files Structure

```
├── First_Level.ipynb          # Vanilla GAN implementation
├── Second_Level.ipynb         # ProGAN mechanism implementation  
├── CycleGAN_paper.pdf         # Reference paper
├── ProGAN_paper.pdf          # Reference paper
├── generative_adversarial_network_paper.pdf  # Reference paper
└── README.md                 # This file
```

## Usage

1. Open either notebook in Google Colab
2. Install required dependencies: `tensorflow_datasets`
3. Run all cells sequentially
4. Monitor training progress and generated images
5. Save trained models for later use

## Dependencies

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- TensorFlow Datasets
- scikit-learn

## Results Summary

Both implementations successfully generate ship images from the CIFAR-10 dataset, with the ProGAN mechanism showing improved training stability and image quality progression. The models demonstrate the effectiveness of GANs for conditional image generation tasks.

## References

- Original GAN Paper: `generative_adversarial_network_paper.pdf`
- ProGAN Paper: `ProGAN_paper.pdf`
- CycleGAN Paper: `CycleGAN_paper.pdf`
