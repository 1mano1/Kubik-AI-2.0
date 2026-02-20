# ![KubikAI Logo](KubikAI/logo/KubikAI-Logo.png) Kubik AI 2.0

Kubik AI 2.0 is an advanced 3D generation AI designed for high-fidelity geometry and texture synthesis. This project is a complete evolution, utilizing a new architecture based on Signed Distance Functions (SDF) and Cross-Attention Flow models to achieve superior results.

## Key Features

- **SDF-VAE Architecture:** Uses Signed Distance Functions for sharp, precise 3D geometries with a 16x16x16 latent grid.
- **Cross-Attention Flow Model:** Implements advanced attention mechanisms for high-fidelity detail synthesis from input images, optimized with DINOv2.
- **Independent Pipeline:** Fully autonomous data processing and training pipeline.
- **Portable & Robust:** Optimized for high-performance environments like Kaggle with reduced memory footprint.

## Project Structure

- `KubikAI/`: Core package containing models, datasets, and trainers.
  - `models/`: SDF-VAE and Cross-Attention Flow implementations.
  - `datasets/`: SDF and Latent dataset handlers.
  - `trainers/`: Specialized training logic for each model stage.
  - `configs/`: Training and model configurations.
  - `logo/`: Project branding assets.
- `train_vae.py`: Entry point for training the SDF-VAE.
- `train_flow.py`: Entry point for training the Flow model.
- `preprocess_data.py`: Multi-view rendering and SDF generation pipeline.
- `requirements.txt`: Project dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Pre-process Data:**
   Use `preprocess_data.py` to prepare your 3D models.
2. **Train VAE:**
   Run `train_vae.py` with the appropriate config.
3. **Train Flow Model:**
   Run `train_flow.py` to train the generation stage.

## Credits and Acknowledgments

Kubik AI 2.0 is a technical evolution that has been made possible thanks to the pioneering research of the **TRELLIS** team and their collaborators. This project is built upon the original concepts and architecture developed by them, extending their capabilities to reach new levels of fidelity in 3D generation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
