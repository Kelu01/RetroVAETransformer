# VAE-Transformer Retrosynthesis

A PyTorch-based **VAE-Transformer** model for predicting reactants from products in chemical reactions. Includes SMILES tokenization, latent-space encoding, and autoregressive decoding. Supports sampling, checkpointing, and experiment tracking with WandB.

---

## Features

- **VAE-Transformer architecture** with continuous latent space.
- **Autoregressive sampling** with argmax decoding for retrosynthesis.
- **SMILES tokenization** with halogen simplification ([Cl] → L, [Br] → R) and restoration.
- **WandB integration** for training logs, learning rate, and model checkpoints.
- **Checkpoint loading** for fine-tuning or inference.
- Supports **long SMILES sequences** (up to 1300 tokens).

---

## Note

The model and the dataset have been compressed for easier upload.
- **Model:** `Retro-4.zip` (contains `Retro-4.ckpt`)  
- **Dataset:** `uspto.zip` (contains `uspto.csv`)   
