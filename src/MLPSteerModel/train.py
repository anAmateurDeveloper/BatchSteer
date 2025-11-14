import torch
torch.manual_seed(42)

from datetime import datetime
today_str = datetime.now().strftime("%m%d")

import pickle
import torch
import matplotlib.pyplot as plt
import os

# from NSS_mlp import SteeringMLP, SteeringMLPDataset, train_steering_mlp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

from .MLP import SteeringMLP, SteeringMLPDataset, train_steering_mlp

# Configuration specifying steering layer indices and their corresponding strengths for each model
steering_config = {
    "llama3.1": [(8, 0.6), (9, 0.6), (10, 0.6), (11, 0.6), (12, 0.4), (13, 0.5), (14, 0.6), (16, 0.6), (18, 0.6), (19, 0.6)],
    "qwen2.5": [(5, 0.6), (6, 0.6), (7, 0.6), (8, 0.6), (9, 0.5), (10, 0.6), (11, 0.5), (12, 0.5), (13, 0.5), (14, 0.3), (15, 0.3), (16, 0.5), (18, 0.5), (19, 0.6)],
    "gemma2": [(6, 0.5), (8, 0.4), (10, 0.6), (11, 0.6), (12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6), (18, 0.6), (22, 0.5)],
}

device = "cuda:6"
base_dir = "data"
save_dir = "data/steering_mlp"

# Loop through each specified model
for model_name in ["llama3.1", "qwen2.5", "gemma2"]:
    print(f"Processing model: {model_name}")
    embeds_dir = f"{base_dir}/embeddings/{model_name}"
    layers_ratio_list = steering_config[model_name]

    # Load benign and borderline preference vectors
    H_benign_train_10000 = torch.load(f"{embeds_dir}/embeds_benign_train.pt", map_location=device)
    H_coconot_pref = torch.load(f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device)
    H_coconot_original = torch.load(f"{embeds_dir}/embeds_coconot_original.pt", map_location=device)

    # Select borderline samples from original coconot set
    indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
    H_benign_train = torch.cat([
        H_benign_train_10000, H_coconot_original[indices_borderline], H_coconot_pref
    ], dim=0).to(device)
    print(f"H_benign_train shape: {H_benign_train.shape}")
    torch.cuda.empty_cache()

    # Load harmful and jailbreak vectors
    H_harmful_train_1000 = torch.load(f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location=device)
    H_jailbreak_train_full = torch.load(f"{embeds_dir}/embeds_jailbreak_train.pt", map_location=device)
    H_jailbreak_additional = torch.load(f"{embeds_dir}/embeds_jailbreak_additional.pt", map_location=device)

    # Increase the number of harmful samples to balance data
    indices = torch.randperm(H_jailbreak_train_full.size(0))[:min(2000, H_jailbreak_train_full.size(0))]
    H_harmful_train = torch.cat([
        H_harmful_train_1000, H_jailbreak_train_full[indices]
    ], dim=0).to(device)
    H_harmful_train_1000 = None
    H_jailbreak_train_full = None
    H_jailbreak_additional = None
    print(f"H_harmful_train shape: {H_harmful_train.shape}")
    torch.cuda.empty_cache()

    # Balance the benign samples: Randomly downsample benign samples to match harmful sample size
    n_harmful = H_harmful_train.shape[0]
    n_benign = H_benign_train.shape[0]

    # Downsample benign samples if too many
    if n_benign > n_harmful * 1.5:
        # Randomly sample benign samples to keep them similar in number to harmful samples
        target_benign_size = int(n_harmful * 1.2)  # Slightly more benign samples allowed
        benign_indices = torch.randperm(n_benign)[:target_benign_size]
        H_benign_train = H_benign_train[benign_indices]
        print(f"Downsampled H_benign_train shape: {H_benign_train.shape}")

    print(f"Final data balance - Benign: {H_benign_train.shape[0]}, Harmful: {H_harmful_train.shape[0]}")
    print(f"Data ratio (Benign:Harmful): {H_benign_train.shape[0]}:{H_harmful_train.shape[0]} = {H_benign_train.shape[0]/H_harmful_train.shape[0]:.2f}:1")

    # Load refusal vectors (used as the steering targets for harmful samples)
    refusal_vectors_path = f"{base_dir}/refusal_vectors/{model_name}/refusal.pkl"
    refusal_vectors = pickle.load(open(refusal_vectors_path, "rb"))
    refusal_vectors = torch.tensor(refusal_vectors, dtype=torch.float32).to(device)
    print(f"Refusal vectors shape: {refusal_vectors.shape}, dtype: {refusal_vectors.dtype}")

    # Get the number of layers and dimensionality from refusal vectors
    num_layers = refusal_vectors.shape[0]
    d_model = refusal_vectors.shape[1]

    for layer, _ in layers_ratio_list:
        print(f"Training MLP for layer {layer}")

        # Prepare input vectors for both benign and harmful samples (at this layer)
        input_vectors = torch.cat([
            H_benign_train[:, layer, :],
            H_harmful_train[:, layer, :]
        ], dim=0)

        # Create targets: zeros for benign, refusal vector for harmful samples
        target_vectors = torch.cat([
            torch.zeros(H_benign_train.shape[0], d_model, device=device),
            refusal_vectors[layer].repeat(H_harmful_train.shape[0], 1)
        ], dim=0)

        # Split dataset into train and validation sets (80% train, 20% validation)
        n_samples = input_vectors.shape[0]
        n_train = int(0.8 * n_samples)
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_input_vectors = input_vectors[train_indices]
        train_target_vectors = target_vectors[train_indices]
        val_input_vectors = input_vectors[val_indices]
        val_target_vectors = target_vectors[val_indices]

        print(f"Layer {layer}: Train samples: {train_input_vectors.shape[0]}, Validation samples: {val_input_vectors.shape[0]}")

        # Train and save the MLP for this layer
        save_path = os.path.join(save_dir, model_name, f"steering_mlp_layer_{layer}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model = train_steering_mlp(
            d_model=d_model,
            input_vectors=train_input_vectors,
            target_vectors=train_target_vectors,
            val_input_vectors=val_input_vectors,
            val_target_vectors=val_target_vectors,
            hidden_dim=d_model,
            dropout_rate=0.1,  # Lower dropout (was previously higher)
            num_epochs=200,    # More training epochs
            batch_size=512,    # Reduce batch size for more frequent updates
            learning_rate=5e-4,  # Smaller learning rate
            weight_decay=1e-3,   # Lower regularization
            noise_std=0.005,     # Lower input noise
            early_stopping_patience=15,  # Higher early stopping patience
            device=device,
            save_path=save_path
        )

        print(f"MLP for layer {layer} saved to {save_path}")
        torch.cuda.empty_cache()

    # Clean up loaded tensors to release GPU memory after each model's training is complete
    H_benign_train = None
    H_harmful_train = None
    torch.cuda.empty_cache()