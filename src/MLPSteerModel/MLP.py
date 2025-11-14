import logging
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple, Union, List, Dict
from transformers.cache_utils import Cache
# from transformers.models.llama.modeling_llama import FlashAttentionKwargs

# For Python 3.10 compatibility
try:
    from typing import Unpack
except ImportError:
    # Fallback for Python < 3.11
    Unpack = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SteeringMLPDataset(Dataset):
    """Dataset for training SteeringMLP."""
    def __init__(self, input_vectors: torch.Tensor, target_vectors: torch.Tensor, noise_std: float = 0.0):
        """
        Initialize the dataset.

        Args:
            input_vectors (torch.Tensor): Input hidden states of shape (n_samples, d_model).
            target_vectors (torch.Tensor): Target steering vectors of shape (n_samples, d_model).
            noise_std (float, optional): Standard deviation of Gaussian noise for data augmentation. Defaults to 0.0.
        """
        if input_vectors.shape != target_vectors.shape:
            raise ValueError(f"Input and target vectors must have the same shape, got {input_vectors.shape} and {target_vectors.shape}")
        if input_vectors.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {input_vectors.dim()}D")
        
        self.input_vectors = input_vectors
        self.target_vectors = target_vectors
        self.noise_std = noise_std
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.input_vectors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index, optionally adding Gaussian noise to inputs."""
        input_vec = self.input_vectors[idx]
        if self.noise_std > 0:
            noise = torch.normal(mean=0.0, std=self.noise_std, size=input_vec.shape, device=input_vec.device)
            input_vec = input_vec + noise
        return input_vec, self.target_vectors[idx]

class SteeringMLP(nn.Module):
    """A multi-layer perceptron for steering hidden states in transformer models."""
    def __init__(self, d_model: int, hidden_dim: int = None, dropout_rate: float = 0.2):
        """
        Initialize the SteeringMLP.

        Args:
            d_model (int): Dimension of the input and output vectors.
            hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to d_model.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model if hidden_dim is None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.mlp(x)

def train_steering_mlp(
    d_model: int,
    input_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    val_input_vectors: Optional[torch.Tensor] = None,
    val_target_vectors: Optional[torch.Tensor] = None,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = 0.2,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    noise_std: float = 0.01,
    early_stopping_patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'steering_mlp.pth'
) -> SteeringMLP:
    """
    Train a SteeringMLP model with regularization and save it to save_path.

    Args:
        d_model (int): Dimension of the input and output vectors.
        input_vectors (torch.Tensor): Input hidden states of shape (n_samples, d_model).
        target_vectors (torch.Tensor): Target steering vectors of shape (n_samples, d_model).
        val_input_vectors (torch.Tensor, optional): Validation input vectors.
        val_target_vectors (torch.Tensor, optional): Validation target vectors.
        hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to d_model.
        dropout_rate (float): Dropout rate for regularization.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization strength.
        noise_std (float): Standard deviation of Gaussian noise for data augmentation.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        device (str): Device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the trained model.

    Returns:
        SteeringMLP: Trained model.
    """
    # Initialize model
    model = SteeringMLP(d_model, hidden_dim, dropout_rate).to(device)
    logger.info(f"Initialized SteeringMLP with d_model={d_model}, hidden_dim={model.hidden_dim}, dropout_rate={dropout_rate}")

    # Create datasets and dataloaders
    train_dataset = SteeringMLPDataset(input_vectors, target_vectors, noise_std=noise_std)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataloader = None
    if val_input_vectors is not None and val_target_vectors is not None:
        val_dataset = SteeringMLPDataset(val_input_vectors, val_target_vectors, noise_std=0.0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for batch_input, batch_target in train_dataloader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_input, val_target in val_dataloader:
                    val_input, val_target = val_input.to(device), val_target.to(device)
                    output = model(val_input)
                    val_loss = criterion(output, val_target)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            model.train()
    
    # Save the best model
    try:
        if best_model_state is not None:
            torch.save(best_model_state, save_path)
        else:
            torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {str(e)}")
        raise
    
    # Load the best model state if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
