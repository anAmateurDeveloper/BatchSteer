import logging
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.cache_utils import Cache
from .MLP import SteeringMLP

from typing import Optional, Tuple, Union, List, Dict

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


class MLPSteerLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, 
                 layer_idx: int, 
                 steering_mlp_path: Optional[str] = None, 
                 strength: float = 0.0
                 ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        # self.steering_vector = None
        
        device = next(self.parameters()).device
        self.steering_mlp = None
        if steering_mlp_path is not None and os.path.exists(steering_mlp_path):
            self.steering_mlp = SteeringMLP(self.d_model)
            state_dict = torch.load(steering_mlp_path, map_location=device)
            self.steering_mlp.load_state_dict(state_dict)
            # Convert to bfloat16 to match the main model's dtype
            self.steering_mlp = self.steering_mlp.to(device).to(torch.bfloat16)
            
        self.strength = strength
        
    def set_steering_parameters(
        self, 
        steering_mlp_path: Optional[str]=None, 
        strength: float = 0.0,
        device: Optional[torch.device]=None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_mlp_path is not None and os.path.exists(steering_mlp_path):
            try:
                self.steering_mlp = SteeringMLP(self.d_model)
                state_dict = torch.load(steering_mlp_path, map_location=device)
                self.steering_mlp.load_state_dict(state_dict)
                # Convert to bfloat16 to match the main model's dtype
                self.steering_mlp = self.steering_mlp.to(device).to(torch.bfloat16)
                print(f"self.steering_mlp.state_dict().keys(): {self.steering_mlp.state_dict().keys()}")

            except Exception as e:
                logger.error(f"Error loading steering MLP from {steering_mlp_path}: {str(e)}")
                raise
            
        self.strength = strength if strength is not None else 0.0
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Ensure steering_matrix is on the same device as hidden_states
        if hidden_states.shape[1] > 1: # Only apply steering on initial input
            
            if self.steering_mlp is not None:
                # Only apply steering once during input processing
                # Calculate steering vector by multiplying the last token's hidden state with the steering matrix
                steering_vector = self.steering_mlp(hidden_states[:, -1, :]) * self.strength
                # Reshape to match hidden_states dimensions and move to the same device
                steering_vector = steering_vector.unsqueeze(1).to(hidden_states.device)
                # Apply steering by adding the steering vector to hidden states
                hidden_states = hidden_states + steering_vector
                
        residual = hidden_states # resid_pre - save for residual connection

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states # resid_mid - save after attention residual
        
        # Normalize hidden states after attention, then pass through MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # resid_post - final residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs



class MLPSteerLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Replace the layers with our custom MLPSteer layers, keeping everything else unchanged
        self.layers = nn.ModuleList(
            [MLPSteerLlamaDecoderLayer(
                config=config, 
                layer_idx=layer_idx,
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self, 
        steering_mlp_paths: Optional[list[str]]=None, 
        strength: Optional[list[float]] = None,
        device: Optional[torch.device] = None):
        device = next(self.parameters()).device if device is None else device
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_mlp_path = None
            layer_strength = 0.0
            if steering_mlp_paths is not None and layer_idx < len(steering_mlp_paths):
                layer_steering_mlp_path = steering_mlp_paths[layer_idx]

            if strength is not None and layer_idx < len(strength):
                layer_strength = strength[layer_idx]
            
            layer.set_steering_parameters(
                steering_mlp_path=layer_steering_mlp_path, 
                strength=layer_strength,
                device=device
            )
            torch.cuda.empty_cache()
        
        # self.print_steering_parameters()
        
    def print_steering_parameters(self):
        logger.info("Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'Steering MLP'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            strength_val = str(layer.strength)
            steering_mlp_str = "Present" if layer.steering_mlp is not None else "None"
            logger.info(f"{layer_idx:<10}{strength_val:<20}{steering_mlp_str}")
    


class MLPSteerLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = MLPSteerLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, 
                        steering_mlp_paths: Optional[list[str]] = None,
                        strength: Optional[list[float]] = None,
                        **kwargs):
        # Call the parent class's from_pretrained method to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_mlp_paths=steering_mlp_paths, strength=strength)
        return model

    def set_steering_parameters(
            self, 
            steering_mlp_paths: Optional[list[str]]=None, 
            strength: Optional[list[float]] = None):
        
        device = next(self.parameters()).device
        
        if steering_mlp_paths is not None:
            print(f"steering_mlp_paths: {steering_mlp_paths}")
            print("len(steering_mlp_paths): ", len(steering_mlp_paths))
            print(f"strength: {strength}")

        self.model.set_steering_parameters(
            steering_mlp_paths=steering_mlp_paths, 
            strength=strength,
            device=device
        )