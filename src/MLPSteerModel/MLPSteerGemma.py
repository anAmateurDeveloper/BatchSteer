import logging
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import Gemma2ForCausalLM, Gemma2Model, Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.cache_utils import Cache
from .MLP import SteeringMLP

from typing import Optional, Tuple, Union, List, Dict
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


class MLPSteerGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, 
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
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -effective_seq_len:]
            # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
            # from the left, with an offset if we are beyond the sliding window
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                # In case we are beyond the sliding window, we need to correctly offset the mask slicing
                # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo
                offset = last_cache_position - effective_seq_len
                # Should only be used when beyond the sliding window (i.e. offset > 0)
                offset = max(0, offset)
                attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]

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
                
                # self.steering_vector = hidden_states[:, -1, :] @ self.steering_matrix * self.strength
                # self.steering_vector = self.steering_vector.unsqueeze(1).to(hidden_states.device) # Same dimensions as hidden_states
        # if self.steering_vector is not None:
        #     if self.steering_vector.device != hidden_states.device:
        #         self.steering_vector = self.steering_vector.to(hidden_states.device)
            
        #     hidden_states = hidden_states + self.steering_vector
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
    
class MLPSteerGemma2Model(Gemma2Model):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        # Replace the layers with our custom MLPSteer layers, keeping everything else unchanged
        self.layers = nn.ModuleList(
            [MLPSteerGemma2DecoderLayer(
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


class MLPSteerGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.model = MLPSteerGemma2Model(config=config)

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