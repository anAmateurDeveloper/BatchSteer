import os
import logging
import datetime

import torch.nn as nn
import torch

from transformers import Gemma2ForCausalLM, Gemma2Model, Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
# Add these imports
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from utils.mask_utils import get_last_valid_token_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Only CausalLM can be called from outside
__all__ = [
    'BatchSteer2ForCausalLM'
    ]


class BatchGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, 
                 layer_idx: int, 
                 steering_matrix: Optional[torch.Tensor] = None, 
                 strength: float = 0.0
                 ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        # self.steering_vector = None
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            self.steering_matrix = steering_matrix.to(device)
        else:
            self.steering_matrix = None
        self.strength = strength
        
    def set_steering_parameters(
        self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: float = 0.0,
        device: Optional[torch.device]=None):
        
    if device is None:
        device = next(self.parameters()).device

    num_layers = len(self.layers)

    # Initialize zero strength for all layers
    layer_strengths = [0.0] * num_layers

    # If steer_layers not provided → default to no steering unless strengths already layer-wise
    if steer_layers is None:
        steer_layers = []

    # Normalize strength format
    if isinstance(strength, (int, float)):
        # Single strength for all chosen layers
        for l in steer_layers:
            if 0 <= l < num_layers:
                layer_strengths[l] = float(strength)
    elif isinstance(strength, list):
        # List must match steer_layers length
        assert len(strength) == len(steer_layers), \
            "Length of strength list must match steer_layers list."
        for idx, l in enumerate(steer_layers):
            if 0 <= l < num_layers:
                layer_strengths[l] = float(strength[idx])
    else:
        # No valid strength provided → keep all zeros
        pass

    # Assign steering to layers
    for layer_idx, layer in enumerate(self.layers):
        layer_matrix = None
        if steering_matrix is not None:
            layer_matrix = steering_matrix.to(device)

        layer.set_steering_parameters(
            steering_matrix=layer_matrix,
            strength=layer_strengths[layer_idx],
            device=device
        )

    torch.cuda.empty_cache()
    self.print_steering_parameters()

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

        should_apply_steering = (
            hidden_states.shape[1] > 1
            and self.steering_matrix is not None
            and torch.any(self.steering_matrix)
            and self.strength != 0.0
        )

        if should_apply_steering:
            if self.steering_matrix.device != hidden_states.device:
                self.steering_matrix = self.steering_matrix.to(hidden_states.device)

            B, T, _ = hidden_states.shape
            device = hidden_states.device

            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )

            batch_idx = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_idx, last_idx, :]
            steering_vector = last_hidden @ self.steering_matrix * self.strength
            steering_vector = steering_vector.unsqueeze(1)
            hidden_states = hidden_states + steering_vector
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
    
    
class BatchGemma2Model(Gemma2Model):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        # Replace the layers with our custom Batch layers, keeping everything else unchanged
        self.layers = nn.ModuleList(
            [BatchGemma2DecoderLayer(
                config=config, 
                layer_idx=layer_idx, 
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(self, 
                                steering_matrix: Optional[torch.Tensor]=None, 
                                strength: Optional[list[float]] = None,
                                device: Optional[torch.device] = None):
        # Get the current device of the model
        if device is None:
            device = next(self.parameters()).device
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx].to(device)
                
            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix, 
                strength=strength[layer_idx] if strength is not None else 0.0
            )
            torch.cuda.empty_cache()
        self.print_steering_parameters()
        
    def print_steering_parameters(self):
        logger.info("Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'Steering Matrix (First Element)'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            # Ensure strength is a string or formattable type
            strength_val = str(layer.strength)
            
            if layer.steering_matrix is not None:
                steering_matrix_str = layer.steering_matrix[0, 0]
            else:
                steering_matrix_str = "None"
            
            logger.info(f"{layer_idx:<10}{strength_val:<20}{steering_matrix_str}")


class BatchGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.model = BatchGemma2Model(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, 
                        steering_matrix: Optional[torch.Tensor]=None,
                        strength: Optional[list[float]]=None,
                        **kwargs):
        # Call the parent class's from_pretrained method to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return model

    def set_steering_parameters(
            self, 
            steering_matrix: Optional[torch.Tensor]=None, 
            strength: Optional[list[float]] = None):
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
            
        self.model.set_steering_parameters(
            steering_matrix=steering_matrix, 
            strength=strength,
            device=device
        )