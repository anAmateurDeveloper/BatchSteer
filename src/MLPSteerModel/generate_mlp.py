import os
# The following environment settings are commented out, adjust as needed:
# os.environ["PYTHONPATH"] = "/NAS/zhangyi/shenc/NullSpaceSteering/scripts"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Set cache size for torch._dynamo
os.environ["TORCH_COMPILE_CACHE_SIZE"] = "128"  # Increase to 128
torch._dynamo.config.cache_size_limit = 128
# Enable verbose recompilation logging
os.environ["TORCH_LOGS"] = "recompiles"

import argparse
import yaml
import logging
import time

import numpy as np
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

from transformers import AutoTokenizer, LlamaConfig
from . import MLPSteerLlamaForCausalLM, MLPSteerQwen2ForCausalLM, MLPSteerGemma2ForCausalLM
import json

from jinja2 import Template

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

template_jinja = """\
Please solve this problem, and put your final answer within \\boxed{}
This is the problem:
{{prompt}}
Please remember to put your final answer within \\boxed{}
"""

def load_config(config_path):
    '''
    Expected arguments in config:
        device: device to use
        model_name: model name
        model_path: path to model
        input_file: path to input file
        output_file: path to output file
        batch_size: batch size
        max_new_tokens: maximum number of tokens to generate
        prompt_column: name of prompt column
        response_column: name of response column
        strength: strength values as comma-separated list, e.g.: 0.0,0.1,0.2
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file")
    return parser.parse_args()

# Only the layer indices for each model, in list format
STEERING_LAYERS = {
    "llama3.1": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen2.5": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19], 
    "gemma2": [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
}

if __name__ == "__main__":
    execute_start_time = time.time()
    args = parse_args()
    config = load_config(args.config_path)
    
    # Assign config values to args object for convenience
    for key, value in config.items():
        setattr(args, key, value)
    logger.info(f"args: {args}")
    
    config = LlamaConfig.from_pretrained(args.model_path)
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    steering_mlp_dir = f"data/steering_mlp/{args.model_name}"
    steering_layers = STEERING_LAYERS[args.model_name]
    steering_mlp_paths = [""] * num_layers
    for layer in steering_layers:
        this_path = os.path.join(
            steering_mlp_dir, f"steering_mlp_layer_{layer}.pth")
        if os.path.exists(this_path):
            steering_mlp_paths[layer] = this_path
        else:
            logger.info(f"steering_mlp_path not found: {this_path}")
            steering_mlp_paths[layer] = ""
    
    logger.info(f"steering_mlp_paths:\n{json.dumps(steering_mlp_paths, indent=4, ensure_ascii=False)}")
    
    logger.info(f"device: {args.device}")

    if "llama" in args.model_name.lower():
        model_class = MLPSteerLlamaForCausalLM
    elif "qwen" in args.model_name.lower():
        model_class = MLPSteerQwen2ForCausalLM
    elif "gemma" in args.model_name.lower():
        model_class = MLPSteerGemma2ForCausalLM
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    model = model_class.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=torch.bfloat16
    )
    
    print("len(steering_mlp_paths): ", len(steering_mlp_paths))
    model.set_steering_parameters(
        steering_mlp_paths=steering_mlp_paths
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    
    # Load input prompts
    with open(args.input_file, "r") as f:
        prompts = json.load(f)
    
    # Handle output file naming and collision
    if hasattr(args, "file_rename") and args.file_rename:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = args.output_file.replace(".json", f"_{timestamp}.json") if hasattr(
            args, "output_file") else args.input_file.replace(".json", f"_{timestamp}.json")
    else:
        output_file = args.output_file
        if os.path.exists(output_file):
            logger.info(f"output_file already exists: {output_file}, load from it")
            with open(output_file, "r") as f:
                prompts = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"len(prompts):\t{len(prompts)}")
    
    # Prepare prompts for generation
    if "gcg" in args.input_file:
        logger.info("gcg, use original prompt directly")
        formatted_prompts = [prompt[args.prompt_column] for prompt in prompts]
    else:
        logger.info("not gcg, use template")
        if "gsm8k" in args.input_file or "math" in args.input_file:
            template = Template(template_jinja)
            messages = [{"role": "user", "content": template.render(prompt=prompt[args.prompt_column])} for prompt in prompts]
        else:
            messages = [{"role": "user", "content": prompt[args.prompt_column]} for prompt in prompts]

        formatted_prompts = [
            tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
    
    total_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size
    
    # Parse strength values, fallback to default if not provided
    const_strength_list = [float(s) for s in str(args.strength).split(",")] if hasattr(
        args, "strength") else [0.0, -0.1, -0.15, -0.2, -0.25, -0.3, -0.4]
    
    logger.info(f"const_strength_list: {const_strength_list}")
    completed_strengths = []
    try:
        for const_strength in const_strength_list:
            # Skip strengths already generated
            if prompts[-1].get(f"response_strength:{const_strength}") is not None:
                logger.info(f"skip {const_strength} because it has already been generated")
                continue
            
            # Set strength for each layer used by this model
            strength = [0.0] * num_layers
            for layer in steering_layers:
                strength[layer] = const_strength
            logger.info(f"strength: {strength}")
            model.eval()
            model.set_steering_parameters(
                steering_mlp_paths=steering_mlp_paths, 
                strength=strength)

            # Find the index to start, skipping prompts already processed for this strength
            for idx in range(0, len(formatted_prompts)):
                if prompts[idx].get(f"response_strength:{const_strength}") is None:
                    start_idx = idx
                    logger.info(f"find start_idx: {start_idx}")
                    break
            
            # Batch generation loop
            for i in range(start_idx, len(formatted_prompts), args.batch_size):
                batch_idx = i // args.batch_size + 1
                logger.info(f"processing batch {batch_idx}")
                
                batch_prompts = formatted_prompts[i:i + args.batch_size]
                batch_inputs = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(args.device)

                input_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
                batch_input_ids = batch_inputs["input_ids"]
                batch_attention_mask = batch_inputs["attention_mask"]

                # Generate responses with model
                with torch.no_grad():
                    start_time = time.time()
                    batch_outputs = model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=1,
                        do_sample=False,
                        temperature=0.0,
                    )
                    end_time = time.time()
                
                # Process generated outputs
                for j, output in enumerate(batch_outputs):
                    generated_part = output[input_lengths[j]:]
                    response = tokenizer.decode(generated_part, skip_special_tokens=True)
                    prompts[i + j][f"response_strength:{const_strength}"] = response

                # Release GPU memory for this batch
                del batch_outputs
                torch.cuda.empty_cache()

                time_per_example = (end_time - start_time) / min(args.batch_size, len(formatted_prompts) - i)
                logger.info(f"Processed batch {batch_idx}/{total_batches}, time taken: {end_time - start_time:.2f} seconds, time per example: {time_per_example:.2f} seconds")
            completed_strengths.append(const_strength)
            
            # Write results to output file after processing this strength
            with open(output_file, "w") as f:
                json.dump(prompts, f, indent=4)
            logger.info(f"this batch saved to {output_file}")
        
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"An error occurred: {e}")
        logger.info({
            "completed_strengths": completed_strengths,
            "output_file": output_file,
        })
        # Save current results to output_file when error occurs or keyboard interrupts
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=4)
        logger.info(f"progress saved to {output_file}")
        raise e
    execute_end_time = time.time()
    logger.info(f"execute time: {execute_end_time - execute_start_time:.2f} seconds")
