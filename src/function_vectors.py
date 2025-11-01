from .component import Component
from tqdm.auto import tqdm
from .utils import tokenize, evaluate
import torch
from functools import partial
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Sequence, List, Tuple    
from tqdm.auto import tqdm


def generate_head_activations_last_token(
    model: HookedTransformer, 
    dataset: Sequence[str], 
    batch_size: int = 32
) -> List[torch.Tensor]:
    """
    Collect per-head activations at the final token position for each prompt batch.

    Args:
        model: HookedTransformer instance of the model.
        dataset: Sequence of prompt strings to encode.
        batch_size: Number of prompts processed per forward pass.

    Returns:
        List of tensors with cached activations for every layer’s attention heads.
    """
    components = [Component('z', layer=i) for i in range(model.cfg.n_layers)]
    activations = [[] for _ in range(len(components))]
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        inputs = model.to_tokens(batch).to(model.cfg.device)
        _, activations_batch= model.run_with_cache(inputs)
        for j in range(len(components)):
            activations[j].append(activations_batch[components[j].valid_hook_name()][:, -1].cpu())
    activations = [torch.cat(activation, dim=0) for activation in activations] # Shape of each activation: (batch_size, num_heads, head_dim)
    return activations


def compute_head_impact(
    model: HookedTransformer, 
    tokenizer: PreTrainedTokenizerBase, 
    dataset: Sequence[Tuple[str, int]], 
    head_activations: List[torch.Tensor], 
    batch_size: int = 50, 
    prepend_bos: bool = True
):
    """
    Estimate per-head impact on answer probabilities by injecting averaged activations.

    Args:
        model: HookedTransformer instance of the model.
        tokenizer: Model tokenizer.
        dataset: Sequence of (prompt, expected_answer) pairs.
        head_activations: Cached head activations aligned with the dataset prompts.
        batch_size: Number of instances processed per forward pass.
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        Tensor of shape (num_layers, num_heads) containing average probability deltas per head.
    """
    model.reset_hooks()
    components = [Component('z', layer=i) for i in range(model.cfg.n_layers)]
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    head_activations_final_token = head_activations
    head_importances = torch.zeros((num_layers, num_heads))
    for layer in tqdm(range(num_layers)):
        head_activations_final_token_layer = head_activations_final_token[layer].mean(dim=0)
        component_to_inject = components[layer]
        for head in range(num_heads):
            all_probs = 0
            for i in range(0, len(dataset), batch_size):
                model.reset_hooks()
                batch = dataset[i:i + batch_size]
                prompts = [item[0] for item in batch]
                if 'pythia' in tokenizer.name_or_path:
                    answers = [' ' + str(item[1]) for item in batch]
                else:
                    answers = [str(item[1]) for item in batch]
                answers_tokens = [tokenize(tokenizer, ans, prepend_bos=False).squeeze() for ans in answers]
                
                inputs = model.to_tokens(prompts, prepend_bos=prepend_bos).to(model.cfg.device)
                logits = model.forward(inputs, return_type='logits')
                target_logits = logits[:, -1, :].cpu() 
                probs = torch.nn.functional.softmax(target_logits, dim=-1)
                corr_probs = probs[torch.arange(probs.shape[0]), answers_tokens]
                
                def function_vector_hook(activation, hook):
                    activation[:, -1, head, :] = head_activations_final_token_layer[head]
                    return activation
                
                model.add_hook(component_to_inject.valid_hook_name(), function_vector_hook)
                logits = model.forward(inputs, return_type='logits')
                target_logits = logits[:, -1, :].cpu() 
                probs = torch.nn.functional.softmax(target_logits, dim=-1)
                corr_probs_fv = probs[torch.arange(probs.shape[0]), answers_tokens]
                
                all_probs += (corr_probs_fv.cpu() - corr_probs.cpu()).sum()
                
                model.reset_hooks()
            all_probs /= len(dataset)
            head_importances[layer, head] = all_probs
    return head_importances


def extract_and_average_head_activations(model, heads_to_average, activations):
    """
    Aggregate selected attention head outputs into a single residual-stream function vector.

    Args:
        model: HookedTransformer instance of the model.
        heads_to_average: Iterable of (layer, head) pairs to include.
        activations: Cached head activations aligned with the dataset prompts.

    Returns:
        Tensor representing the summed residual contribution of the chosen heads.
    """
    function_vector = torch.zeros(model.cfg.d_model)
    for layer, head in heads_to_average:
        # Extract the activations for the specified head
        head_activations = activations[layer].mean(dim=0)[head, :]
        # Project to residual stream using W_O
        W_O = model.blocks[layer].attn.W_O[head].cpu()
        residual_contribution = head_activations @ W_O
        function_vector += residual_contribution

    return function_vector


def apply_function_vector(model, tokenizer, dataset, function_vector, batch_size=100):
    """
    Inject a function vector across layers and report accuracy improvements.

    Args:
        model: HookedTransformer instance of the model.
        tokenizer: Model tokenizer.
        dataset: Sequence of (prompt, expected_answer) pairs.
        function_vector: Residual-stream vector to add at each layer.
        batch_size: Number of instances processed per forward pass.

    Returns:
        List of mean accuracies observed after applying the function vector per layer.
    """
    def add_to_residual_stream_hook(activation, hook, function_vector):
        activation[:, -1, :] += function_vector
        return activation

    accuracies = []
    for layer in tqdm(range(model.cfg.n_layers)):
        model.reset_hooks()
        model.add_hook(f"blocks.{layer}.hook_resid_post", partial(add_to_residual_stream_hook, function_vector=function_vector))
        res = evaluate(tokenizer, model, dataset, batch_size=batch_size, verbose=False)
        accuracies.append(np.mean(res))
        
    model.reset_hooks()
    return accuracies