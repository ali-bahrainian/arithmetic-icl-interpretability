import torch
from functools import partial
from tqdm.auto import tqdm
import transformer_lens.patching as patching
from .utils import tokenize, tokenize_and_pad_with_whitespace, predict_result_from_tokens
from typing import Any, Dict, List, Sequence, Tuple
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

OPERATORS = {'+', '-', '*', '/', 'times', 'minus', 'plus', 'divide'}
EQUAL_SIGNS = {'=', 'is', 'equals'}

def get_logit_diff(logits: torch.Tensor, answer_token_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean correct-minus-incorrect logit gap for a batch of model predictions.

    Args:
        logits (torch.Tensor): Tensor of shape (batch, seq_len, vocab) or (batch, vocab) containing model logits.
        answer_token_indices (torch.Tensor): Tensor of shape (batch, 2) with [correct_index, incorrect_index] pairs.

    Returns:
        torch.Tensor: Scalar tensor with the average logit difference across the batch.
    """
    # Collapse sequence dimension if logits were provided per position.
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    # Gather logits for the correct and incorrect tokens.
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    # Average the logit gap across the batch.
    return (correct_logits - incorrect_logits).mean()


def compute_macro_mean(
    patched_results: Sequence[torch.Tensor],
    clean_logit_diffs: Sequence[float],
    corrupted_logit_diffs: Sequence[float],
) -> torch.Tensor:
    """
    Compute macro-averaged patching effects.
    """
    patched_stack = torch.stack(patched_results)
    clean_mean = torch.tensor(clean_logit_diffs).mean()
    corrupted_mean = torch.tensor(corrupted_logit_diffs).mean()
    return (patched_stack.mean(dim=0) - corrupted_mean) / (clean_mean - corrupted_mean)


def compute_micro_mean(
    patched_results: Sequence[torch.Tensor],
    clean_logit_diffs: Sequence[float],
    corrupted_logit_diffs: Sequence[float],
) -> torch.Tensor:
    """
    Compute micro-averaged patching effects.
    """
    patched_stack = torch.stack(patched_results)
    clean_tensor = torch.tensor(clean_logit_diffs)[:, None, None]
    corrupted_tensor = torch.tensor(corrupted_logit_diffs)[:, None, None]
    return torch.mean((patched_stack - corrupted_tensor) / (clean_tensor - corrupted_tensor), dim=0)


def compute_pe(
    patched_results: Sequence[torch.Tensor],
    corrupted_logit_diffs: Sequence[float],
) -> torch.Tensor:
    """
    Compute raw patching effects
    """
    patched_stack = torch.stack(patched_results)
    corrupted_tensor = torch.tensor(corrupted_logit_diffs)[:, None, None]
    return torch.mean(patched_stack - corrupted_tensor, dim=0)


def prepare_diagnostic_dataset(tokenizer: PreTrainedTokenizerBase,
    model: HookedTransformer,
    dataset_ice: Sequence[Tuple[str, str]],
    dataset_no_ice: Sequence[Tuple[str, str]],
    prepend_bos: bool = True,
) -> List[Dict[str, Any]]:
    """
    Collect prompt pairs where ICE guidance fixes an otherwise incorrect prediction.

    Args:
        tokenizer: Model tokenizer.
        model: HookedTransformer instance of the model.
        dataset_ice: Sequence of (prompt, answer, template) tuples with ICE examples.
        dataset_no_ice: Sequence of (prompt, answer, template) tuples without ICE examples.
        device: Target device for generated token tensors.
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        List of dictionaries describing qualifying prompt pairs and their tokenized answers.
    """
    dataset = []
    
    # Iterate over aligned ICE/no-ICE prompt pairs.
    for i in tqdm(range(len(dataset_ice))):
        entry_ice, ans_ice = dataset_ice[i]
        entry_no_ice, ans_no_ice = dataset_no_ice[i]
        
        # Tokenize both prompts, padding the no-ICE version to match length.
        entry_ice_tokens = tokenize(tokenizer, entry_ice, prepend_bos=prepend_bos)
        entry_no_ice_tokens = tokenize_and_pad_with_whitespace(tokenizer, entry_no_ice, entry_ice_tokens.shape[-1], prepend_bos=prepend_bos)
        
        # Generate predictions with and without in-context examples.
        prediction_no_ice = predict_result_from_tokens(tokenizer, model, entry_no_ice_tokens)[0]
        prediction_ice = predict_result_from_tokens(tokenizer, model, entry_ice_tokens)[0]    
        
        # Keep cases corrected by ICE while the baseline remains wrong.
        if prediction_no_ice.strip() != str(ans_no_ice) and prediction_ice.strip() == str(ans_ice):
            dataset.append({'prompt_ice': entry_ice, 
                            'correct_answer': str(ans_ice), 
                            'correct_answer_tok': tokenizer.encode(str(ans_ice), add_special_tokens=False)[0],
                            'prompt_no_ice': entry_no_ice, 
                            'predicted_answer_ice': prediction_ice,
                            'predicted_answer_ice_tok': tokenizer.encode(prediction_ice, add_special_tokens=False)[0],
                            'predicted_answer_no_ice': prediction_no_ice,
                            'predicted_answer_no_ice_tok': tokenizer.encode(prediction_no_ice, add_special_tokens=False)[0]
                            })
    return dataset


def get_patched_result(
    batch: Sequence[Dict[str, Any]],
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    activation_name: str = "resid_pre",
    prepend_bos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute clean, corrupted, and activation-patched logit differences for a batch of prompts.

    Args:
        batch: Sequence of prompt metadata dictionaries.
        model: HookedTransformer instance of the model.
        tokenizer: Model tokenizer.
        max_len: Target token length used to pad all prompts.
        activation_name: Activation type to patch (`"resid_pre"`, `"attn_layer"`, `"mlp"`, or `"head"`).
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        Tuple of patched, clean, and corrupted logit differences.
    """
    # Tokenize ICE prompts.
    prompts_ice_tokenized = [
        tokenize_and_pad_with_whitespace(tokenizer, item["prompt_ice"], target_len=max_len, prepend_bos=prepend_bos)
        for item in batch
    ]
    # Tokenize no-ICE prompts, padded to the same length.
    prompts_no_ice_tokenized = [
        tokenize_and_pad_with_whitespace(tokenizer, item["prompt_no_ice"], target_len=max_len, prepend_bos=prepend_bos)
        for item in batch
    ]
    
    prompts_ice_tokenized = torch.cat(prompts_ice_tokenized).to(model.cfg.device)
    prompts_no_ice_tokenized = torch.cat(prompts_no_ice_tokenized).to(model.cfg.device)
    
    clean_logits, clean_cache = model.run_with_cache(prompts_ice_tokenized)
    corrupted_logits = model(prompts_no_ice_tokenized)
    
    answer_token_indices = torch.tensor([[item['correct_answer_tok'], item['predicted_answer_no_ice_tok']] for item in batch]).to(model.cfg.device)
    
    clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).cpu()
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).cpu()
    
    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)
    
    if activation_name == 'resid_pre':
        patched_logit_diff = patching.get_act_patch_resid_pre(model, prompts_no_ice_tokenized, clean_cache, metric).cpu()
    elif activation_name == 'attn_layer':
        patched_logit_diff = patching.get_act_patch_attn_out(model, prompts_no_ice_tokenized, clean_cache, metric).cpu()
    elif activation_name == 'mlp':
        patched_logit_diff = patching.get_act_patch_mlp_out(model, prompts_no_ice_tokenized, clean_cache, metric).cpu()
    elif activation_name == 'head':
        patched_logit_diff = patching.get_act_patch_attn_head_out_all_pos(model, prompts_no_ice_tokenized, clean_cache, metric).cpu()

    return patched_logit_diff, clean_logit_diff, corrupted_logit_diff


def patch_everything(
    model: HookedTransformer,
    dataset: Sequence[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 100,
    activation_name: str = "resid_pre",
    prepend_bos: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply activation patching across the dataset and collect logit difference statistics.

    Args:
        model: HookedTransformer instance of the model.
        dataset: Sequence of filtered prompt dictionaries.
        tokenizer: Model tokenizer.
        batch_size: Number of samples processed per patching batch.
        activation_name: Activation site to patch (`"resid_pre"`, `"attn_layer"`, `"mlp"`, or `"head"`).
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        Tuple containing lists of patched, clean, and corrupted logit differences for all batches.
    """
    num_batches = len(dataset) // batch_size + 1
    all_patched_logit_diffs = []
    all_clean_logit_diffs = []
    all_corrupted_logit_diffs = []
    
    for i in tqdm(range(num_batches)):
        batch = dataset[i * batch_size:(i + 1) * batch_size]
        if len(batch) > 0:
            # Determine maximum token length inside the dataset for padding.
            max_len = max([tokenize(tokenizer, item["prompt_ice"], prepend_bos=prepend_bos).shape[-1] for item in dataset])
            patched_logit_diff, clean_logit_diff, corrupted_logit_diff = get_patched_result(
                batch, model, tokenizer, max_len, activation_name=activation_name, prepend_bos=prepend_bos
            )
            
            all_patched_logit_diffs.append(patched_logit_diff)
            all_clean_logit_diffs.append(clean_logit_diff)
            all_corrupted_logit_diffs.append(corrupted_logit_diff)
    
    return all_patched_logit_diffs, all_clean_logit_diffs, all_corrupted_logit_diffs


def extract_operands_and_operators_results(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Sequence[Dict[str, Any]],
    patched_results: Sequence[torch.Tensor],
    prepend_bos: bool = True,
) -> List[torch.Tensor]:
    """
    Restrict patched activation metrics to tokens representing operands or operators.

    Args:
        tokenizer: Model tokenizer.
        dataset: Sequence of prompt dictionaries aligned with `patched_results`.
        patched_results: Sequence of tensors containing patched statistics per position.
        prepend_bos: Whether the prompts were tokenized with a BOS token.

    Returns:
        List of tensors sliced to operand/operator positions for each prompt.
    """
    patched_results_operands_and_operators: List[torch.Tensor] = []
    for idx, item in enumerate(dataset[:len(patched_results)]):
        # Tokenize the ICE prompt so positions align with cached metrics.
        prompt_ice_tokenized = tokenize(tokenizer, item['prompt_ice'], prepend_bos=prepend_bos)
        # Decode tokens for simple string-based operator checks.
        curr_tokens = [tokenizer.decode(token) for token in prompt_ice_tokenized[0]]
        considered_positions = []
        for position, token in enumerate(curr_tokens):
            stripped = token.strip()
            # Track indices containing arithmetic symbols or numerals.
            if stripped in OPERATORS or stripped in EQUAL_SIGNS or stripped.isnumeric():
                considered_positions.append(position)
        # Store only the patching results associated with the selected positions.
        patched_results_operands_and_operators.append(patched_results[idx][:, considered_positions])
    return patched_results_operands_and_operators