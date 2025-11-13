import torch
from tqdm.auto import tqdm
from typing import List, Sequence, Tuple
from transformer_lens import HookedTransformer


def extract_partial_sums(
    dataset: Sequence[Tuple[Sequence[Tuple[Tuple[str, ...], str]], Sequence[str], str, str]]
) -> Tuple[List[List[Tuple[int, int, int]]], List[Tuple[int, int, int]]]:
    """
    Extract partial sums for ICE examples and task prompts from the templated dataset.

    Args:
        dataset: Sequence of (ice_examples, task_prompt_operands, task_prompt, template) entries.

    Returns:
        ice_partial_sums: list per example containing partial sums for each ICE prompt.
        task_partial_sums: list of partial sums for the final task prompt.
    """
    ice_partial_sums: List[List[Tuple[int, int, int]]] = []
    task_partial_sums: List[Tuple[int, int, int]] = []

    for ice_examples, task_prompt_operands, _, _ in dataset:
        ice_sums: List[Tuple[int, int, int]] = []
        for operands, _ in ice_examples:
            # We use a, b, c to denote ICE operands
            a, b, c = map(int, operands[:3])
            ice_sums.append((a + b, b + c, a + b + c))
        ice_partial_sums.append(ice_sums)

        # We use x, y, z to denote task prompt operands
        x, y, z= map(int, task_prompt_operands[:3])
        task_partial_sums.append((x + y, y + z, x + y + z))

    return ice_partial_sums, task_partial_sums


def is_partial_sum_in_topk_logit_lens(
    logits: torch.Tensor,
    partial_sum_str: str,
    model: HookedTransformer,
    k: int = 5,
) -> bool:
    """
    Determine whether a partial sum token appears within the model’s top-k predictions.

    Args:
        logits: Token logits of shape (vocab_size,) for the current prediction step.
        partial_sum_str: Partial sum string to evaluate (e.g., "12" or " 5").
        model: HookedTransformer instance of the model.
        k: Number of top tokens to consider.

    Returns:
        True if any tokenized variant of `partial_sum_str` lies within the top-k tokens.
    """
    # Consider both raw and space-prefixed versions to handle tokenizer nuances.
    possible_forms = [partial_sum_str, f" {partial_sum_str}"]
    token_ids: List[int] = []

    for form in possible_forms:
        # Tokenize each candidate form and keep only single-token matches.
        tokens = model.to_tokens(form, prepend_bos=False)
        if tokens.shape[-1] == 1:
            token_ids.append(tokens[0, -1].item())

    if not token_ids:
        return False  # None of the forms map to a single token.

    # Select the top-k token IDs predicted for the current position.
    _, topk_indices = torch.topk(logits, k)
    topk_set = set(topk_indices.cpu().tolist())

    # Return True if any candidate token ID appears in the top-k predictions.
    return any(token_id in topk_set for token_id in token_ids)


def probe_partial_sums_logit_lens(
    dataset: Sequence[Tuple[str, ...]],
    partial_sums_list: Sequence[Sequence[int]],
    model: HookedTransformer,
    include_ice: bool = True,
    k: int = 5,
    prepend_bos: bool = True,
) -> torch.Tensor:
    """
    Track how often partial sums appear in the model’s top-k predictions across layers and positions.

    Args:
        dataset: Sequence of prompts and metadata.
        partial_sums_list: Partial sums aligned with each dataset entry.
        model: HookedTransformer instance of the model.
        include_ice: Whether prompts include in-context examples.
        k: Number of top tokens to consider.
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        Tensor of shape (n_examples, 3, n_layers + 1, max_seq_len) counting partial-sum hits.
    """
    n_layers = model.cfg.n_layers
    max_seq_len = len(model.to_tokens(dataset[0][0], prepend_bos=prepend_bos)[0]) if include_ice else len(
        model.to_tokens(dataset[0][0].split(". ")[1], prepend_bos=prepend_bos)[0]
    )
    partial_sums_heatmaps = []

    # Iterate over each prompt with its aligned partial sums.
    for prompt_bundle, partial_sums in tqdm(zip(dataset, partial_sums_list), total=len(dataset)):
        # Track three partial sums across layers and token positions.
        curr_heatmaps = torch.zeros((3, n_layers + 1, max_seq_len))
        prompt_text = prompt_bundle[0] if include_ice else prompt_bundle[0].split(". ")[1].strip()
        # Tokenize once so cached activations cover every layer.
        tokens = model.to_tokens(prompt_text, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers + 1):
            resid_name = f"blocks.{layer}.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
            resid = cache[resid_name][0]
            # Convert residual activations to logits at each position.
            logits_all = model.ln_final(resid) @ model.W_U + model.b_U
            probs_all = torch.nn.functional.softmax(logits_all, dim=-1)

            for token_pos in range(max_seq_len):
                probs = probs_all[token_pos, :]
                for ps_idx, partial_sum in enumerate(partial_sums):
                    # Record hits when the partial sum enters the top-k predictions.
                    if is_partial_sum_in_topk_logit_lens(probs, str(partial_sum), model, k=k):
                        curr_heatmaps[ps_idx, layer, token_pos] += 1

        partial_sums_heatmaps.append(curr_heatmaps)

    return torch.stack(partial_sums_heatmaps)