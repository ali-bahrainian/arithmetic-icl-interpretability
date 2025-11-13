from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer
import torch
from tqdm.auto import tqdm
from .component import Component
from typing import List, Sequence, Tuple, Optional


WORD_NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
]


def load_model_and_tokenizer(model_name: str, device: str = "cuda") -> Tuple[PreTrainedTokenizerBase, HookedTransformer]:
    """
    Load a HookedTransformer model together with its tokenizer for inference.

    Args:
        model_name: Hugging Face repository ID or local path pointing to the model.
        device: Target torch device string used when initializing the transformer weights.

    Returns:
        Tuple of (tokenizer, model) configured for evaluation mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.float16, 
        fold_ln=True,  # Fold layer-norm parameters into linear layers.
        center_unembed=True,  # Center the unembedding matrix.
        center_writing_weights=True,  # Center writing weights.
    )
    model.eval()
    return tokenizer, model


def train_test_split_per_template(
    dataset: Sequence, test_size: float = 0.4, num_samples_per_template: int = 50
) -> Tuple[Sequence, Sequence]:
    """
    Split a templated dataset into train and test portions while preserving per-template ordering.

    Args:
        dataset: Ordered collection where every `num_samples_per_template` items share a template.
        test_size: Fraction of each template chunk allocated to the test set.
        num_samples_per_template: Number of consecutive samples belonging to the same template.

    Returns:
        Two sequences containing the train and test subsets respectively.
    """
    train, test = [], []
    chunk_size = num_samples_per_template
    train_cutoff = int(chunk_size * (1 - test_size))

    for start in range(0, len(dataset), chunk_size):
        chunk = dataset[start:start + chunk_size]
        if not chunk:
            break
        train.extend(chunk[:train_cutoff])
        test.extend(chunk[train_cutoff:chunk_size])

    return train, test


def tokenize(
    tokenizer: PreTrainedTokenizerBase, text: str, prepend_bos: bool = True, prepend_eos: bool = False
) -> torch.Tensor:
    """
    Tokenize the input text using the provided tokenizer.
    Returns the tokenized input as a tensor.
    """
    tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
    if prepend_bos:
        tokenized = [tokenizer.bos_token_id] + tokenized
    if prepend_eos:
        tokenized = [tokenizer.eos_token_id] + tokenized
    return torch.tensor(tokenized).unsqueeze(0)


def tokenize_and_pad_with_whitespace(
    tokenizer: PreTrainedTokenizerBase, text: str, target_len: int, prepend_bos: bool = True
) -> torch.Tensor:
    """
    Tokenize the input text and pad it with whitespace tokens to reach the target length.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        text: The input text to tokenize.
        target_len: The desired length of the tokenized output.
        prepend_bos: Whether to prepend the BOS token.

    Returns:
        A tensor containing the tokenized and padded input.
    """
    tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
    if prepend_bos:
        tokenized = [tokenizer.bos_token_id] + tokenized
    whitespace_token = tokenizer(" ", add_special_tokens=False)['input_ids'][0]
    tokenized = [whitespace_token] * (target_len - len(tokenized)) + tokenized
    return torch.tensor(tokenized).unsqueeze(0)


def obtain_all_ice_variants(
    dataset: Sequence[Tuple[Sequence[Tuple[str, str]], Sequence[str], str]]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Build multiple prompt variants with differing numbers of in-context examples.

    Args:
        dataset: Iterable of (ice_examples, answers, final_prompt) tuples where
            ice_examples is a sequence of (metadata, formatted_prompt) pairs,
            answers contains the ground-truth result selector, and final_prompt
            is the query without ICE prefixes.

    Returns:
        Tuple of (no_ice, single_ice, two_ice, full_ice) prompt/answer pairs ready for evaluation.
    """
    no_ice = []
    single_ice = []
    two_ice = []
    all_ice = []

    def join_ice_segments(segments: Sequence[str], count: Optional[int] = None) -> str:
        if count is not None:
            count = min(len(segments), count)
            text = "".join(segments[:count])
        else:
            text = "".join(segments)
        return text

    for ice_examples, task_prompt_operands, task_prompt, template in dataset:
        ice_prompts = [example[-1] for example in ice_examples]
        expected_answer = str(task_prompt_operands[-1])

        no_ice.append((task_prompt, expected_answer))
        single_ice.append((join_ice_segments(ice_prompts, count=1) + task_prompt, expected_answer))
        two_ice.append((join_ice_segments(ice_prompts, count=2) + task_prompt, expected_answer))
        all_ice.append((join_ice_segments(ice_prompts) + task_prompt, expected_answer))

    return no_ice, single_ice, two_ice, all_ice


# def predict_sum(
#     tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, prompt: str, max_new_tokens: int = 1, device: str = "cuda", prepend_bos: bool = True
# ) -> str:
#     """
#     Generate the model output for a short numeric answer.
#     We will parse the first token(s) that appear as the model's guess for the sum.
#     """
#     inputs = tokenize(tokenizer, prompt, prepend_bos=prepend_bos).to(device)
#     with torch.no_grad():
#         outputs = model.generate(inputs, max_new_tokens=max_new_tokens, verbose=False)
#     # Decode only the newly generated tokens beyond the prompt
#     generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
#     return generated_text.strip()


# def predict_sum_transformer_lens(tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, prompt: str, device: str = "cuda", prepend_bos: bool = True) -> str:
#     """
#     Generate the model output for a short numeric answer.
#     We will parse the first token(s) that appear as the model's guess for the sum.
#     """
#     inputs = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
#     logits = model(inputs)
#     target_logits = logits[0, -1, :].cpu()
#     probs = torch.nn.functional.softmax(target_logits, dim=-1)
#     _, top_indices = torch.topk(probs, 1)
#     generated_token = tokenizer.decode(top_indices[0].item(), skip_special_tokens=True)
#     return generated_token.strip()


def predict_result_transformerlens(
    tokenizer: PreTrainedTokenizerBase, 
    model: HookedTransformer, 
    prompts: Sequence[str], 
    prepend_bos: bool = True
) -> List[str]:
    """
    Decode the model’s top-1 prediction for each prompt using transformer_lens.

    Args:
        tokenizer: Model tokenizer.
        model: HookedTransformer instance of the model.
        prompts: Batched text prompts.
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.

    Returns:
        List of decoded strings containing the generated tokens per prompt.
    """
    device = model.cfg.device
    inputs = model.to_tokens(prompts, prepend_bos=prepend_bos).to(device)
    logits = model(inputs)
    target_logits = logits[:, -1, :].cpu() 
    probs = torch.nn.functional.softmax(target_logits, dim=-1)
    _, top_indices = torch.topk(probs, 1, dim=-1)
    
    generated_tokens = []
    for i in range(top_indices.shape[0]):
        token_id = top_indices[i, 0].item()
        decoded_token = tokenizer.decode(token_id, skip_special_tokens=True)
        generated_tokens.append(decoded_token)
        
    return generated_tokens


def predict_result_from_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model: HookedTransformer, 
    tokenized_prompts: torch.Tensor
) -> List[str]:
    """
    Decode the model’s top-1 prediction for each tokenized prompt.

    Args:
        model: HookedTransformer instance of the model.
        tokenized_prompts: Batched tokenized prompts as a tensor.

    Returns:
        List of decoded strings containing the generated tokens per prompt.
    """
    device = model.cfg.device
    logits = model(tokenized_prompts.to(device))
    target_logits = logits[:, -1, :].cpu()
    probs = torch.nn.functional.softmax(target_logits, dim=-1)
    _, top_indices = torch.topk(probs, 1, dim=-1)

    generated_tokens = []
    for i in range(top_indices.shape[0]):
        token_id = top_indices[i, 0].item()
        decoded_token = tokenizer.decode(token_id, skip_special_tokens=True)
        generated_tokens.append(decoded_token)

    return generated_tokens


# def evaluate_model_on_dataset(tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, dataset: Sequence[Tuple[str, int]], device: str = "cuda", prepend_bos: bool = True) -> List[int]:
#     accuracy = []
#     for entry, ans in tqdm(dataset):    

#         prediction = predict_sum_transformer_lens(tokenizer, model, entry, device=device, prepend_bos=prepend_bos)
#         try:
#             pred_num = int(prediction.split()[0])
#             accuracy.append(1) if pred_num == ans else accuracy.append(0)
#         except:
#             pred_num= -999
#             accuracy.append(0)  
#     return accuracy


# def evaluate_model_on_dataset_str(tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, dataset: Sequence[Tuple[str, str]], device: str = "cuda", prepend_bos: bool = True) -> List[int]:
#     accuracy = []
#     for entry, ans in tqdm(dataset):    

#         prediction = predict_sum_transformer_lens(tokenizer, model, entry, device=device, prepend_bos=prepend_bos)
#         pred_num = prediction.strip()
#         accuracy.append(1) if pred_num == ans else accuracy.append(0)
#     return accuracy


# def evaluate_model_on_dataset_batches(tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, dataset: Sequence[Tuple[str, int]], device: str = "cuda", prepend_bos: bool = True, batch_size: int = 32) -> List[int]:
#     accuracy = []
#     for i in tqdm(range(0, len(dataset), batch_size)):
#         batch = dataset[i:i + batch_size]
#         entries, answers = zip(*batch)
#         predictions = predict_sum_transformer_lens_batch(tokenizer, model, entries, device=device, prepend_bos=prepend_bos)
        
#         for prediction, ans in zip(predictions, answers):
#             try:
#                 pred_num = int(prediction.split()[0])
#                 accuracy.append(1) if pred_num == ans else accuracy.append(0)
#             except:
#                 pred_num= -999
#                 accuracy.append(0)  
#     return accuracy


# def evaluate_model_on_dataset_batches_str(
#     tokenizer: PreTrainedTokenizerBase, model: HookedTransformer, dataset: Sequence[Tuple[str, str]], device: str = "cuda", prepend_bos: bool = True, batch_size: int = 32
# ) -> List[int]:
#     accuracy = []
#     for i in tqdm(range(0, len(dataset), batch_size)):
#         batch = dataset[i:i + batch_size]
#         entries, answers = zip(*batch)
#         predictions = predict_sum_transformer_lens_batch(tokenizer, model, entries, device=device, prepend_bos=prepend_bos)
        
#         for prediction, ans in zip(predictions, answers):
#             pred_num = prediction.strip()
#             accuracy.append(1) if pred_num == ans else accuracy.append(0)
#     return accuracy


def evaluate(
    tokenizer: PreTrainedTokenizerBase, 
    model: HookedTransformer, 
    dataset: Sequence[Tuple[str, str]], 
    prepend_bos: bool = True, 
    batch_size: int = 32,
    result_format: str = "num",
    verbose: bool = True
) -> List[int]:
    """
    Batch-evaluate model outputs against expected answers and return per-sample accuracy flags.

    Args:
        tokenizer: Model tokenizer.
        model: HookedTransformer instance of the model.
        dataset: Sequence of (prompt, expected_answer) pairs.
        prepend_bos: Whether to prepend the BOS token when tokenizing prompts.
        batch_size: Number of instances processed per forward pass.
        result_format: Format of predictions: `"num"` parses integers, `"str"` parses strings.

    Returns:
        List of integers where 1 indicates a correct prediction and 0 indicates an error.
    """
    accuracies = []
    for i in tqdm(range(0, len(dataset), batch_size), disable=not verbose):
        batch = dataset[i:i + batch_size]
        entries, answers = zip(*batch)
        predictions = predict_result_transformerlens(tokenizer, model, entries, prepend_bos=prepend_bos)
        if result_format == "num":
            for prediction, ans in zip(predictions, answers):
                try:
                    pred_num = int(prediction.split()[0])
                    accuracies.append(1) if pred_num == ans else accuracies.append(0)
                except:
                    accuracies.append(0)  
        elif result_format == "str":
            for prediction, ans in zip(predictions, answers):
                pred_num = prediction.strip()
                accuracies.append(1) if pred_num == ans else accuracies.append(0)
        else:
            raise ValueError(f"Unknown result_format: {result_format}")
    return accuracies
