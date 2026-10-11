import torch
import torch.nn.functional as F

def build_sft_batch_impl(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal lengths")
    if len(prompts) == 0:
        raise ValueError("batch must contain at least one example")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    eos_suffix = [eos_id] if eos_id is not None else []
    unpadded_inputs = []
    unpadded_labels = []

    for prompt, response in zip(prompts, responses):
        p_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        r_tokens = tokenizer.encode(response, add_special_tokens=False)
        input_ids = p_tokens + r_tokens + eos_suffix
        labels = [-100] * len(p_tokens) + r_tokens + eos_suffix
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        unpadded_inputs.append(input_ids)
        unpadded_labels.append(labels)
        
    batch_max_len = max(len(seq) for seq in unpadded_inputs)
    final_input_ids = []
    final_attention_masks = []
    final_labels = []
    
    for input_ids, labels in zip(unpadded_inputs, unpadded_labels):
        pad_len = batch_max_len - len(input_ids)
        padded_input_ids = input_ids + [pad_id] * pad_len
        padded_labels = labels + [-100] * pad_len
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        
        final_input_ids.append(padded_input_ids)
        final_attention_masks.append(attention_mask)
        final_labels.append(padded_labels)
        
    return {
        "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(final_attention_masks, dtype=torch.long),
        "labels": torch.tensor(final_labels, dtype=torch.long)
    }


def compute_response_logprobs_impl(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch_impl(tokenizer=tokenizer, prompts=prompts, responses=responses, max_length=max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    index_labels = shift_labels.clone()
    index_labels[index_labels == -100] = 0
    target_log_probs = torch.gather(log_probs, dim=-1, index=index_labels.unsqueeze(-1)).squeeze(-1)
    
    valid_mask = (shift_labels != -100).to(target_log_probs.dtype)
    target_log_probs = target_log_probs * valid_mask
    response_log_prob_sums = target_log_probs.sum(dim=1)
    return response_log_prob_sums


def compute_sft_loss_impl(model, tokenizer, prompts, responses, max_length):
    logps = compute_response_logprobs_impl(model=model, tokenizer=tokenizer, prompts=prompts, responses=responses, max_length=max_length)
    return -logps.mean()