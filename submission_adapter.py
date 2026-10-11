"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
from src.sft import build_sft_batch_impl, compute_response_logprobs_impl, compute_sft_loss_impl
from src.dpo import build_dpo_batch_impl, compute_dpo_loss_from_logps_impl, compute_dpo_loss_impl


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_sft_batch_impl(tokenizer=tokenizer, prompts=prompts, responses=responses, max_length=max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return compute_response_logprobs_impl(model=model, tokenizer=tokenizer, prompts=prompts, responses=responses, max_length=max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return compute_sft_loss_impl(model=model, tokenizer=tokenizer, prompts=prompts, responses=responses, max_length=max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return build_dpo_batch_impl(tokenizer=tokenizer, prompts=prompts, chosen_responses=chosen_responses, rejected_responses=rejected_responses, max_length=max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return compute_dpo_loss_from_logps_impl(policy_chosen_logps=policy_chosen_logps, policy_rejected_logps=policy_rejected_logps, reference_chosen_logps=reference_chosen_logps, reference_rejected_logps=reference_rejected_logps, beta=beta)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return compute_dpo_loss_impl(policy_model=policy_model, reference_model=reference_model, tokenizer=tokenizer, prompts=prompts, chosen_responses=chosen_responses, rejected_responses=rejected_responses, beta=beta, max_length=max_length)
