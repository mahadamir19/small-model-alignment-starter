import torch
import torch.nn.functional as F
from src.sft import build_sft_batch_impl, compute_response_logprobs_impl


def build_dpo_batch_impl(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen_batch = build_sft_batch_impl(tokenizer=tokenizer, prompts=prompts, responses=chosen_responses, 
                                        max_length=max_length)
    rejected_batch = build_sft_batch_impl(tokenizer=tokenizer, prompts=prompts, responses=rejected_responses, 
                                        max_length=max_length)

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch
    }

def compute_dpo_loss_from_logps_impl(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, 
                                reference_rejected_logps, beta):
    logw = policy_chosen_logps - reference_chosen_logps
    logl = policy_rejected_logps - reference_rejected_logps
    reward = beta * (logw - logl)
    return -F.logsigmoid(reward).mean()



def compute_dpo_loss_impl(policy_model, reference_model, tokenizer, prompts, chosen_responses, 
                          rejected_responses, beta, max_length):
    policy_chosen_logps = compute_response_logprobs_impl(model=policy_model, tokenizer=tokenizer, prompts=prompts, responses=chosen_responses, max_length=max_length)
    policy_rejected_logps = compute_response_logprobs_impl(model=policy_model, tokenizer=tokenizer, prompts=prompts, responses=rejected_responses, max_length=max_length)
    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs_impl(model=reference_model, tokenizer=tokenizer, prompts=prompts, responses=chosen_responses, max_length=max_length)
        reference_rejected_logps = compute_response_logprobs_impl(model=reference_model, tokenizer=tokenizer, prompts=prompts, responses=rejected_responses, max_length=max_length)

    reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device)
    reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device)
    return compute_dpo_loss_from_logps_impl(policy_chosen_logps=policy_chosen_logps, policy_rejected_logps=policy_rejected_logps, reference_chosen_logps=reference_chosen_logps, reference_rejected_logps=reference_rejected_logps, beta=beta)
