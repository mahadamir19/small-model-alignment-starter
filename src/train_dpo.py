import argparse
import gc
import json
import math
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.dpo import build_dpo_batch_impl, compute_dpo_loss_from_logps_impl


@dataclass
class DPOConfig:
    track: str = "hh_rlhf"
    sft_checkpoint: str = "outputs/sft/best"
    train_examples: int = 15000
    val_examples: int = 1000
    max_length: int = 512
    max_prompt_length: int = 256
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    beta: float = 0.1
    max_grad_norm: float = 1.0
    eval_every_steps: int = 100
    log_every_steps: int = 10
    seed: int = 42
    use_fp16: bool = True
    gradient_checkpointing: bool = True
    output_root: str = "outputs/dpo"
    run_name: str = "full"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    if config.track not in ("hh_rlhf", "ultrafeedback"):
        raise ValueError("Invalid DPO training track")
    if config.train_examples < 1 or config.val_examples < 1:
        raise ValueError("Training and validation sizes must be positive")
    if config.batch_size < 1 or config.gradient_accumulation_steps < 1 or config.num_epochs < 1:
        raise ValueError("Batch size, accumulation, and epochs must be positive")
    if not 0 < config.max_prompt_length < config.max_length:
        raise ValueError("Invalid prompt and sequence length settings")
    if config.beta <= 0 or config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("Invalid optimization settings")
    if config.eval_every_steps < 1 or config.log_every_steps < 1:
        raise ValueError("Logging and evaluation intervals must be positive")
    if not config.run_name or config.run_name != Path(config.run_name).name:
        raise ValueError("run_name must be a single directory name")


def load_policy_reference_and_tokenizer(config):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This full DPO run requires Kaggle GPU T4 x2")
    checkpoint = Path(config.sft_checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing trained SFT checkpoint: {checkpoint}")
    model_config = AutoConfig.from_pretrained(str(checkpoint))
    if model_config.model_type != "qwen2":
        raise ValueError("Expected the Qwen2.5-0.5B SFT checkpoint, not a local GPT-2 smoke-test checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must provide a valid padding token or EOS fallback")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    policy_device = torch.device("cuda:0")
    reference_device = torch.device("cuda:1")
    policy = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(policy_device)
    reference = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(reference_device)

    for model in (policy, reference):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.train()
    reference.eval()
    reference.requires_grad_(False)
    return policy, reference, tokenizer


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def parse_hh_row(row):
    pattern = r"^\s*Human:\s*(.*?)\s*\n\nAssistant:\s*(.*?)\s*$"
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    match_chosen = re.match(pattern, chosen, flags=re.DOTALL)
    match_rejected = re.match(pattern, rejected, flags=re.DOTALL)
    if match_chosen is None or match_rejected is None:
        return None
    prompt_chosen, response_chosen = match_chosen.groups()
    prompt_rejected, response_rejected = match_rejected.groups()
    if prompt_chosen != prompt_rejected:
        return None
    if re.search(r"\n\n(?:Human|Assistant):", prompt_chosen):
        return None
    if re.search(r"\n\n(?:Human|Assistant):", response_chosen + "\n\n" + response_rejected):
        return None
    return {"instruction": prompt_chosen, "chosen": response_chosen, "rejected": response_rejected}


def parse_ultrafeedback_row(row):
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if [turn.get("role") for turn in chosen] != ["user", "assistant"]:
        return None
    if [turn.get("role") for turn in rejected] != ["user", "assistant"]:
        return None
    if chosen[0].get("content") != rejected[0].get("content"):
        return None
    return {
        "instruction": chosen[0].get("content"),
        "chosen": chosen[1].get("content"),
        "rejected": rejected[1].get("content"),
    }


def select_pairs(rows, parser, tokenizer, target_count, config, seen_prompts=None):
    seen = set() if seen_prompts is None else set(seen_prompts)
    selected = []
    rejected_counts = {"format": 0, "empty": 0, "duplicate": 0, "length": 0}
    for row in rows:
        item = parser(row)
        if item is None:
            rejected_counts["format"] += 1
            continue
        if any(not isinstance(item.get(key), str) for key in ("instruction", "chosen", "rejected")):
            rejected_counts["format"] += 1
            continue
        instruction = item["instruction"].strip()
        chosen = item["chosen"].strip()
        rejected = item["rejected"].strip()
        if not instruction or not chosen or not rejected or chosen == rejected:
            rejected_counts["empty"] += 1
            continue
        prompt = format_prompt(instruction)
        if prompt in seen:
            rejected_counts["duplicate"] += 1
            continue
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not 0 < len(prompt_ids) <= config.max_prompt_length or len(prompt_ids) >= config.max_length:
            rejected_counts["length"] += 1
            continue
        if not tokenizer.encode(chosen, add_special_tokens=False) or not tokenizer.encode(rejected, add_special_tokens=False):
            rejected_counts["empty"] += 1
            continue
        selected.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        seen.add(prompt)
        if len(selected) == target_count:
            break
    if len(selected) < target_count:
        raise RuntimeError(f"Only {len(selected)} usable pairs found out of {target_count} requested; exclusions: {rejected_counts}")
    return selected, rejected_counts


def prepare_dpo_datasets(config, tokenizer):
    if config.track == "hh_rlhf":
        name = "Anthropic/hh-rlhf"
        parser = parse_hh_row
    else:
        name = "HuggingFaceH4/ultrafeedback_binarized"
        parser = parse_ultrafeedback_row
    train_split, val_split = ("train", "test") if config.track == "hh_rlhf" else ("train_prefs", "test_prefs")
    raw_train = load_dataset(name, split=train_split)
    raw_val = load_dataset(name, split=val_split)
    train_rows = raw_train.shuffle(seed=config.seed)
    val_rows = raw_val.shuffle(seed=config.seed)
    train, train_exclusions = select_pairs(train_rows, parser, tokenizer, config.train_examples, config)
    val, val_exclusions = select_pairs(
        val_rows, parser, tokenizer, config.val_examples, config,
        seen_prompts={row["prompt"] for row in train}
    )
    details = {
        "dataset_name": name,
        "train_split": train_split,
        "val_split": val_split,
        "seed": config.seed,
        "train_examples": len(train),
        "val_examples": len(val),
        "train_exclusions": train_exclusions,
        "val_exclusions": val_exclusions,
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n"
    }
    return train, val, details


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def create_dpo_dataloaders(train_data, val_data, config):
    if not train_data or not val_data:
        raise ValueError("Training and validation datasets must be nonempty")
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0, generator=generator),
        DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0),
    )


def create_dpo_optimizer(policy, config):
    return torch.optim.AdamW(
        (param for param in policy.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def score_response_batch(model, batch, amp_enabled):
    device = next(model.parameters()).device
    inputs = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    shifted_logits = logits[:, :-1, :].float()
    targets = inputs["labels"][:, 1:]
    valid = targets.ne(-100)
    safe_targets = targets.masked_fill(~valid, 0)
    selected = shifted_logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    log_normalizer = torch.logsumexp(shifted_logits, dim=-1)
    token_logps = selected - log_normalizer
    return token_logps.masked_fill(~valid, 0.0).sum(dim=-1)


def compute_dpo_scores(policy, reference, tokenizer, prompts, chosen, rejected, config):
    batches = build_dpo_batch_impl(tokenizer, prompts, chosen, rejected, config.max_length)
    policy_chosen = score_response_batch(policy, batches["chosen"], config.use_fp16)
    policy_rejected = score_response_batch(policy, batches["rejected"], config.use_fp16)
    with torch.no_grad():
        reference_chosen = score_response_batch(reference, batches["chosen"], False)
        reference_rejected = score_response_batch(reference, batches["rejected"], False)
    device = policy_chosen.device
    reference_chosen = reference_chosen.to(device)
    reference_rejected = reference_rejected.to(device)
    return policy_chosen, policy_rejected, reference_chosen, reference_rejected


def compute_dpo_loss_and_margins(policy, reference, tokenizer, batch, config):
    pc, pr, rc, rr = compute_dpo_scores(
        policy, reference, tokenizer,
        batch["prompt"], batch["chosen"], batch["rejected"], config
    )
    loss = compute_dpo_loss_from_logps_impl(pc, pr, rc, rr, config.beta)
    margins = config.beta * ((pc - rc) - (pr - rr))
    return loss, margins


def evaluate_dpo(policy, reference, tokenizer, val_loader, config):
    original_mode = policy.training
    policy.eval()
    reference.eval()
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    total_pairs = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                loss, margins = compute_dpo_loss_and_margins(policy, reference, tokenizer, batch, config)
                if not torch.isfinite(loss).item() or not torch.isfinite(margins).all().item():
                    raise FloatingPointError("Non-finite DPO validation metrics")
                count = len(batch["prompt"])
                total_loss += loss.item() * count
                total_correct += (margins > 0).sum().item()
                total_margin += margins.sum().item()
                total_pairs += count
    finally:
        policy.train(original_mode)
    if total_pairs == 0:
        raise ValueError("Validation set is empty")
    return {
        "val_loss": total_loss / total_pairs,
        "reward_accuracy": total_correct / total_pairs,
        "reward_margin": total_margin / total_pairs,
        "val_pairs": total_pairs,
    }


def save_dpo_checkpoint(policy, tokenizer, config, path, step, epoch, metrics):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    original_use_cache = policy.config.use_cache
    policy.config.use_cache = True
    try:
        policy.save_pretrained(path, safe_serialization=True)
    finally:
        policy.config.use_cache = original_use_cache
    tokenizer.save_pretrained(path)
    with (path / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump({"config": asdict(config), "step": step, "epoch": epoch, "validation": metrics}, stream, indent=2)


def plot_dpo_metrics(metrics_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(metrics_path).open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    train = [record for record in records if record["type"] == "train"]
    val = [record for record in records if record["type"] == "validation"]
    plots = [
        ("dpo_loss.png", "DPO loss", "Loss", [(train, "train_loss", "Training"), (val, "val_loss", "Validation")]),
        ("reward_accuracy.png", "Validation reward accuracy", "Fraction", [(val, "reward_accuracy", "Reward accuracy")]),
        ("reward_margin.png", "Validation reward margin", "Mean margin", [(val, "reward_margin", "Reward margin")]),
    ]
    for filename, title, ylabel, series in plots:
        figure, axis = plt.subplots(figsize=(9, 5))
        for data, key, label in series:
            if data:
                axis.plot([item["step"] for item in data], [item[key] for item in data], label=label)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)


def train_dpo(policy, reference, tokenizer, train_loader, val_loader, optimizer, config):
    device = next(policy.parameters()).device
    use_amp = config.use_fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(config.output_root) / config.track / config.run_name
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Existing metrics at {metrics_path}; choose a different run_name")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference.eval()
    reference.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    latest_metrics = None

    def validate_and_save(epoch):
        nonlocal best_accuracy, best_loss, latest_metrics
        metrics = evaluate_dpo(policy, reference, tokenizer, val_loader, config)
        latest_metrics = metrics
        append_jsonl(metrics_path, {"type": "validation", "epoch": epoch, "step": global_step, **metrics})
        print(f"Validation | step {global_step} | loss {metrics['val_loss']:.4f} | accuracy {metrics['reward_accuracy']:.4f} | margin {metrics['reward_margin']:.4f}", flush=True)
        better = metrics["reward_accuracy"] > best_accuracy or (
            metrics["reward_accuracy"] == best_accuracy and metrics["val_loss"] < best_loss
        )
        if better:
            best_accuracy = metrics["reward_accuracy"]
            best_loss = metrics["val_loss"]
            save_dpo_checkpoint(policy, tokenizer, config, output_dir / "best", global_step, epoch, metrics)
            print("Saved best checkpoint", flush=True)

    for epoch in range(config.num_epochs):
        policy.train()
        accumulated_examples = 0
        accumulated_loss = 0.0
        last_batch_validated = False
        for index, batch in enumerate(train_loader):
            count = len(batch["prompt"])
            loss, _ = compute_dpo_loss_and_margins(policy, reference, tokenizer, batch, config)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite DPO loss at epoch {epoch + 1} batch {index + 1}")
            scaler.scale(loss * count).backward()
            accumulated_examples += count
            accumulated_loss += loss.detach().item() * count
            last_batch = index + 1 == len(train_loader)
            update_due = (index + 1) % config.gradient_accumulation_steps == 0 or last_batch
            if not update_due:
                continue
            scaler.unscale_(optimizer)
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.div_(accumulated_examples)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_skipped = use_amp and scaler.get_scale() < scale_before
            if not update_skipped:
                global_step += 1
                record = {
                    "type": "train", "epoch": epoch + 1, "step": global_step,
                    "train_loss": accumulated_loss / accumulated_examples,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm.detach().item(),
                }
                append_jsonl(metrics_path, record)
                if global_step % config.log_every_steps == 0 or last_batch:
                    print(f"Epoch {epoch + 1} | step {global_step} | loss {record['train_loss']:.4f} | lr {record['learning_rate']:.8f} | norm {record['grad_norm']:.4f}", flush=True)
                if global_step % config.eval_every_steps == 0 or last_batch:
                    validate_and_save(epoch + 1)
                    if last_batch:
                        last_batch_validated = True
            optimizer.zero_grad(set_to_none=True)
            accumulated_examples = 0
            accumulated_loss = 0.0
        if not last_batch_validated:
            validate_and_save(epoch + 1)
    save_dpo_checkpoint(policy, tokenizer, config, output_dir / "final", global_step, config.num_epochs, latest_metrics)
    plot_dpo_metrics(metrics_path, output_dir)
    print(f"Completed {config.track}: {global_step} steps, best reward accuracy {best_accuracy:.4f}", flush=True)
    return {"steps": global_step, "best_reward_accuracy": best_accuracy, "output_dir": str(output_dir)}


def run_track(config):
    validate_config(config)
    set_seed(config.seed)
    output_dir = Path(config.output_root) / config.track / config.run_name
    if (output_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Existing run at {output_dir}; change --run-name")
    policy, reference, tokenizer = load_policy_reference_and_tokenizer(config)
    train_data, val_data, details = prepare_dpo_datasets(config, tokenizer)
    save_jsonl(output_dir / "data" / "train.jsonl", train_data)
    save_jsonl(output_dir / "data" / "validation.jsonl", val_data)
    with (output_dir / "data" / "data_info.json").open("w", encoding="utf-8") as stream:
        json.dump(details, stream, indent=2)
    train_loader, val_loader = create_dpo_dataloaders(train_data, val_data, config)
    optimizer = create_dpo_optimizer(policy, config)
    print(f"Track: {config.track} | train: {len(train_data)} | validation: {len(val_data)}", flush=True)
    print(f"Policy: {next(policy.parameters()).device} | reference: {next(reference.parameters()).device}", flush=True)
    try:
        return train_dpo(policy, reference, tokenizer, train_loader, val_loader, optimizer, config)
    finally:
        del optimizer, policy, reference
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("hh_rlhf", "ultrafeedback", "both"), default="both")
    parser.add_argument("--sft-checkpoint", default="outputs/sft/best")
    parser.add_argument("--output-root", default="outputs/dpo")
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=10)
    arguments = parser.parse_args()
    tracks = ("hh_rlhf", "ultrafeedback") if arguments.track == "both" else (arguments.track,)
    for track in tracks:
        config = DPOConfig(
            track=track,
            sft_checkpoint=arguments.sft_checkpoint,
            output_root=arguments.output_root,
            run_name=arguments.run_name,
            seed=arguments.seed,
            eval_every_steps=arguments.eval_every_steps,
            log_every_steps=arguments.log_every_steps,
        )
        run_track(config)


if __name__ == "__main__":
    main()
