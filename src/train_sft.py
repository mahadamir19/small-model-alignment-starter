import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.sft import compute_sft_loss_impl


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str = "train_sft"
    val_split: str = "test_sft"
    train_examples: int = 30000
    val_examples: int = 1000
    max_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_every_steps: int = 100
    log_every_steps: int = 10
    seed: int = 42
    use_fp16: bool = True
    gradient_checkpointing: bool = True
    shuffle_buffer_size: int = 10000
    output_dir: str = "outputs/sft"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle SFT training requires a CUDA GPU accelerator")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The selected tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model = model.to(device)
    return model, tokenizer, device


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def select_single_turn_examples(config, tokenizer, split, target_count, seed, excluded_prompts=None):
    if target_count < 1:
        raise ValueError("The requested dataset size must be positive")
    excluded_prompts = set() if excluded_prompts is None else set(excluded_prompts)
    selected = []
    seen = set(excluded_prompts)
    stream = load_dataset(config.dataset_name, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=config.shuffle_buffer_size)

    for row in stream:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        first, second = messages[0], messages[1]
        if not isinstance(first, dict) or not isinstance(second, dict):
            continue
        if first.get("role") != "user" or second.get("role") != "assistant":
            continue
        instruction = first.get("content")
        response = second.get("content")
        if not isinstance(instruction, str) or not isinstance(response, str):
            continue
        instruction = instruction.strip()
        response = response.strip()
        if not instruction or not response or instruction in seen:
            continue
        prompt = format_prompt(instruction)
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_length > config.max_length - 2:
            continue
        if not tokenizer.encode(response, add_special_tokens=False):
            continue
        selected.append({
            "prompt": prompt,
            "response": response,
            "source_id": str(row.get("prompt_id") or "")
        })
        seen.add(instruction)
        if len(selected) == target_count:
            break

    if len(selected) != target_count:
        raise RuntimeError(
            f"Found only {len(selected)} usable examples in {split}; "
            f"needed {target_count}"
        )
    return selected, seen


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_sft_dataset(config, tokenizer, output_dir):
    train_data, train_prompts = select_single_turn_examples(
        config, tokenizer, config.train_split, config.train_examples, config.seed
    )
    val_data, _ = select_single_turn_examples(
        config, tokenizer, config.val_split, config.val_examples,
        config.seed + 1, excluded_prompts=train_prompts
    )
    data_dir = Path(output_dir) / "data"
    save_jsonl(data_dir / "train.jsonl", train_data)
    save_jsonl(data_dir / "validation.jsonl", val_data)
    with (data_dir / "data_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": config.dataset_name,
            "train_split": config.train_split,
            "validation_split": config.val_split,
            "train_examples": len(train_data),
            "validation_examples": len(val_data),
            "seed": config.seed,
            "validation_seed": config.seed + 1,
            "max_length": config.max_length,
            "selection": "First user-assistant exchange per dialogue, unique prompts, prompt-length filtered",
            "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n"
        }, f, indent=2)
    return train_data, val_data


def create_sft_dataloaders(train_data, val_data, config):
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader


def create_sft_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False
    )


def create_sft_scheduler(optimizer, train_loader, config):
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    total_steps = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    ) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler, total_steps, warmup_steps


def evaluate_sft(model, tokenizer, val_loader, config):
    if not len(val_loader):
        raise ValueError("Validation DataLoader cannot be empty")
    device = next(model.parameters()).device
    amp_enabled = config.use_fp16 and device.type == "cuda"
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                batch_size = len(batch["prompt"])
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    loss = compute_sft_loss_impl(
                        model, tokenizer, batch["prompt"],
                        batch["response"], config.max_length
                    )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite validation loss")
                total_loss += loss.item() * batch_size
                total_examples += batch_size
    finally:
        model.train(was_training)
    return total_loss / total_examples


def save_sft_checkpoint(model, tokenizer, config, checkpoint_dir, step, epoch, val_loss):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_cache = model.config.use_cache
    try:
        model.config.use_cache = True
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
    finally:
        model.config.use_cache = previous_cache
    tokenizer.save_pretrained(checkpoint_dir)
    with (checkpoint_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(config),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss
        }, f, indent=2)


def append_metrics(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def train_sft(model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, output_dir):
    if config.gradient_accumulation_steps < 1 or config.num_epochs < 1:
        raise ValueError("Invalid training configuration")
    if config.eval_every_steps < 1 or config.log_every_steps < 1:
        raise ValueError("Invalid logging intervals")
    if not len(train_loader) or not len(val_loader):
        raise ValueError("DataLoaders cannot be empty")
    device = next(model.parameters()).device
    amp_enabled = config.use_fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_path = Path(output_dir) / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Existing training log: {metrics_path}")
    global_step = 0
    best_val_loss = float("inf")
    last_val_loss = None
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.num_epochs):
        model.train()
        accumulated_examples = 0
        accumulated_loss = 0.0
        last_validation_step = None

        for batch_idx, batch in enumerate(train_loader):
            prompts = batch["prompt"]
            responses = batch["response"]
            batch_size = len(prompts)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss = compute_sft_loss_impl(
                    model, tokenizer, prompts, responses, config.max_length
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite SFT loss at epoch {epoch + 1}, batch {batch_idx + 1}"
                )
            scaler.scale(loss * batch_size).backward()
            accumulated_examples += batch_size
            accumulated_loss += loss.detach().item() * batch_size
            is_last_batch = batch_idx + 1 == len(train_loader)
            should_update = (
                (batch_idx + 1) % config.gradient_accumulation_steps == 0
                or is_last_batch
            )
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_examples)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = amp_enabled and scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)

            if not skipped:
                scheduler.step()
                global_step += 1
                average_loss = accumulated_loss / accumulated_examples
                train_record = {
                    "type": "train",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": average_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": grad_norm.detach().item()
                }
                append_metrics(metrics_path, train_record)
                history.append(train_record)

                if global_step % config.log_every_steps == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | "
                        f"Train loss: {average_loss:.4f} | "
                        f"LR: {learning_rate:.8f} | Grad norm: {grad_norm.item():.4f}",
                        flush=True
                    )

                if global_step % config.eval_every_steps == 0 or is_last_batch:
                    last_val_loss = evaluate_sft(model, tokenizer, val_loader, config)
                    last_validation_step = global_step
                    record = {
                        "type": "validation",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "val_loss": last_val_loss
                    }
                    append_metrics(metrics_path, record)
                    history.append(record)
                    print(
                        f"Validation | Step {global_step} | Loss: {last_val_loss:.4f}",
                        flush=True
                    )
                    if last_val_loss < best_val_loss:
                        best_val_loss = last_val_loss
                        save_sft_checkpoint(
                            model, tokenizer, config, Path(output_dir) / "best",
                            global_step, epoch + 1, best_val_loss
                        )
                        print("Saved new best checkpoint", flush=True)
            else:
                append_metrics(metrics_path, {
                    "type": "skipped_step",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch": batch_idx + 1
                })
            accumulated_examples = 0
            accumulated_loss = 0.0

        if last_validation_step != global_step:
            last_val_loss = evaluate_sft(model, tokenizer, val_loader, config)
            record = {
                "type": "validation",
                "epoch": epoch + 1,
                "step": global_step,
                "val_loss": last_val_loss
            }
            append_metrics(metrics_path, record)
            history.append(record)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                save_sft_checkpoint(
                    model, tokenizer, config, Path(output_dir) / "best",
                    global_step, epoch + 1, best_val_loss
                )

    save_sft_checkpoint(
        model, tokenizer, config, Path(output_dir) / "final",
        global_step, config.num_epochs, last_val_loss
    )
    print(f"Completed {global_step} optimizer updates", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    return history


def plot_sft_metrics(metrics_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(metrics_path).open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    train_records = [r for r in records if r["type"] == "train"]
    val_records = [r for r in records if r["type"] == "validation"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [r["step"] for r in train_records],
        [r["train_loss"] for r in train_records],
        label="Training loss"
    )
    ax.plot(
        [r["step"] for r in val_records],
        [r["val_loss"] for r in val_records],
        marker="o",
        label="Validation loss"
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("SFT loss")
    ax.set_title("SFT training and validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    config = SFTConfig()
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    set_seed(config.seed)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if (output_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Existing training log: {output_dir / 'metrics.jsonl'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Visible CUDA GPUs:", torch.cuda.device_count(), flush=True)
    model, tokenizer, device = load_model_and_tokenizer(config)
    print("Training device:", device, torch.cuda.get_device_name(device), flush=True)
    print("Loading real UltraChat examples", flush=True)
    train_data, val_data = prepare_sft_dataset(config, tokenizer, output_dir)
    print("Training examples:", len(train_data), flush=True)
    print("Validation examples:", len(val_data), flush=True)
    train_loader, val_loader = create_sft_dataloaders(train_data, val_data, config)
    optimizer = create_sft_optimizer(model, config)
    scheduler, total_steps, warmup_steps = create_sft_scheduler(
        optimizer, train_loader, config
    )
    print("Planned optimizer steps:", total_steps, flush=True)
    print("Warmup steps:", warmup_steps, flush=True)
    train_sft(
        model, tokenizer, train_loader, val_loader,
        optimizer, scheduler, config, output_dir
    )
    plot_sft_metrics(
        output_dir / "metrics.jsonl", output_dir / "loss_plot.png"
    )
    print("Artifacts saved to:", output_dir, flush=True)


if __name__ == "__main__":
    main()
