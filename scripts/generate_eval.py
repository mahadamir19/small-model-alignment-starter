import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKING_ROOT = Path('/kaggle/working')


@dataclass
class GenerationConfig:
    model_key: str = 'sft'
    eval_set: str = 'alpaca_eval'
    base_model_name: str = 'Qwen/Qwen2.5-0.5B'
    sft_checkpoint: str = str(REPO_ROOT / 'outputs/sft/best')
    hh_dpo_checkpoint: str = str(REPO_ROOT / 'outputs/dpo/hh_rlhf/full/best')
    ultrafeedback_dpo_checkpoint: str = str(REPO_ROOT / 'outputs/dpo/ultrafeedback/full/best')
    hh_eval_path: str = str(REPO_ROOT / 'eval_sets/hh_rlhf_eval_300_seed42.json')
    alpaca_subset_path: str = str(WORKING_ROOT / 'outputs/eval/fixed_subsets/alpaca_eval_300_seed42.json')
    output_dir: str = str(WORKING_ROOT / 'outputs/eval/generations')
    subset_size: int = 300
    num_prompts: int = 300
    seed: int = 42
    max_prompt_tokens: int = 256
    max_new_tokens: int = 256
    max_total_tokens: int = 512
    batch_size: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    if config.model_key not in ('base', 'sft', 'hh_dpo', 'ultrafeedback_dpo'):
        raise ValueError('Invalid model_key')
    if config.eval_set not in ('alpaca_eval', 'hh_rlhf'):
        raise ValueError('Invalid eval_set')
    if config.subset_size <= 0 or not 0 < config.num_prompts <= config.subset_size:
        raise ValueError('num_prompts must be between 1 and subset_size')
    if config.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if config.max_prompt_tokens <= 0 or config.max_new_tokens <= 0 or config.max_total_tokens <= 0:
        raise ValueError('Token limits must be positive')
    if config.max_prompt_tokens + config.max_new_tokens > config.max_total_tokens:
        raise ValueError('Prompt and generation exceed max_total_tokens')
    if config.repetition_penalty <= 0:
        raise ValueError('repetition_penalty must be positive')
    if config.no_repeat_ngram_size < 0:
        raise ValueError('no_repeat_ngram_size cannot be negative')


def get_output_path(config):
    return Path(config.output_dir) / config.eval_set / f'{config.model_key}.json'


def validate_eval_rows(rows, source):
    if not isinstance(rows, list):
        raise ValueError(f'{source} must be a JSON list')
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'{source} row {index} is not an object')
        if not isinstance(row.get('instruction'), str) or not row['instruction'].strip():
            raise ValueError(f'{source} row {index} has no instruction')
        if 'id' not in row or str(row['id']) in seen_ids:
            raise ValueError(f'{source} row {index} has a missing or duplicated ID')
        seen_ids.add(str(row['id']))
    return rows


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    try:
        for attempt in range(8):
            try:
                os.replace(temporary_path, path)
                return
            except PermissionError as exc:
                if attempt == 7:
                    raise PermissionError(f'Cannot replace {path}') from exc
                time.sleep(0.5 * (attempt + 1))
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except PermissionError:
                pass


def load_alpaca_eval_prompts(config):
    subset_path = Path(config.alpaca_subset_path)
    if subset_path.is_file():
        with subset_path.open('r', encoding='utf-8') as file:
            selected = json.load(file)
    else:
        data_path = hf_hub_download(
            repo_id='tatsu-lab/alpaca_eval',
            repo_type='dataset',
            filename='alpaca_eval.json',
        )
        with open(data_path, 'r', encoding='utf-8') as file:
            original_rows = json.load(file)
        if not isinstance(original_rows, list) or len(original_rows) < config.subset_size:
            raise ValueError('Not enough AlpacaEval examples')
        indices = random.Random(config.seed).sample(range(len(original_rows)), config.subset_size)
        selected = [
            {
                'id': f'alpaca-eval-{index:04d}',
                'instruction': original_rows[index]['instruction'],
                'dataset': 'alpaca_eval',
                'source_index': index,
            }
            for index in indices
        ]
        validate_eval_rows(selected, 'AlpacaEval subset')
        write_json_atomic(subset_path, selected)
    validate_eval_rows(selected, 'AlpacaEval subset')
    if len(selected) != config.subset_size:
        raise ValueError('Saved AlpacaEval subset size differs from the configured subset_size')
    return selected[:config.num_prompts]


def load_hh_rlhf_eval_prompts(config):
    path = Path(config.hh_eval_path)
    if not path.is_file():
        raise FileNotFoundError(f'HH-RLHF evaluation file not found: {path}')
    with path.open('r', encoding='utf-8') as file:
        rows = json.load(file)
    validate_eval_rows(rows, 'HH-RLHF eval')
    if len(rows) < config.num_prompts:
        raise ValueError('Not enough HH-RLHF evaluation prompts')
    return rows[:config.num_prompts]


def load_eval_prompts(config):
    if config.eval_set == 'alpaca_eval':
        return load_alpaca_eval_prompts(config)
    return load_hh_rlhf_eval_prompts(config)


def resolve_model_source(config, checkpoint_override=None):
    sources = {
        'base': config.base_model_name,
        'sft': config.sft_checkpoint,
        'hh_dpo': config.hh_dpo_checkpoint,
        'ultrafeedback_dpo': config.ultrafeedback_dpo_checkpoint,
    }
    source = checkpoint_override or sources[config.model_key]
    if config.model_key != 'base' and not Path(source).is_dir():
        raise FileNotFoundError(
            f'Model checkpoint directory not found: {source}. '
            'Pass --checkpoint-path pointing to the checkpoint attached under /kaggle/input '
            'or stored under /kaggle/working.'
        )
    return source


def load_generation_model(config, checkpoint_override=None):
    source = resolve_model_source(config, checkpoint_override)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError('Tokenizer has neither pad nor EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer, device


def format_instruction(instruction):
    return f'### Instruction:\n{instruction}\n\n### Response:\n'


def prepare_generation_prompt(tokenizer, instruction, max_prompt_tokens):
    prompt = format_instruction(instruction)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_prompt_tokens:
        return prompt
    instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
    left, right = 0, len(instruction_tokens)
    best_prompt = None
    while left <= right:
        middle = (left + right) // 2
        shortened = tokenizer.decode(
            instruction_tokens[:middle],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        candidate = format_instruction(shortened)
        count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if count <= max_prompt_tokens:
            best_prompt = candidate
            left = middle + 1
        else:
            right = middle - 1
    if best_prompt is None:
        raise ValueError('max_prompt_tokens is too small for the prompt template')
    return best_prompt


def generate_responses_batch(model, tokenizer, device, instructions, config):
    if not instructions:
        return []
    formatted_prompts = [
        prepare_generation_prompt(tokenizer, instruction, config.max_prompt_tokens)
        for instruction in instructions
    ]
    inputs = tokenizer(
        formatted_prompts,
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
    )
    padded_input_length = inputs['input_ids'].shape[1]
    if padded_input_length + config.max_new_tokens > config.max_total_tokens:
        raise ValueError('Padded inputs and generation exceed max_total_tokens')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(sequence[padded_input_length:], skip_special_tokens=True).strip()
        for sequence in generated
    ]


def generate_response(model, tokenizer, device, instruction, config):
    return generate_responses_batch(model, tokenizer, device, [instruction], config)[0]


def make_generation_record(config, prompt, response):
    generator_names = {
        'base': 'qwen2.5-0.5b-base',
        'sft': 'qwen2.5-0.5b-sft',
        'hh_dpo': 'qwen2.5-0.5b-hh-rlhf-dpo',
        'ultrafeedback_dpo': 'qwen2.5-0.5b-ultrafeedback-dpo',
    }
    record = {
        'id': prompt['id'],
        'instruction': prompt['instruction'],
        'output': response,
        'generator': generator_names[config.model_key],
        'dataset': config.eval_set,
    }
    for field in ('category', 'expected_behavior', 'source_index'):
        if field in prompt:
            record[field] = prompt[field]
    return record


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_manifest(config, checkpoint_override=None):
    source = resolve_model_source(config, checkpoint_override)
    if Path(source).is_dir():
        source = str(Path(source).resolve())
    eval_source_path = Path(
        config.alpaca_subset_path if config.eval_set == 'alpaca_eval' else config.hh_eval_path
    ).resolve()
    return {
        'schema_version': 2,
        'model_key': config.model_key,
        'model_source': source,
        'eval_set': config.eval_set,
        'eval_source': str(eval_source_path),
        'eval_source_sha256': sha256_file(eval_source_path),
        'seed': config.seed,
        'subset_size': config.subset_size,
        'max_prompt_tokens': config.max_prompt_tokens,
        'max_new_tokens': config.max_new_tokens,
        'max_total_tokens': config.max_total_tokens,
        'batch_size': config.batch_size,
        'repetition_penalty': config.repetition_penalty,
        'no_repeat_ngram_size': config.no_repeat_ngram_size,
        'prompt_format': 'instruction_response_v1',
        'decoding': 'greedy',
    }


def load_saved_generations(output_path, manifest_path, expected_manifest, prompts, overwrite=False):
    if overwrite:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if output_path.exists() and not manifest_path.exists():
        raise ValueError(
            f'Existing generation file has no run manifest: {output_path}. '
            'Use --overwrite to start a new run.'
        )
    if manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8') as file:
            existing_manifest = json.load(file)
        if existing_manifest != expected_manifest:
            raise ValueError(
                f'Generation settings, evaluation data, or checkpoint changed for {output_path}. '
                'Use a different --output-dir or --overwrite to regenerate.'
            )
    else:
        write_json_atomic(manifest_path, expected_manifest)
    if not output_path.exists():
        return {}
    try:
        with output_path.open('r', encoding='utf-8') as file:
            existing_rows = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Existing generation file is invalid JSON: {output_path}. '
            'Preserve it for inspection, or use --overwrite to start a new run.'
        ) from exc
    if not isinstance(existing_rows, list):
        raise ValueError(f'Existing generation file must be a JSON list: {output_path}')
    prompt_by_id = {str(example['id']): example for example in prompts}
    saved = {}
    for index, record in enumerate(existing_rows):
        if not isinstance(record, dict) or 'id' not in record:
            raise ValueError(f'Invalid generation record at index {index}')
        record_id = str(record['id'])
        if record_id in saved:
            raise ValueError(f'Duplicate generated ID: {record_id}')
        if not isinstance(record.get('output'), str):
            raise ValueError(f'Missing generated output for ID: {record_id}')
        if record_id in prompt_by_id:
            expected = prompt_by_id[record_id]
            if record.get('instruction') != expected['instruction']:
                raise ValueError(f'Prompt changed for previously generated ID: {record_id}')
        saved[record_id] = record
    return saved


def run_generation(config, checkpoint_override=None, overwrite=False):
    validate_config(config)
    set_seed(config.seed)
    prompts = load_eval_prompts(config)
    output_path = get_output_path(config)
    manifest_path = output_path.with_name(output_path.stem + '.manifest.json')
    expected_manifest = create_run_manifest(config, checkpoint_override)
    saved = load_saved_generations(
        output_path, manifest_path, expected_manifest, prompts, overwrite=overwrite
    )
    pending = [example for example in prompts if str(example['id']) not in saved]
    print(f'Model: {expected_manifest["model_source"]}')
    print(f'Evaluation set: {config.eval_set} | Requested prompts: {len(prompts)}')
    print(f'Already generated: {len(prompts) - len(pending)} | Remaining: {len(pending)}')
    print(f'Output: {output_path}')
    if not pending:
        print('All requested prompts already have saved responses.')
        return output_path
    model, tokenizer, device = load_generation_model(config, checkpoint_override)
    print(f'Device: {device} | Batch size: {config.batch_size} | Decoding: greedy')
    completed = len(prompts) - len(pending)
    for offset in range(0, len(pending), config.batch_size):
        batch = pending[offset:offset + config.batch_size]
        responses = generate_responses_batch(
            model, tokenizer, device, [example['instruction'] for example in batch], config
        )
        if len(responses) != len(batch):
            raise RuntimeError('Generation returned a different number of responses')
        for example, response in zip(batch, responses):
            saved[str(example['id'])] = make_generation_record(config, example, response)
        ordered = [saved[str(example['id'])] for example in prompts if str(example['id']) in saved]
        included_ids = {str(example['id']) for example in prompts}
        ordered.extend(record for id_, record in saved.items() if id_ not in included_ids)
        write_json_atomic(output_path, ordered)
        print(f'Saved {completed + offset + len(batch)}/{len(prompts)} requested responses', flush=True)
    print(f'Generation complete: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-key', choices=['base', 'sft', 'hh_dpo', 'ultrafeedback_dpo'], required=True)
    parser.add_argument('--eval-set', choices=['alpaca_eval', 'hh_rlhf'], required=True)
    parser.add_argument('--num-prompts', type=int, default=300)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default=str(WORKING_ROOT / 'outputs/eval/generations'))
    parser.add_argument('--hh-eval-path', type=str, default=str(REPO_ROOT / 'eval_sets/hh_rlhf_eval_300_seed42.json'))
    parser.add_argument('--alpaca-subset-path', type=str, default=str(WORKING_ROOT / 'outputs/eval/fixed_subsets/alpaca_eval_300_seed42.json'))
    parser.add_argument('--max-prompt-tokens', type=int, default=256)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-total-tokens', type=int, default=512)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = GenerationConfig(
        model_key=args.model_key,
        eval_set=args.eval_set,
        num_prompts=args.num_prompts,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        hh_eval_path=args.hh_eval_path,
        alpaca_subset_path=args.alpaca_subset_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    run_generation(config, args.checkpoint_path, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
