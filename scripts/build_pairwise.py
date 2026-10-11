import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairwiseConfig:
    eval_set: str = "hh_rlhf"
    generations_root: str = "outputs/eval/generations"
    output_root: str = "outputs/eval/pairwise"
    seed: int = 42
    max_pairs: int | None = None
    sft_path: str | None = None
    dpo_path: str | None = None


def validate_config(config):
    if config.eval_set not in ("hh_rlhf", "alpaca_eval"):
        raise ValueError("eval_set must be hh_rlhf or alpaca_eval")
    if config.max_pairs is not None and config.max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if config.seed < 0:
        raise ValueError("seed must be nonnegative")


def resolve_pairwise_paths(config):
    generations_dir = Path(config.generations_root) / config.eval_set
    output_dir = Path(config.output_root) / config.eval_set
    dpo_key = "hh_dpo" if config.eval_set == "hh_rlhf" else "ultrafeedback_dpo"
    return {
        "sft": Path(config.sft_path) if config.sft_path else generations_dir / "sft.json",
        "dpo": Path(config.dpo_path) if config.dpo_path else generations_dir / f"{dpo_key}.json",
        "pairs": output_dir / "pairs.jsonl",
        "mapping": output_dir / "answer_key.jsonl",
        "manifest": output_dir / "manifest.json",
        "dpo_key": dpo_key,
    }


def load_generation_file(path, eval_set, expected_generator):
    if not path.is_file():
        raise FileNotFoundError(f"Generation file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Generation file must contain a nonempty JSON list: {path}")

    by_id = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid record at {path}, row {index}")
        identifier = row.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError(f"Invalid prompt ID at {path}, row {index}")
        prompt_id = str(identifier)
        if prompt_id in by_id:
            raise ValueError(f"Duplicate prompt ID {prompt_id!r} in {path}")
        for field in ("instruction", "output", "generator", "dataset"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or empty {field!r} at {path}, row {index}")
        if row["dataset"] != eval_set:
            raise ValueError(f"Unexpected evaluation set for {prompt_id!r} in {path}")
        if row["generator"] != expected_generator:
            raise ValueError(f"Unexpected generator for {prompt_id!r} in {path}: {row['generator']}")
        by_id[prompt_id] = row
    return by_id


def check_generation_settings(paths):
    settings = []
    for key in ("sft", "dpo"):
        path = paths[key]
        manifest_path = path.with_name(f"{path.stem}.manifest.json")
        if not manifest_path.is_file():
            return
        with manifest_path.open("r", encoding="utf-8") as file:
            settings.append(json.load(file))
    comparable = (
        "eval_set", "seed", "max_prompt_tokens", "max_new_tokens",
        "max_total_tokens", "repetition_penalty", "no_repeat_ngram_size",
        "prompt_format", "decoding",
    )
    for field in comparable:
        if settings[0].get(field) != settings[1].get(field):
            raise ValueError(f"SFT and DPO generation settings differ: {field}")


def match_generation_pairs(sft_by_id, dpo_by_id):
    sft_ids = set(sft_by_id)
    dpo_ids = set(dpo_by_id)
    if sft_ids != dpo_ids:
        missing_dpo = sorted(sft_ids - dpo_ids)
        missing_sft = sorted(dpo_ids - sft_ids)
        raise ValueError(
            f"Generation files have different prompt IDs. "
            f"Missing DPO: {missing_dpo[:5]} ({len(missing_dpo)} total); "
            f"missing SFT: {missing_sft[:5]} ({len(missing_sft)} total)."
        )
    matched = []
    for prompt_id, sft_row in sft_by_id.items():
        dpo_row = dpo_by_id[prompt_id]
        if sft_row["instruction"] != dpo_row["instruction"]:
            raise ValueError(f"Different instructions for prompt ID {prompt_id!r}")
        if sft_row.get("category") != dpo_row.get("category"):
            raise ValueError(f"Different categories for prompt ID {prompt_id!r}")
        matched.append((prompt_id, sft_row, dpo_row))
    return matched


def build_blinded_pairs(matched, config, dpo_key):
    selected = matched[:config.max_pairs] if config.max_pairs is not None else matched
    rng = random.Random(config.seed)
    sft_first = [True] * ((len(selected) + 1) // 2) + [False] * (len(selected) // 2)
    rng.shuffle(sft_first)
    pairs = []
    answer_key = []

    for (prompt_id, sft_row, dpo_row), sft_is_a in zip(selected, sft_first):
        pair_id = f"{config.eval_set}:{prompt_id}"
        pair = {
            "pair_id": pair_id,
            "id": prompt_id,
            "instruction": sft_row["instruction"],
            "response_a": sft_row["output"] if sft_is_a else dpo_row["output"],
            "response_b": dpo_row["output"] if sft_is_a else sft_row["output"],
            "dataset": config.eval_set,
        }
        if "category" in sft_row:
            pair["category"] = sft_row["category"]
        pairs.append(pair)
        answer_key.append({
            "pair_id": pair_id,
            "id": prompt_id,
            "response_a_model": "sft" if sft_is_a else dpo_key,
            "response_b_model": dpo_key if sft_is_a else "sft",
        })
    return pairs, answer_key


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path, payload, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        if jsonl:
            for row in payload:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.5 * (attempt + 1))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def run_pairwise(config, overwrite=False):
    validate_config(config)
    paths = resolve_pairwise_paths(config)
    generator_names = {
        "sft": "qwen2.5-0.5b-sft",
        "hh_dpo": "qwen2.5-0.5b-hh-rlhf-dpo",
        "ultrafeedback_dpo": "qwen2.5-0.5b-ultrafeedback-dpo",
    }
    sft = load_generation_file(paths["sft"], config.eval_set, generator_names["sft"])
    dpo = load_generation_file(paths["dpo"], config.eval_set, generator_names[paths["dpo_key"]])
    check_generation_settings(paths)
    matched = match_generation_pairs(sft, dpo)
    pairs, answer_key = build_blinded_pairs(matched, config, paths["dpo_key"])
    manifest = {
        "eval_set": config.eval_set,
        "seed": config.seed,
        "max_pairs": config.max_pairs,
        "pair_count": len(pairs),
        "sft_path": str(paths["sft"].resolve()),
        "dpo_path": str(paths["dpo"].resolve()),
        "sft_sha256": sha256_file(paths["sft"]),
        "dpo_sha256": sha256_file(paths["dpo"]),
        "response_a_sft_count": sum(row["response_a_model"] == "sft" for row in answer_key),
        "response_a_dpo_count": sum(row["response_a_model"] == paths["dpo_key"] for row in answer_key),
    }

    if paths["manifest"].is_file() and not overwrite:
        with paths["manifest"].open("r", encoding="utf-8") as file:
            old_manifest = json.load(file)
        if old_manifest != manifest:
            raise ValueError("Existing pairwise run differs. Use --overwrite to rebuild it.")

    write_atomic(paths["pairs"], pairs, jsonl=True)
    write_atomic(paths["mapping"], answer_key, jsonl=True)
    write_atomic(paths["manifest"], manifest)

    print(f"Evaluation set: {config.eval_set}")
    print(f"Matched prompt IDs: {len(matched)}")
    print(f"Saved pairs: {len(pairs)}")
    print(f"SFT in A: {manifest['response_a_sft_count']}; DPO in A: {manifest['response_a_dpo_count']}")
    print(f"Judge-facing pairs: {paths['pairs']}")
    print(f"Private answer key: {paths['mapping']}")
    print(f"Manifest: {paths['manifest']}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", choices=["hh_rlhf", "alpaca_eval"], default="hh_rlhf")
    parser.add_argument("--generations-root", default="outputs/eval/generations")
    parser.add_argument("--output-root", default="outputs/eval/pairwise")
    parser.add_argument("--sft-path", default=None)
    parser.add_argument("--dpo-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = PairwiseConfig(
        eval_set=args.eval_set,
        generations_root=args.generations_root,
        output_root=args.output_root,
        seed=args.seed,
        max_pairs=args.max_pairs,
        sft_path=args.sft_path,
        dpo_path=args.dpo_path,
    )
    run_pairwise(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
