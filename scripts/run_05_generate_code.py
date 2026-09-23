#!/usr/bin/env python3

'''
Run:
python scripts/run_05_generate_code.py --config configs/05_generate_code.yaml'''
from __future__ import annotations

from pipeline_utils import ensure_dir, ensure_parent, load_config, optional_path, parse_config_arg, require_file, require_keys, resolve_path, run_script_main


def main() -> None:
    args = parse_config_arg("Stage 05: build generation prompts and generate code.")
    config = load_config(args.config)
    require_keys(
        config,
        [
            "retrieval_results_jsonl",
            "metadata_jsonl",
            "generation_context_jsonl",
            "generation_prompts_jsonl",
            "generation_outputs_jsonl",
        ],
    )

    retrieval_results = require_file(config["retrieval_results_jsonl"], "retrieval results JSONL")
    metadata_jsonl = require_file(config["metadata_jsonl"], "retrieval metadata JSONL")
    generation_context = resolve_path(config["generation_context_jsonl"])
    generation_prompts = resolve_path(config["generation_prompts_jsonl"])
    generation_outputs = resolve_path(config["generation_outputs_jsonl"])
    ensure_parent(generation_context)
    ensure_parent(generation_prompts)
    ensure_parent(generation_outputs)

    context = config.get("context") or {}
    context_args = [
        retrieval_results,
        metadata_jsonl,
        generation_context,
        "--top-k",
        context.get("top_k", 5),
        "--max-guidance",
        context.get("max_guidance", 4),
        "--max-consequences",
        context.get("max_consequences", 8),
        "--max-examples",
        context.get("max_examples", 4),
    ]
    if bool(context.get("include_task_tags", False)):
        context_args.append("--include-task-tags")

    print(f"Building generation context: {generation_context}")
    run_script_main("generation/build_generation_context.py", context_args)

    prompt = config.get("prompt") or {}
    prompt_args = [
        generation_context,
        generation_prompts,
        "--max-guidance",
        prompt.get("max_guidance", 3),
        "--max-consequences",
        prompt.get("max_consequences", 6),
        "--max-examples",
        prompt.get("max_examples", 2),
    ]
    if bool(prompt.get("include_task_tags", False)):
        prompt_args.append("--include-task-tags")

    print(f"Building model-ready generation prompts: {generation_prompts}")
    run_script_main("generation/build_generation_prompts.py", prompt_args)

    generation = config.get("generation") or {}
    generation_args = [
        generation_prompts,
        generation_outputs,
        "--model",
        generation.get("model", "gpt-5.4-mini"),
        "--temperature",
        generation.get("temperature", 0.2),
        "--delay",
        generation.get("delay_seconds", 0.4),
        "--max-retries",
        generation.get("max_retries", 3),
        "--start-index",
        generation.get("start_index", 1),
        "--max-records",
        generation.get("max_records", 0),
        "--code-extension",
        generation.get("code_extension", "txt"),
    ]
    artifacts_dir = optional_path(generation.get("artifacts_dir"))
    if artifacts_dir is not None:
        ensure_dir(artifacts_dir)
        generation_args.extend(["--artifacts-dir", artifacts_dir])

    print(f"Running code generation: {generation_outputs}")
    run_script_main("generation/run_generation.py", generation_args)
    print(f"Stage 05 output: {generation_outputs}")


if __name__ == "__main__":
    main()

