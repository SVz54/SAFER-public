#!/usr/bin/env python3
'''
Run:
python scripts/run_03_generate_embeddings.py --config configs/03_generate_embeddings.yaml
'''
from __future__ import annotations

from pipeline_utils import ensure_dir, load_config, parse_config_arg, require_file, require_keys, resolve_path, run_script_main


def main() -> None:
    args = parse_config_arg("Stage 03: generate embeddings and retrieval index.")
    config = load_config(args.config)
    require_keys(config, ["input_jsonl", "output_dir"])

    input_jsonl = require_file(config["input_jsonl"], "retrieval-ready JSONL input")
    output_dir = resolve_path(config["output_dir"])
    ensure_dir(output_dir)

    env = {
        "EMBEDDING_MODEL": config.get("embedding_model"),
        "EMBEDDING_DIMENSIONS": config.get("embedding_dimensions"),
    }
    index_args = [
        input_jsonl,
        output_dir,
        "--batch-size",
        config.get("batch_size", 64),
        "--sleep",
        config.get("sleep_seconds", 0.1),
    ]

    print(f"Generating embedding index from: {input_jsonl}")
    run_script_main("retrieval/scripts/build_index.py", index_args, env=env)
    print(f"Stage 03 output directory: {output_dir}")


if __name__ == "__main__":
    main()

