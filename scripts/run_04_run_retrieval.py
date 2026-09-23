#!/usr/bin/env python3

'''
Run:
python scripts/run_04_run_retrieval.py --config configs/04_run_retrieval.yaml
'''
from __future__ import annotations

from pipeline_utils import ensure_parent, load_config, parse_config_arg, require_dir, require_file, require_keys, resolve_path, run_script_main


def main() -> None:
    args = parse_config_arg("Stage 04: run retrieval against prompt/benchmark data.")
    config = load_config(args.config)
    require_keys(config, ["index_dir", "prompt_file", "output_jsonl"])

    index_dir = require_dir(config["index_dir"], "retrieval index directory")
    require_file(index_dir / "metadata.jsonl", "retrieval index metadata")
    require_file(index_dir / "embeddings.npy", "retrieval index embeddings")
    prompt_file = require_file(config["prompt_file"], "prompt or benchmark file")
    output_jsonl = resolve_path(config["output_jsonl"])
    ensure_parent(output_jsonl)

    method = config.get("method", "hybrid")
    retrieval_args = [
        index_dir,
        prompt_file,
        output_jsonl,
        "--method",
        method,
        "--top-k",
        config.get("top_k", 10),
    ]
    if config.get("run_name"):
        retrieval_args.extend(["--run-name", config["run_name"]])
    if method == "hybrid":
        retrieval_args.extend(
            [
                "--vector-weight",
                config.get("vector_weight", 0.7),
                "--bm25-weight",
                config.get("bm25_weight", 0.3),
            ]
        )

    env = {
        "EMBEDDING_MODEL": config.get("embedding_model"),
        "EMBEDDING_DIMENSIONS": config.get("embedding_dimensions"),
    }

    print(f"Running {method} retrieval for prompts: {prompt_file}")
    run_script_main("retrieval/scripts/run_retrieval.py", retrieval_args, env=env)
    print(f"Stage 04 output: {output_jsonl}")


if __name__ == "__main__":
    main()

