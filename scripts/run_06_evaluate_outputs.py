#!/usr/bin/env python3
''' 
Run:
python scripts/run_06_evaluate_outputs.py --config configs/06_evaluate_outputs.yaml
'''
from __future__ import annotations

from pipeline_utils import ensure_parent, load_config, parse_config_arg, require_file, require_keys, resolve_path, run_script_main


def main() -> None:
    args = parse_config_arg("Stage 06: evaluate retrieval outputs.")
    config = load_config(args.config)
    require_keys(config, ["benchmark_file", "retrieval_results_jsonl", "output_prefix"])

    benchmark_file = require_file(config["benchmark_file"], "benchmark file")
    retrieval_results = require_file(config["retrieval_results_jsonl"], "retrieval results JSONL")
    output_prefix = resolve_path(config["output_prefix"])
    ensure_parent(output_prefix)

    print(f"Scoring retrieval results: {retrieval_results}")
    run_script_main(
        "retrieval/scripts/score_retrieval_results.py",
        [benchmark_file, retrieval_results, output_prefix],
    )

    comparison = config.get("comparison") or {}
    summary_files = comparison.get("summary_files") or []
    if summary_files:
        compare_args = [require_file(path, "retrieval summary file") for path in summary_files]
        if comparison.get("output_json"):
            output_json = resolve_path(comparison["output_json"])
            ensure_parent(output_json)
            compare_args.extend(["--output-json", output_json])
        print("Comparing retrieval evaluation summaries")
        run_script_main("retrieval/scripts/compare_retrieval_runs.py", compare_args)

    print(f"Stage 06 output prefix: {output_prefix}")


if __name__ == "__main__":
    main()

