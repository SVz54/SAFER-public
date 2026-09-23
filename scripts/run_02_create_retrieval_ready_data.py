#!/usr/bin/env python3
'''
Run:
python scripts/run_02_create_retrieval_ready_data.py --config configs/02_create_retrieval_ready_data.yaml
'''
from __future__ import annotations

from pipeline_utils import (
    ensure_parent,
    load_config,
    optional_path,
    parse_config_arg,
    require_file,
    require_keys,
    resolve_path,
    run_script_main,
)


def main() -> None:
    args = parse_config_arg("Stage 02: create retrieval-ready CWE data.")
    config = load_config(args.config)
    require_keys(config, ["input_xml", "retrieval_ready_output"])

    input_xml = require_file(config["input_xml"], "cleaned or raw CWE XML input")
    retrieval_ready_output = resolve_path(config["retrieval_ready_output"])
    ensure_parent(retrieval_ready_output)

    canonical_output = optional_path(config.get("canonical_output"))
    if canonical_output is not None:
        ensure_parent(canonical_output)
        canonical_args = [input_xml, canonical_output]
        if bool(config.get("canonical_jsonl", False)):
            canonical_args.append("--jsonl")
        print(f"Writing canonical CWE records: {canonical_output}")
        run_script_main(
            "data_curation/retreival_ready_data/build_canonical_cwe_json.py",
            canonical_args,
        )

    print(f"Writing deterministic retrieval-ready records: {retrieval_ready_output}")
    run_script_main(
        "data_curation/retreival_ready_data/build_retrieval_dataset.py",
        [input_xml, retrieval_ready_output],
    )

    enrichment = config.get("openai_enrichment") or {}
    if bool(enrichment.get("enabled", False)):
        enriched_output = resolve_path(enrichment.get("output_file"))
        ensure_parent(enriched_output)
        enrich_args = [
            retrieval_ready_output,
            enriched_output,
            "--model",
            enrichment.get("model", "gpt-4.1"),
            "--delay",
            enrichment.get("delay_seconds", 0.25),
            "--max-retries",
            enrichment.get("max_retries", 3),
            "--start-index",
            enrichment.get("start_index", 1),
            "--max-records",
            enrichment.get("max_records", 0),
        ]
        print(f"Running optional OpenAI retrieval enrichment: {enriched_output}")
        run_script_main(
            "data_curation/retreival_ready_data/enrich_retrieval_with_openai.py",
            enrich_args,
        )
        print(f"Stage 02 enriched output: {enriched_output}")
    else:
        print(f"Stage 02 output: {retrieval_ready_output}")


if __name__ == "__main__":
    main()

