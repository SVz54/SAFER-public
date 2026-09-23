#!/usr/bin/env python3
'''
Run: 
python scripts/run_all.py --config configs/full_pipeline.yaml
'''
from __future__ import annotations

from pipeline_utils import load_config, parse_config_arg, require_file, run_script_main


STAGES = [
    ("01_clean_raw_data", "scripts/run_01_clean_raw_data.py"),
    ("02_create_retrieval_ready_data", "scripts/run_02_create_retrieval_ready_data.py"),
    ("03_generate_embeddings", "scripts/run_03_generate_embeddings.py"),
    ("04_run_retrieval", "scripts/run_04_run_retrieval.py"),
    ("05_generate_code", "scripts/run_05_generate_code.py"),
    ("06_evaluate_outputs", "scripts/run_06_evaluate_outputs.py"),
    ("07_run_static_analysis", "scripts/run_07_run_static_analysis.py"),
]


def main() -> None:
    args = parse_config_arg("Run the thesis pipeline stages in order.")
    config = load_config(args.config)
    stages = config.get("stages") or {}
    if not isinstance(stages, dict):
        raise ValueError("full_pipeline.yaml must contain a 'stages' mapping")

    for stage_name, runner_path in STAGES:
        stage_config = stages.get(stage_name) or {}
        if not isinstance(stage_config, dict):
            raise ValueError(f"Stage config for {stage_name} must be a mapping")

        enabled = bool(stage_config.get("enabled", True))
        if not enabled:
            print(f"Skipping {stage_name}: disabled")
            continue

        config_path = require_file(stage_config.get("config"), f"{stage_name} config")
        print(f"\n=== Running {stage_name} ===")
        run_script_main(runner_path, ["--config", config_path])

    print("\nFull pipeline runner finished.")


if __name__ == "__main__":
    main()

