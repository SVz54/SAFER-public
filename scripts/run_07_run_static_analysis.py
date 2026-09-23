#!/usr/bin/env python3

'''
Run:
python scripts/run_07_run_static_analysis.py --config configs/07_run_static_analysis.yaml
'''
from __future__ import annotations

from pipeline_utils import load_config, parse_config_arg


def main() -> None:
    args = parse_config_arg("Stage 07: run static/security analysis.")
    config = load_config(args.config)

    if not bool(config.get("enabled", False)):
        print("Stage 07 is disabled.")
        print("TODO: No static/security analysis implementation exists in this repository yet.")
        print("Add a real scanner stage, then set enabled: true in the config.")
        return

    raise NotImplementedError(
        "Static/security analysis is not wired because the repository does not currently "
        "contain scanner logic such as Bandit, Semgrep, CodeQL, or a custom evaluator."
    )


if __name__ == "__main__":
    main()

