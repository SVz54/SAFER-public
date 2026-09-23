#!/usr/bin/env python3

'''
Run:
python scripts/run_01_clean_raw_data.py --config configs/01_clean_raw_data.yaml
'''
from __future__ import annotations

from pipeline_utils import ensure_parent, load_config, parse_config_arg, require_file, require_keys, resolve_path, run_script_main


def main() -> None:
    args = parse_config_arg("Stage 01: clean raw CWE XML into reduced XML.")
    config = load_config(args.config)
    require_keys(config, ["input_xml", "output_xml"])

    input_xml = require_file(config["input_xml"], "raw CWE XML input")
    output_xml = resolve_path(config["output_xml"])
    ensure_parent(output_xml)

    print(f"Cleaning raw CWE XML: {input_xml}")
    run_script_main(
        "data_curation/raw_data_cleaning/clean_cwe_xml.py",
        [input_xml, output_xml],
    )
    print(f"Stage 01 output: {output_xml}")


if __name__ == "__main__":
    main()

