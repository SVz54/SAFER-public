# Pipeline Structure

The project is organized as a sequence of small stages. The new `scripts/` runners provide consistent commands, while the original project logic remains in `data_curation/`, `retrieval/`, and `generation/`.

## 1. Raw CWE XML Cleaning

Input: original MITRE CWE XML.

Logic: `data_curation/raw_data_cleaning/clean_cwe_xml.py`

Output: reduced cleaned XML containing the CWE fields needed for retrieval and generation.

## 2. Retrieval-Ready Data Creation

Input: cleaned CWE XML.

Logic:

- `data_curation/retreival_ready_data/build_canonical_cwe_json.py`
- `data_curation/retreival_ready_data/build_retrieval_dataset.py`
- optional: `data_curation/retreival_ready_data/enrich_retrieval_with_openai.py`

Output: canonical CWE JSON and retrieval-ready JSONL records.

## 3. Embedding and Index Creation

Input: retrieval-ready JSONL.

Logic: `retrieval/scripts/build_index.py`

Output: embedding matrix, metadata JSONL, and manifest JSON.

## 4. Retrieval

Input: retrieval index plus prompt/benchmark file.

Logic: `retrieval/scripts/run_retrieval.py`

Output: JSONL retrieval run with ranked CWE results per query.

## 5. Code Generation

Input: retrieval results and retrieval metadata.

Logic:

- `generation/build_generation_context.py`
- `generation/build_generation_prompts.py`
- `generation/run_generation.py`

Output: generation context JSONL, generation prompt JSONL, generated-code output JSONL, and optional code artifacts.

## 6. Evaluation

Input: benchmark file and retrieval run.

Logic:

- `retrieval/scripts/score_retrieval_results.py`
- optional: `retrieval/scripts/compare_retrieval_runs.py`

Output: retrieval metric summary and detailed per-query scoring.

## 7. Static/Security Analysis

No implemented scanner stage is currently present. The runner exists only as a clear placeholder so the pipeline shape is explicit.

