# Retrieval Evaluation Script

## What it does
Evaluates:
- vector retrieval
- BM25 lexical retrieval
- hybrid retrieval

against a benchmark file containing prompts and expected CWE labels.

## Supported benchmark columns
Required prompt column (any one):
- prompt
- prompt_text
- query
- question

Expected label columns:
- expected_cwe_primary
- expected_cwe_secondary

Optional:
- difficulty
- task_type

Secondary CWE values can be pipe- or comma-delimited, for example:
- CWE-22|CWE-73
- CWE-22, CWE-73

## Example
```bash
python evaluate_retrieval.py cwe_index benchmark.csv --top-k 10 --output-dir eval_out
```

## Outputs
If `--output-dir` is provided, the script writes:
- `summary.json`
- `vector_detailed.jsonl`
- `bm25_detailed.jsonl`
- `hybrid_detailed.jsonl`

## Metrics
- Hit@k
- Recall@k
- MRR

## Why this matters
This lets you verify:
- whether dense embeddings help
- whether lexical retrieval helps
- whether hybrid retrieval actually improves over either alone
- whether performance changes by difficulty or task type
