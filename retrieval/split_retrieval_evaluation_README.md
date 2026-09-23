# Split Retrieval + Evaluation Codebase

## 1) Build index
```bash
python build_index.py retrieval_ready_enriched.jsonl index/cwe_index
```

## 2) Run retrieval once per configuration
```bash
python run_retrieval.py index/cwe_index benchmark.csv retrieval_runs/vector_v1.jsonl --method vector --top-k 10
python run_retrieval.py index/cwe_index benchmark.csv retrieval_runs/bm25_v1.jsonl --method bm25 --top-k 10
python run_retrieval.py index/cwe_index benchmark.csv retrieval_runs/hybrid_v1.jsonl --method hybrid --top-k 10 --vector-weight 0.7 --bm25-weight 0.3
```

## 3) Score retrieval runs separately
```bash
python score_retrieval_results.py benchmark.csv retrieval_runs/vector_v1.jsonl eval_runs/vector_v1
python score_retrieval_results.py benchmark.csv retrieval_runs/bm25_v1.jsonl eval_runs/bm25_v1
python score_retrieval_results.py benchmark.csv retrieval_runs/hybrid_v1.jsonl eval_runs/hybrid_v1
```

## 4) Compare runs
```bash
python compare_retrieval_runs.py eval_runs/vector_v1_summary.json eval_runs/bm25_v1_summary.json eval_runs/hybrid_v1_summary.json
```

## Why this split is better
- Retrieval runs are saved once and reused
- Evaluation can be changed without rerunning retrieval
- Multiple configurations can be compared cleanly
- No API-dependent retrieval needs to be rerun just to compute new metrics
