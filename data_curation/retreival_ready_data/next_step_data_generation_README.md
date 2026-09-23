# Next step in data generation

You now have a 2-layer retrieval-data pipeline.

## Layer 1: baseline deterministic generation
`build_retrieval_dataset.py`

This:
- extracts source fields from CWE XML
- builds baseline summaries, tags, keywords, prompt patterns, and retrieval text
- gives you a stable retrieval-ready JSONL file

## Layer 2: OpenAI enrichment
`enrich_retrieval_with_openai.py`

This:
- reads the baseline JSONL
- keeps all source-derived fields unchanged
- rewrites only the retrieval-facing fields using the OpenAI API
- makes summaries, keywords, prompt patterns, guidance, and retrieval text stronger

## Example workflow

1. Clean XML
```bash
python clean_cwe_xml.py 1435.xml cleaned_1435.xml
```

2. Build baseline retrieval dataset
```bash
python build_retrieval_dataset.py cleaned_1435.xml retrieval_ready_baseline.jsonl
```

3. Enrich with OpenAI
```bash
python enrich_retrieval_with_openai.py retrieval_ready_baseline.jsonl retrieval_ready_enriched.jsonl
```

## Hybrid retrieval later
This stage still prepares the data only.

Later:
- vector retrieval can use `retrieval_text`
- lexical / BM25 retrieval can use `retrieval_keywords`, `prompt_patterns`, `name`, `alternate_terms`
- reranking can use `task_tags`, `consequence_keywords`, `secure_coding_guidance`

python data_curation/retreival_ready_data/enrich_retrieval_with_openai.py \
data_curation/retreival_ready_data/generated_data/retrieval_ready_baseline.jsonl \
data_curation/retreival_ready_data/generated_data/retrieval_ready_enriched.jsonl