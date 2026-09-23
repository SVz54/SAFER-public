#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate retrieval performance for the CWE hybrid retriever.

Usage:
    python evaluate_retrieval.py index_dir benchmark.csv
    python evaluate_retrieval.py index_dir benchmark.csv --top-k 10 --output-dir eval_out
"""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+\-/.]{2,}")
CWE_RE = re.compile(r"(?:CWE-)?(\d+)", re.IGNORECASE)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")


def normalize_text(text: Any) -> str:
    return " ".join(str(text).split()).strip()


def normalize_cwe(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = normalize_text(value)
    if not text:
        return None
    match = CWE_RE.search(text)
    if not match:
        return None
    return f"CWE-{match.group(1)}"


def parse_cwe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = normalize_text(value)
        if not text:
            return []
        raw_items = re.split(r"[|,;/]+", text)

    out: List[str] = []
    seen = set()
    for item in raw_items:
        cwe = normalize_cwe(item)
        if not cwe:
            continue
        key = cwe.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cwe)
    return out


def compact_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = normalize_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def build_lexical_document(record: Dict[str, Any]) -> str:
    parts: List[str] = []

    def add_text(value: str, repeat: int = 1) -> None:
        value = normalize_text(value)
        if not value:
            return
        for _ in range(repeat):
            parts.append(value)

    def add_list(values: Any, repeat: int = 1) -> None:
        for item in compact_list(values):
            add_text(item, repeat=repeat)

    add_text(record.get("name", ""), repeat=3)
    add_text(record.get("retrieval_title", ""), repeat=2)
    add_list(record.get("alternate_terms"), repeat=2)
    add_list(record.get("retrieval_keywords"), repeat=3)
    add_list(record.get("prompt_patterns"), repeat=3)
    add_list(record.get("task_tags"), repeat=2)
    add_list(record.get("consequence_keywords"), repeat=2)
    add_list(record.get("secure_coding_guidance"), repeat=1)
    add_text(record.get("weakness_summary", ""), repeat=1)
    add_text(record.get("developer_risk_summary", ""), repeat=1)
    add_text(record.get("retrieval_text", ""), repeat=1)

    return "\n".join(parts)


@dataclass
class BM25Index:
    tokenized_docs: List[List[str]]
    doc_freqs: List[Dict[str, int]]
    idf: Dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: List[str]) -> "BM25Index":
        tokenized_docs = [tokenize(doc) for doc in docs]
        doc_freqs: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        total_len = 0

        for tokens in tokenized_docs:
            total_len += len(tokens)
            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1

        n_docs = max(len(tokenized_docs), 1)
        avgdl = total_len / n_docs if n_docs else 0.0
        idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return cls(tokenized_docs=tokenized_docs, doc_freqs=doc_freqs, idf=idf, avgdl=avgdl)

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.zeros(len(self.tokenized_docs), dtype=np.float32)

        for i, freqs in enumerate(self.doc_freqs):
            doc_len = len(self.tokenized_docs[i]) or 1
            score = 0.0
            for term in q_tokens:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = score
        return scores


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_v = float(scores.min())
    max_v = float(scores.max())
    if abs(max_v - min_v) < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


PROMPT_CANDIDATES = ["prompt", "prompt_text", "query", "question"]
PRIMARY_CANDIDATES = ["expected_cwe_primary", "cwe_primary", "primary_cwe", "expected_primary_cwe"]
SECONDARY_CANDIDATES = ["expected_cwe_secondary", "cwe_secondary", "secondary_cwe", "expected_secondary_cwe"]
DIFFICULTY_CANDIDATES = ["difficulty", "level"]
TASK_TYPE_CANDIDATES = ["task_type", "task", "category"]


def first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] not in ("", None):
            return row[key]
    return None


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        raw_rows = load_jsonl(path)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            raw_rows = list(csv.DictReader(f))
    else:
        raise ValueError("Benchmark file must be .csv or .jsonl")

    benchmark = []
    for i, row in enumerate(raw_rows, start=1):
        prompt = normalize_text(first_present(row, PROMPT_CANDIDATES))
        if not prompt:
            raise ValueError(f"Row {i} missing prompt field. Tried: {PROMPT_CANDIDATES}")

        primary = parse_cwe_list(first_present(row, PRIMARY_CANDIDATES))
        secondary = parse_cwe_list(first_present(row, SECONDARY_CANDIDATES))

        expected_all: List[str] = []
        seen = set()
        for cwe in primary + secondary:
            key = cwe.lower()
            if key in seen:
                continue
            seen.add(key)
            expected_all.append(cwe)

        benchmark.append({
            "row_id": i,
            "prompt": prompt,
            "expected_cwe_primary": primary,
            "expected_cwe_secondary": secondary,
            "expected_cwes_all": expected_all,
            "difficulty": normalize_text(first_present(row, DIFFICULTY_CANDIDATES) or ""),
            "task_type": normalize_text(first_present(row, TASK_TYPE_CANDIDATES) or ""),
            "raw_row": row,
        })
    return benchmark


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    kwargs = {"model": DEFAULT_MODEL, "input": query}
    if DEFAULT_DIMENSIONS:
        kwargs["dimensions"] = int(DEFAULT_DIMENSIONS)
    response = client.embeddings.create(**kwargs)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def score_query(query: str, client: OpenAI, embeddings: np.ndarray, bm25: BM25Index) -> Tuple[np.ndarray, np.ndarray]:
    q_vec = embed_query(client, query)
    vector_scores = embeddings @ q_vec
    bm25_scores = bm25.score(query)
    return vector_scores, bm25_scores


def rank_indices(vector_scores: np.ndarray, bm25_scores: np.ndarray, mode: str, vector_weight: float, bm25_weight: float) -> np.ndarray:
    if mode == "vector":
        return np.argsort(-vector_scores)
    if mode == "bm25":
        return np.argsort(-bm25_scores)
    if mode == "hybrid":
        vector_scaled = minmax_scale(vector_scores)
        bm25_scaled = minmax_scale(bm25_scores)
        hybrid_scores = vector_weight * vector_scaled + bm25_weight * bm25_scaled
        return np.argsort(-hybrid_scores)
    raise ValueError(f"Unknown mode: {mode}")


def hit_at_k(ranked_cwes: List[str], relevant: List[str], k: int) -> int:
    top = set(ranked_cwes[:k])
    return int(any(cwe in top for cwe in relevant))


def recall_at_k(ranked_cwes: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = set(ranked_cwes[:k]).intersection(set(relevant))
    return len(found) / len(set(relevant))


def reciprocal_rank(ranked_cwes: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, cwe in enumerate(ranked_cwes, start=1):
        if cwe in relevant_set:
            return 1.0 / i
    return 0.0


def first_relevant_rank(ranked_cwes: List[str], relevant: List[str]) -> Optional[int]:
    relevant_set = set(relevant)
    for i, cwe in enumerate(ranked_cwes, start=1):
        if cwe in relevant_set:
            return i
    return None


def aggregate_metrics(rows: List[Dict[str, Any]], top_k_values: List[int]) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"num_queries": n, "mrr": 0.0}
    if n == 0:
        for k in top_k_values:
            out[f"hit@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
        return out

    out["mrr"] = sum(row["mrr"] for row in rows) / n
    for k in top_k_values:
        out[f"hit@{k}"] = sum(row[f"hit@{k}"] for row in rows) / n
        out[f"recall@{k}"] = sum(row[f"recall@{k}"] for row in rows) / n
    return out


def group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = normalize_text(row.get(key, "")) or "UNSPECIFIED"
        groups.setdefault(value, []).append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("benchmark_file", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if abs((args.vector_weight + args.bm25_weight) - 1.0) > 1e-9:
        raise ValueError("vector-weight and bm25-weight must sum to 1.0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    metadata = load_jsonl(args.index_dir / "metadata.jsonl")
    embeddings = np.load(args.index_dir / "embeddings.npy")
    lexical_docs = [build_lexical_document(record) for record in metadata]
    bm25 = BM25Index.build(lexical_docs)
    client = OpenAI(api_key=api_key)

    benchmark = load_benchmark(args.benchmark_file)
    top_k_values = sorted(set([1, 3, 5, args.top_k]))

    modes = ["vector", "bm25", "hybrid"]
    detailed_results: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}

    for idx, item in enumerate(benchmark, start=1):
        prompt = item["prompt"]
        expected = item["expected_cwes_all"]
        vector_scores, bm25_scores = score_query(prompt, client, embeddings, bm25)

        for mode in modes:
            ranked_idx = rank_indices(
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                mode=mode,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
            )

            ranked_cwes = [normalize_cwe(metadata[int(i)].get("cwe_id")) for i in ranked_idx]
            ranked_cwes = [c for c in ranked_cwes if c]
            row = {
                "row_id": item["row_id"],
                "prompt": prompt,
                "difficulty": item["difficulty"],
                "task_type": item["task_type"],
                "expected_cwe_primary": item["expected_cwe_primary"],
                "expected_cwe_secondary": item["expected_cwe_secondary"],
                "expected_cwes_all": expected,
                "first_relevant_rank": first_relevant_rank(ranked_cwes, expected),
                "mrr": reciprocal_rank(ranked_cwes, expected),
                "top_results": [],
            }

            for k in top_k_values:
                row[f"hit@{k}"] = hit_at_k(ranked_cwes, expected, k)
                row[f"recall@{k}"] = recall_at_k(ranked_cwes, expected, k)

            for rank, rec_idx in enumerate(ranked_idx[:args.top_k], start=1):
                record = metadata[int(rec_idx)]
                row["top_results"].append({
                    "rank": rank,
                    "cwe_id": normalize_cwe(record.get("cwe_id")),
                    "name": record.get("name"),
                    "retrieval_title": record.get("retrieval_title"),
                })

            detailed_results[mode].append(row)

        print(f"Evaluated {idx}/{len(benchmark)} queries")

    summary: Dict[str, Any] = {
        "settings": {
            "index_dir": str(args.index_dir),
            "benchmark_file": str(args.benchmark_file),
            "embedding_model": DEFAULT_MODEL,
            "embedding_dimensions": int(DEFAULT_DIMENSIONS) if DEFAULT_DIMENSIONS else None,
            "top_k": args.top_k,
            "vector_weight": args.vector_weight,
            "bm25_weight": args.bm25_weight,
        },
        "overall": {},
        "by_difficulty": {},
        "by_task_type": {},
    }

    for mode in modes:
        rows = detailed_results[mode]
        summary["overall"][mode] = aggregate_metrics(rows, top_k_values)

        diff_groups = group_rows(rows, "difficulty")
        summary["by_difficulty"][mode] = {
            group: aggregate_metrics(group_rows_list, top_k_values)
            for group, group_rows_list in diff_groups.items()
        }

        task_groups = group_rows(rows, "task_type")
        summary["by_task_type"][mode] = {
            group: aggregate_metrics(group_rows_list, top_k_values)
            for group, group_rows_list in task_groups.items()
        }

    print("\n=== Overall metrics ===")
    for mode in modes:
        metrics = summary["overall"][mode]
        print(f"\n[{mode}]")
        print(f"queries:   {metrics['num_queries']}")
        print(f"MRR:       {metrics['mrr']:.4f}")
        for k in top_k_values:
            print(f"Hit@{k}:    {metrics[f'hit@{k}']:.4f}")
            print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for mode in modes:
            dump_jsonl(args.output_dir / f"{mode}_detailed.jsonl", detailed_results[mode])
        print(f"\nWrote evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
