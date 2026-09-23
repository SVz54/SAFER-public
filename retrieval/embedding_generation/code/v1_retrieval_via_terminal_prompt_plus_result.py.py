#!/usr/bin/env python3
from __future__ import annotations
"""
Hybrid retrieval over the embedded CWE retrieval dataset.

Usage:
    python hybrid_retriever.py index_dir "build a file upload api"
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+\-/.]{2,}")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


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


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def build_lexical_document(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ["name", "retrieval_title", "retrieval_text", "weakness_summary", "developer_risk_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    for key in ["alternate_terms", "retrieval_keywords", "prompt_patterns", "secure_coding_guidance", "task_tags", "consequence_keywords"]:
        parts.extend(compact_list(record.get(key)))
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
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            doc_freqs.append(freqs)
            for term in freqs:
                df[term] = df.get(term, 0) + 1
        n_docs = max(len(tokenized_docs), 1)
        avgdl = total_len / n_docs if n_docs else 0.0
        idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        return cls(tokenized_docs, doc_freqs, idf, avgdl)

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


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    kwargs = {"model": DEFAULT_MODEL, "input": query}
    if DEFAULT_DIMENSIONS:
        kwargs["dimensions"] = int(DEFAULT_DIMENSIONS)
    response = client.embeddings.create(**kwargs)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def hybrid_search(metadata, embeddings, bm25, query, client, top_k=5, vector_weight=0.7, bm25_weight=0.3):
    q_vec = embed_query(client, query)
    vector_scores = embeddings @ q_vec
    bm25_scores = bm25.score(query)

    vector_scaled = minmax_scale(vector_scores)
    bm25_scaled = minmax_scale(bm25_scores)
    hybrid_scores = vector_weight * vector_scaled + bm25_weight * bm25_scaled

    top_idx = np.argsort(-hybrid_scores)[:top_k]
    results = []
    for idx in top_idx:
        record = metadata[int(idx)]
        results.append({
            "rank": len(results) + 1,
            "doc_id": record.get("doc_id"),
            "cwe_id": record.get("cwe_id"),
            "name": record.get("name"),
            "retrieval_title": record.get("retrieval_title"),
            "vector_score": float(vector_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
            "task_tags": record.get("task_tags", []),
            "secure_coding_guidance": record.get("secure_coding_guidance", [])[:4],
            "retrieval_text": record.get("retrieval_text", ""),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    metadata = load_jsonl(args.index_dir / "metadata.jsonl")
    embeddings = np.load(args.index_dir / "embeddings.npy")
    docs = [build_lexical_document(record) for record in metadata]
    bm25 = BM25Index.build(docs)

    client = OpenAI(api_key=api_key)
    results = hybrid_search(
        metadata=metadata,
        embeddings=embeddings,
        bm25=bm25,
        query=args.query,
        client=client,
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"Query: {args.query}\n")
    for result in results:
        print(f"[{result['rank']}] {result['retrieval_title']} ({result['hybrid_score']:.4f})")
        print(f"    vector={result['vector_score']:.4f}  bm25={result['bm25_score']:.4f}")
        if result["task_tags"]:
            print(f"    tags: {', '.join(result['task_tags'])}")
        guidance = result.get("secure_coding_guidance", [])
        if guidance:
            print("    guidance:")
            for item in guidance:
                print(f"      - {item}")
        snippet = normalize_text(result.get("retrieval_text", ""))[:280]
        if snippet:
            print(f"    snippet: {snippet}...")
        print()


if __name__ == "__main__":
    main()
