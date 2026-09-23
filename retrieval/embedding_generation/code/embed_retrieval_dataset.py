#!/usr/bin/env python3
from __future__ import annotations
"""
Embed a retrieval-ready CWE JSONL dataset using the OpenAI Embeddings API.

Usage:
    python embed_retrieval_dataset.py input.jsonl output_dir

Environment:
    OPENAI_API_KEY=...
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_DIMENSIONS=1024   # optional
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from openai import OpenAI

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def compact_list(values: Any, max_items: int = 8) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
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
        if len(out) >= max_items:
            break
    return out


def build_embedding_text(record: Dict[str, Any]) -> str:
    title = normalize_text(record.get("retrieval_title", "") or record.get("name", ""))
    retrieval_text = normalize_text(record.get("retrieval_text", ""))
    keywords = compact_list(record.get("retrieval_keywords"), max_items=15)
    prompts = compact_list(record.get("prompt_patterns"), max_items=8)
    guidance = compact_list(record.get("secure_coding_guidance"), max_items=5)

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if retrieval_text:
        parts.append(f"Body: {retrieval_text}")
    if keywords:
        parts.append("Keywords: " + "; ".join(keywords))
    if prompts:
        parts.append("Prompt patterns: " + "; ".join(prompts))
    if guidance:
        parts.append("Secure guidance: " + "; ".join(guidance))

    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(f"Record {record.get('doc_id', '<unknown>')} has no usable text to embed.")
    return text


def chunked(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    rows = load_jsonl(args.input_jsonl)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [build_embedding_text(r) for r in rows]
    client = OpenAI(api_key=api_key)

    dimensions = int(DEFAULT_DIMENSIONS) if DEFAULT_DIMENSIONS else None
    vectors: List[List[float]] = []

    processed = 0
    for batch in chunked(texts, args.batch_size):
        kwargs = {"model": DEFAULT_MODEL, "input": batch}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        vectors.extend([item.embedding for item in response.data])
        processed += len(batch)
        print(f"Embedded {processed}/{len(rows)} records")
        time.sleep(args.sleep)

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    metadata = []
    for row, text in zip(rows, texts):
        merged = dict(row)
        merged["embedding_text"] = text
        metadata.append(merged)

    np.save(output_dir / "embeddings.npy", matrix)
    dump_jsonl(output_dir / "metadata.jsonl", metadata)

    manifest = {
        "source_file": str(args.input_jsonl),
        "num_records": len(rows),
        "embedding_model": DEFAULT_MODEL,
        "dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else None,
        "normalized": True,
        "files": {
            "embeddings": "embeddings.npy",
            "metadata": "metadata.jsonl"
        }
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote embedding index to: {output_dir}")


if __name__ == "__main__":
    main()
