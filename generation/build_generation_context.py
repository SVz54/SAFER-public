#!/usr/bin/env python3
from __future__ import annotations

"""
Build compact generation context from retrieval outputs and CWE metadata.

Usage:
    python generation/build_generation_context.py retrieval_results.jsonl metadata.jsonl generation_context.jsonl

Example:
    python generation/build_generation_context.py \
      retrieval/embedding_generation/code/retrieval_runs/hybrid_v1.jsonl \
      retrieval/embedding_generation/code/embedding_files/metadata.jsonl \
      generation/generation_context.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CWE_RE = re.compile(r"(?:CWE-)?(\d+)", re.IGNORECASE)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} in {path} must be a JSON object.")
            rows.append(obj)
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_string_list(values: Any, max_items: int) -> List[str]:
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


def build_metadata_lookup(metadata_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in metadata_rows:
        cwe_id = normalize_cwe(row.get("cwe_id") or row.get("doc_id"))
        if not cwe_id:
            continue
        if cwe_id not in lookup:
            lookup[cwe_id] = row
    return lookup


def build_context_item(
    *,
    rank: int,
    cwe_id: str,
    metadata: Dict[str, Any],
    max_guidance: int,
    max_consequences: int,
    max_examples: int,
    include_task_tags: bool,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "rank": rank,
        "cwe_id": cwe_id,
        "name": normalize_text(metadata.get("name", "")),
        "weakness_summary": normalize_text(metadata.get("weakness_summary", "")),
        "developer_risk_summary": normalize_text(metadata.get("developer_risk_summary", "")),
        "secure_coding_guidance": compact_string_list(metadata.get("secure_coding_guidance"), max_guidance),
        "consequence_keywords": compact_string_list(metadata.get("consequence_keywords"), max_consequences),
        "example_signals": compact_string_list(metadata.get("example_signals"), max_examples),
    }

    if include_task_tags:
        item["task_tags"] = compact_string_list(metadata.get("task_tags"), max_items=12)

    return item


def build_generation_context(
    *,
    retrieval_rows: List[Dict[str, Any]],
    metadata_lookup: Dict[str, Dict[str, Any]],
    top_k: int,
    max_guidance: int,
    max_consequences: int,
    max_examples: int,
    include_task_tags: bool,
) -> tuple[List[Dict[str, Any]], int]:
    out_rows: List[Dict[str, Any]] = []
    missing_count = 0

    for row in retrieval_rows:
        query_id = normalize_text(row.get("query_id"))
        prompt = normalize_text(row.get("prompt"))
        ranked = row.get("ranked_results", [])

        if not isinstance(ranked, list):
            ranked = []

        retrieved_context: List[Dict[str, Any]] = []
        seen_cwes = set()

        for result in ranked:
            if len(retrieved_context) >= top_k:
                break
            if not isinstance(result, dict):
                continue

            cwe_id = normalize_cwe(result.get("cwe_id"))
            if not cwe_id:
                continue
            key = cwe_id.lower()
            if key in seen_cwes:
                continue
            seen_cwes.add(key)

            metadata = metadata_lookup.get(cwe_id)
            if metadata is None:
                missing_count += 1
                continue

            context_item = build_context_item(
                rank=len(retrieved_context) + 1,
                cwe_id=cwe_id,
                metadata=metadata,
                max_guidance=max_guidance,
                max_consequences=max_consequences,
                max_examples=max_examples,
                include_task_tags=include_task_tags,
            )
            retrieved_context.append(context_item)

        out_rows.append(
            {
                "query_id": query_id,
                "prompt": prompt,
                "retrieved_context": retrieved_context,
            }
        )

    return out_rows, missing_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact generation context from retrieval outputs.")
    parser.add_argument("retrieval_results_jsonl", type=Path, help="Retrieval run output JSONL")
    parser.add_argument("metadata_jsonl", type=Path, help="Metadata JSONL from embedding index")
    parser.add_argument("output_jsonl", type=Path, help="Output generation context JSONL")
    parser.add_argument("--top-k", type=int, default=5, help="Max number of CWE context items per query")
    parser.add_argument("--max-guidance", type=int, default=4, help="Max secure coding guidance items per CWE")
    parser.add_argument("--max-consequences", type=int, default=8, help="Max consequence keywords per CWE")
    parser.add_argument("--max-examples", type=int, default=4, help="Max example signals per CWE")
    parser.add_argument("--include-task-tags", action="store_true", help="Include task_tags in each CWE context item")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    retrieval_rows = load_jsonl(args.retrieval_results_jsonl)
    metadata_rows = load_jsonl(args.metadata_jsonl)
    metadata_lookup = build_metadata_lookup(metadata_rows)

    output_rows, missing_count = build_generation_context(
        retrieval_rows=retrieval_rows,
        metadata_lookup=metadata_lookup,
        top_k=args.top_k,
        max_guidance=max(1, args.max_guidance),
        max_consequences=max(1, args.max_consequences),
        max_examples=max(1, args.max_examples),
        include_task_tags=args.include_task_tags,
    )

    dump_jsonl(args.output_jsonl, output_rows)

    print(f"Queries processed: {len(retrieval_rows)}")
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Metadata misses: {missing_count}")
    print(f"Wrote generation context: {args.output_jsonl}")


if __name__ == "__main__":
    main()

