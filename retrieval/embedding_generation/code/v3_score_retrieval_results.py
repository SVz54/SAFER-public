#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

CWE_RE = re.compile(r"(?:CWE-)?(\d+)", re.IGNORECASE)
PROMPT_CANDIDATES=["prompt","prompt_text","query","question"]
QUERY_ID_CANDIDATES=["query_id","id","prompt_id"]
PRIMARY_CANDIDATES=["expected_cwe_primary","cwe_primary","primary_cwe","expected_primary_cwe"]
SECONDARY_CANDIDATES=["expected_cwe_secondary","cwe_secondary","secondary_cwe","expected_secondary_cwe"]
DIFFICULTY_CANDIDATES=["difficulty","level"]
TASK_TYPE_CANDIDATES=["task_type","task","category"]

def normalize_text(text: Any) -> str: return " ".join(str(text).split()).strip()
def normalize_cwe(value: Any) -> Optional[str]:
    if value is None: return None
    text=normalize_text(value)
    if not text: return None
    m=CWE_RE.search(text)
    return f"CWE-{m.group(1)}" if m else None
def parse_cwe_list(value: Any) -> List[str]:
    if value is None: return []
    raw_items = value if isinstance(value, list) else re.split(r"[|,;/]+", normalize_text(value))
    out=[]; seen=set()
    for item in raw_items:
        cwe=normalize_cwe(item)
        if not cwe: continue
        key=cwe.lower()
        if key in seen: continue
        seen.add(key); out.append(cwe)
    return out
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows
def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
def first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] not in ("", None): return row[key]
    return None
def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    suffix=path.suffix.lower()
    if suffix==".jsonl": raw_rows=load_jsonl(path)
    elif suffix==".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f: raw_rows=list(csv.DictReader(f))
    else: raise ValueError("Benchmark file must be .csv or .jsonl")
    rows=[]
    for i,row in enumerate(raw_rows, start=1):
        prompt=normalize_text(first_present(row, PROMPT_CANDIDATES))
        if not prompt: raise ValueError(f"Row {i} missing prompt field.")
        query_id=normalize_text(first_present(row, QUERY_ID_CANDIDATES) or f"Q{i:03d}")
        primary=parse_cwe_list(first_present(row, PRIMARY_CANDIDATES))
        secondary=parse_cwe_list(first_present(row, SECONDARY_CANDIDATES))
        expected_all=[]; seen=set()
        for cwe in primary+secondary:
            key=cwe.lower()
            if key in seen: continue
            seen.add(key); expected_all.append(cwe)
        rows.append({"query_id": query_id, "prompt": prompt, "expected_cwe_primary": primary,
                     "expected_cwe_secondary": secondary, "expected_cwes_all": expected_all,
                     "difficulty": normalize_text(first_present(row, DIFFICULTY_CANDIDATES) or ""),
                     "task_type": normalize_text(first_present(row, TASK_TYPE_CANDIDATES) or "")})
    return rows
def hit_at_k(ranked_cwes: List[str], relevant: List[str], k: int) -> int:
    return int(any(cwe in set(ranked_cwes[:k]) for cwe in relevant))
def recall_at_k(ranked_cwes: List[str], relevant: List[str], k: int) -> float:
    if not relevant: return 0.0
    found=set(ranked_cwes[:k]).intersection(set(relevant))
    return len(found)/len(set(relevant))
def reciprocal_rank(ranked_cwes: List[str], relevant: List[str]) -> float:
    relevant_set=set(relevant)
    for i,cwe in enumerate(ranked_cwes, start=1):
        if cwe in relevant_set: return 1.0/i
    return 0.0
def first_relevant_rank(ranked_cwes: List[str], relevant: List[str]) -> Optional[int]:
    relevant_set=set(relevant)
    for i,cwe in enumerate(ranked_cwes, start=1):
        if cwe in relevant_set: return i
    return None
def aggregate_metrics(rows: List[Dict[str, Any]], top_k_values: List[int]) -> Dict[str, Any]:
    n=len(rows); out={"num_queries": n, "mrr": 0.0}
    if n==0:
        for k in top_k_values:
            out[f"hit@{k}"]=0.0; out[f"recall@{k}"]=0.0
        return out
    out["mrr"]=sum(row["mrr"] for row in rows)/n
    for k in top_k_values:
        out[f"hit@{k}"]=sum(row[f"hit@{k}"] for row in rows)/n
        out[f"recall@{k}"]=sum(row[f"recall@{k}"] for row in rows)/n
    return out
def group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups={}
    for row in rows:
        value=normalize_text(row.get(key,"")) or "UNSPECIFIED"
        groups.setdefault(value, []).append(row)
    return groups
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("benchmark_file", type=Path)
    parser.add_argument("retrieval_results_jsonl", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args=parser.parse_args()
    benchmark=load_benchmark(args.benchmark_file)
    truth_by_query={row["query_id"]: row for row in benchmark}
    retrieval_rows=load_jsonl(args.retrieval_results_jsonl)
    if not retrieval_rows: raise ValueError("No retrieval rows found.")
    top_k_max=max(len(row.get("ranked_results", [])) for row in retrieval_rows)
    top_k_values=sorted(set([1,3,5,top_k_max]))
    detailed=[]
    for row in retrieval_rows:
        query_id=normalize_text(row.get("query_id"))
        if query_id not in truth_by_query: raise ValueError(f"query_id not found in benchmark: {query_id}")
        truth=truth_by_query[query_id]
        ranked_cwes=[normalize_cwe(item.get("cwe_id")) for item in row.get("ranked_results",[])]
        ranked_cwes=[c for c in ranked_cwes if c]
        expected=truth["expected_cwes_all"]
        scored={"query_id": query_id, "prompt": truth["prompt"], "method": row.get("method"), "run_name": row.get("run_name"),
                "difficulty": truth["difficulty"], "task_type": truth["task_type"],
                "expected_cwe_primary": truth["expected_cwe_primary"], "expected_cwe_secondary": truth["expected_cwe_secondary"],
                "expected_cwes_all": expected, "first_relevant_rank": first_relevant_rank(ranked_cwes, expected),
                "mrr": reciprocal_rank(ranked_cwes, expected), "top_results": row.get("ranked_results",[])}
        for k in top_k_values:
            scored[f"hit@{k}"]=hit_at_k(ranked_cwes, expected, k)
            scored[f"recall@{k}"]=recall_at_k(ranked_cwes, expected, k)
        detailed.append(scored)
    summary={"method": retrieval_rows[0].get("method"), "run_name": retrieval_rows[0].get("run_name"),
             "overall": aggregate_metrics(detailed, top_k_values),
             "by_difficulty": {g: aggregate_metrics(rows, top_k_values) for g,rows in group_rows(detailed,"difficulty").items()},
             "by_task_type": {g: aggregate_metrics(rows, top_k_values) for g,rows in group_rows(detailed,"task_type").items()}}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path=args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    detailed_path=args.output_prefix.with_name(args.output_prefix.name + "_detailed.jsonl")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dump_jsonl(detailed_path, detailed)
    print(f"Wrote summary:  {summary_path}")
    print(f"Wrote detailed: {detailed_path}")
if __name__=="__main__":
    main()
