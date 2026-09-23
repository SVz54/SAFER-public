#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List

def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("summary_files", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args=parser.parse_args()
    rows: List[Dict[str, Any]]=[]
    for path in args.summary_files:
        summary=load_summary(path); overall=summary.get("overall", {})
        rows.append({"file": str(path), "run_name": summary.get("run_name"), "method": summary.get("method"),
                     "num_queries": overall.get("num_queries"), "mrr": overall.get("mrr"),
                     "hit@1": overall.get("hit@1"), "hit@3": overall.get("hit@3"),
                     "hit@5": overall.get("hit@5"), "hit@10": overall.get("hit@10"),
                     "recall@5": overall.get("recall@5"), "recall@10": overall.get("recall@10")})
    rows_sorted=sorted(rows, key=lambda r: (r["mrr"] if r["mrr"] is not None else -1), reverse=True)
    headers=["run_name","method","mrr","hit@1","hit@3","hit@5","hit@10","recall@5","recall@10","file"]
    widths={h: max(len(h), max(len(fmt(row.get(h))) for row in rows_sorted)) for h in headers}
    line=" | ".join(h.ljust(widths[h]) for h in headers)
    sep="-+-".join("-"*widths[h] for h in headers)
    print(line); print(sep)
    for row in rows_sorted:
        print(" | ".join(fmt(row.get(h)).ljust(widths[h]) for h in headers))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
        print(f"\nWrote comparison JSON: {args.output_json}")
if __name__=="__main__":
    main()
